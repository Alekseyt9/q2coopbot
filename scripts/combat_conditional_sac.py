"""CUDA-only conditional SAC for aiming under a frozen discrete/motion policy.

The two SAC coordinates are normalized tanh yaw/pitch; aim-mode is part of the
critic condition. This is not a SAC implementation for the whole hybrid policy.
"""
import math
from train_combat_bc import torch, nn
from combat_target_head import distribution as target_distribution
from combat_precision_head import mode_distribution, selected_means
from combat_weapon_head import distribution as weapon_distribution, feature_mask


def exogenous(raw, features, std):
    """Frozen policy draws: target/mode, pose, weapon, fire and movement."""
    target = target_distribution(raw[:, :45], features).sample()
    mode = mode_distribution(raw, target).sample()
    vertical = torch.distributions.Categorical(logits=raw[:, 5:8]).sample()
    weapon = weapon_distribution(raw[:, 8:20], feature_mask(features[:, :845])).sample()
    attack = torch.distributions.Bernoulli(logits=raw[:, 4]).sample().long()
    movement = raw[:, :2] + std[:2].exp() * torch.randn_like(raw[:, :2])
    return torch.stack((target, mode, vertical, weapon, attack), -1), movement


def condition_state(features, history, condition, movement):
    assert features.is_cuda and history.is_cuda and condition.is_cuda and movement.is_cuda
    assert condition.dtype == torch.long and condition.shape == (len(features), 5)
    pieces = [features, history]
    for i, count in enumerate((9, 2, 3, 12)):
        pieces.append(torch.nn.functional.one_hot(condition[:, i], count).float())
    pieces.extend((condition[:, 4:5].float(), movement.tanh()))
    return torch.cat(pieces, -1)


def aim_distribution(raw, features, condition, movement, branch, std):
    means = selected_means(raw, condition[:, 0], condition[:, 1], features, movement, branch, condition[:, 2])[:, 2:]
    return torch.distributions.Normal(means, std[2:].exp())


def behavior_aim_kl(raw,features,condition,movement,branch,std,teacher_branch,teacher_std):
    """KL(current || behavior) at the same frozen policy condition.

    Common tanh and mode scale are invertible per condition and cancel in KL.
    Only the current conditional aim parameters/variance receive gradients.
    """
    with torch.no_grad():reference=aim_distribution(raw,features,condition,movement,teacher_branch,teacher_std)
    current=aim_distribution(raw,features,condition,movement,branch,std)
    return torch.distributions.kl_divergence(current,reference).sum(-1)


def sample_aim(raw, features, condition, movement, branch, std):
    normal = aim_distribution(raw, features, condition, movement, branch, std)
    z = normal.rsample()
    return z.tanh(), squashed_log_probability(normal, z)


def squashed_log_probability(normal, z):
    assert z.is_cuda
    # Stable log(1-tanh(z)^2), including saturated latent samples.
    log_jacobian = 2 * (math.log(2) - z - torch.nn.functional.softplus(-2 * z))
    return (normal.log_prob(z) - log_jacobian).sum(-1)


class TwinQ(nn.Module):
    def __init__(self, state_width):
        super().__init__()
        def network():
            return nn.Sequential(nn.Linear(state_width + 2, 256, device='cuda'), nn.ReLU(),
                nn.Linear(256, 128, device='cuda'), nn.ReLU(), nn.Linear(128, 1, device='cuda'))
        self.first, self.second = network(), network()

    def forward(self, state, action):
        x = torch.cat((state, action), -1)
        return self.first(x).squeeze(-1), self.second(x).squeeze(-1)


def soft_backup(reward, bootstrap, first, second, log_probability, gamma, alpha):
    assert all(x.is_cuda for x in (reward, bootstrap, first, second, log_probability))
    return reward + gamma * bootstrap.float() * (torch.minimum(first, second) - alpha * log_probability)


def polyak_update(current, target, retention):
    with torch.no_grad():
        for source, destination in zip(current.parameters(), target.parameters()):
            destination.mul_(retention).add_(source, alpha=1-retention)

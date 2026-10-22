"""Train the sampled-motion aim branch and aim variance; CUDA only.

All categorical logits, movement, spatial modes and history encoder are frozen.
The runtime contract is unchanged. The shared std Adam clock advances, while its
movement coordinates and moments are restored after every proposed step.
"""
from train_combat_bc import torch


def freeze_conditional_aim(actor, std, optimizer):
    assert std.is_cuda and std.shape == (4,)
    branch = actor.sampled_motion_aim_branch
    assert branch is not None
    for parameter in actor.parameters(): parameter.requires_grad_(False)
    for parameter in branch.parameters(): parameter.requires_grad_(True)
    initial = std[:2].detach().clone()
    state = optimizer.state.get(std, {})
    moments = {key: value[:2].detach().clone() for key, value in state.items()
               if torch.is_tensor(value) and value.shape == std.shape}
    std.register_hook(lambda grad: torch.cat((torch.zeros_like(grad[:2]), grad[2:])))

    def project():
        std.clamp_(-8, 1)
        std[:2].copy_(initial)
        for key, value in optimizer.state.get(std, {}).items():
            if torch.is_tensor(value) and value.shape == std.shape:
                if key in moments: value[:2].copy_(moments[key])
                else: value[:2].zero_()

    return project, sum(p.numel() for p in branch.parameters()) + 2

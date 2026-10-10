"""Experimental shared target residual; not enabled by the runtime yet.

Inputs contain observed features only. Target availability remains the policy's
existing mask; remembered enemies never become selectable through this module.
"""
from train_combat_bc import torch, nn
from combat_spatial_aim import inputs as spatial_inputs

VERSION = 'combat_shared_target_residual_v1'
INPUT_WIDTH = 219  # spatial119 + previous1 + navigation27 + memory60 + group12


def masked_summary(values):
    present = values[..., :1] > 0
    count = present.sum(-2).clamp_min(1)
    mean = torch.where(present, values, 0.).sum(-2) / count
    maximum = values.masked_fill(~present, -torch.inf).amax(-2)
    maximum = torch.where(present.any(-2), maximum, 0.)
    return mean, maximum


def inputs(features, raw):
    if features.shape[-1] != 1121 or raw.shape[-1] != 81:
        raise ValueError('shared target requires v9 features and precision outputs')
    enemy = features[..., 73:169].reshape(*features.shape[:-1], 8, 12)
    memory = features[..., 881:1121].reshape(*features.shape[:-1], 8, 30)
    memory_mean, memory_max = masked_summary(memory)
    group_mean, _ = masked_summary(enemy)
    common = torch.cat((features[..., 854:881], memory_mean,
                        memory_max, group_mean), -1)
    common = common.unsqueeze(-2).expand(*features.shape[:-1], 8, 99)
    # Index845 is previous "none";846:854 follows the eight current enemy slots.
    previous = features[..., 846:854].unsqueeze(-1)
    result = torch.cat((spatial_inputs(features, raw), previous, common), -1)
    if result.shape[-1] != INPUT_WIDTH or not bool(torch.isfinite(result).all()):
        raise ValueError('nonfinite or malformed shared target inputs')
    return result


def initialize():
    branch = nn.Sequential(nn.Linear(INPUT_WIDTH, 64, device='cuda'), nn.ReLU(),
                           nn.Linear(64, 32, device='cuda'), nn.ReLU(),
                           nn.Linear(32, 1, device='cuda'))
    with torch.no_grad():
        branch[-1].weight.zero_()
        branch[-1].bias.zero_()
    return branch


def apply(branch, features, raw):
    residual = branch(inputs(features, raw)).squeeze(-1)
    result = raw.clone()
    result[..., 21:29] = result[..., 21:29] + residual
    return result

"""Observed SG fine-only learned residual; frozen legacy path outside the gate."""
import copy
import math
from train_combat_bc import torch, nn

VERSION = 'combat_observed_shotgun_fine_aim_v1'
INPUT_WIDTH = 153


def validate(model):
    s = model.get('fine_weapon_aim')
    if s is None:
        return
    assert s['version'] == VERSION and model.get('sampled_motion_aim', {}).get('version') == 'combat_sampled_motion_aim_v1'
    assert model.get('spatial_aim') and model.get('aim_mode_head') == 'combat_target_coarse_fine_aim_v1'
    assert len(s['layers']) == 3
    width = INPUT_WIDTH
    for layer, output in zip(s['layers'], (32, 16, 2)):
        assert len(layer['weight']) == len(layer['bias']) == output
        assert all(len(row) == width and all(math.isfinite(v) and abs(v) <= 1e4 for v in row+[bias])
                   for row, bias in zip(layer['weight'], layer['bias']))
        width = output


def initialize(model):
    assert not model.get('fine_weapon_aim') and torch.cuda.is_available()
    from ppo_combat import layers
    head = nn.Sequential(nn.Linear(INPUT_WIDTH, 32, device='cuda'), nn.ReLU(),
                         nn.Linear(32, 16, device='cuda'), nn.ReLU(), nn.Linear(16, 2, device='cuda'))
    with torch.no_grad():
        head[-1].weight.zero_(); head[-1].bias.zero_()
    result = copy.deepcopy(model)
    result['fine_weapon_aim'] = dict(version=VERSION, layers=layers(head))
    validate(result)
    return result


def build(model):
    validate(model)
    if not model.get('fine_weapon_aim'):
        return None
    from ppo_combat import network
    return network(model['fine_weapon_aim']['layers'])


def export(model, head):
    if head is not None:
        from ppo_combat import layers
        model['fine_weapon_aim'] = dict(version=VERSION, layers=layers(head))
        validate(model)


def active(features, target, mode):
    assert features.is_cuda and target.is_cuda and mode.is_cuda and features.shape[1] >= 1147
    slot = (target.clamp_min(1)-1)[:, None]
    visible = features[:, 73:169].reshape(-1, 8, 12)[:, :, 0].gather(1, slot)[:, 0]
    available = features[:, 426:466].reshape(-1, 8, 5)[:, :, 0].gather(1, slot)[:, 0]
    return (features[:, 0] > 0) & features[:, 1123].eq(1) & target.gt(0) & mode.eq(1) & visible.eq(1) & available.eq(1)


def inputs(features, raw, movement, target, mode, legacy_aim, reference_branch=None):
    from combat_sampled_motion_aim import inputs as motion_inputs
    if reference_branch is not None:
        from combat_motor_residual import reference
        raw = reference(reference_branch, features, raw)
    return torch.cat((motion_inputs(features, raw, movement, target, mode),
                      features[:, 1121:1147], legacy_aim.clamp(-8, 8)), -1)


def apply(head, features, raw, movement, target, mode, legacy_aim, reference_branch=None):
    if head is None:
        return legacy_aim
    gate = active(features, target, mode)
    x = inputs(features, raw, movement, target, mode, legacy_aim, reference_branch)
    delta = head(x)
    assert delta.shape == legacy_aim.shape and bool(torch.isfinite(delta).all())
    # The false branch returns legacy values directly, not an approximate penalty.
    return torch.where(gate[:, None], legacy_aim+delta, legacy_aim)

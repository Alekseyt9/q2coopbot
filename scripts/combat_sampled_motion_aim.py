"""Autoregressive yaw/pitch residual conditioned on sampled movement (CUDA)."""
import copy
import math
from train_combat_bc import torch, nn

VERSION = 'combat_sampled_motion_aim_v1'
VERTICAL_VERSION = 'combat_sampled_motion_vertical_aim_v2'
VERTICAL_CONTROL_VERSION = 'combat_sampled_motion_vertical_ablated_v2'
INPUT_WIDTH = 125


def validate(model):
    spec = model.get('sampled_motion_aim')
    if spec is None: return
    assert spec['version'] in (VERSION, VERTICAL_VERSION, VERTICAL_CONTROL_VERSION) and model['feature_version'] == 'combat_features_v10'
    assert model.get('spatial_aim') and model.get('aim_mode_head') == 'combat_target_coarse_fine_aim_v1'
    width = INPUT_WIDTH if spec['version'] == VERSION else INPUT_WIDTH + 3
    assert len(spec['layers']) == 3
    for layer, output in zip(spec['layers'], (64, 32, 2)):
        assert len(layer['weight']) == len(layer['bias']) == output
        for row, bias in zip(layer['weight'], layer['bias']):
            assert len(row) == width and all(math.isfinite(v) and abs(v) <= 1e4 for v in row+[bias])
        width = output


def initialize(model):
    assert not model.get('sampled_motion_aim') and torch.cuda.is_available()
    from ppo_combat import layers
    module = nn.Sequential(nn.Linear(INPUT_WIDTH, 64, device='cuda'), nn.ReLU(),
        nn.Linear(64, 32, device='cuda'), nn.ReLU(), nn.Linear(32, 2, device='cuda'))
    with torch.no_grad(): module[-1].weight.zero_(); module[-1].bias.zero_()
    result = copy.deepcopy(model)
    result['sampled_motion_aim'] = dict(version=VERSION, layers=layers(module))
    validate(result)
    return result


def build(model):
    validate(model)
    if not model.get('sampled_motion_aim'): return None
    from ppo_combat import network
    module = network(model['sampled_motion_aim']['layers'])
    module.motion_version = model['sampled_motion_aim']['version']
    return module


def export(model, module):
    if module is None: return
    from ppo_combat import layers
    model['sampled_motion_aim'] = dict(version=model['sampled_motion_aim']['version'], layers=layers(module))
    validate(model)


def inputs(features, raw, movement, target, mode):
    from combat_spatial_aim import inputs as spatial_inputs
    assert features.is_cuda and raw.is_cuda and movement.is_cuda
    assert raw.ndim == 2 and raw.shape[1] == 81 and features.shape == (len(raw), 1147)
    assert movement.shape == (len(raw), 2) and target.shape == mode.shape == (len(raw),)
    assert target.dtype == mode.dtype == torch.long
    assert bool(((target >= 0) & (target <= 8)).all()) and bool(((mode >= 0) & (mode <= 1)).all())
    assert bool(torch.isfinite(movement).all())
    spatial = spatial_inputs(features, raw)
    chosen = spatial.gather(1, (target.clamp_min(1)-1)[:, None, None].expand(-1, 1, 119))[:, 0]
    physical = movement.tanh()
    return torch.cat((chosen, physical, physical-raw[:, :2].tanh(),
        torch.nn.functional.one_hot(mode, 2).to(raw.dtype)), -1)


def uses_vertical(module):
    return module is not None and module[0].in_features == INPUT_WIDTH + 3


def correction(module, features, raw, movement, target, mode, vertical=None, reference_branch=None):
    if getattr(module,'requires_motor_reference',False):
        assert reference_branch is not None,'Motor policy requires an explicit reference branch'
    if reference_branch is not None:
        from combat_motor_residual import reference
        raw=reference(reference_branch,features,raw)
    x = inputs(features, raw, movement, target, mode)
    if uses_vertical(module):
        assert vertical is not None and vertical.is_cuda and vertical.dtype == torch.long
        assert vertical.shape == (len(raw),) and bool(((vertical >= 0) & (vertical < 3)).all())
        extra = torch.nn.functional.one_hot(vertical, 3).to(raw.dtype)
        if module.motion_version == VERTICAL_CONTROL_VERSION:
            extra = torch.zeros_like(extra)
        x = torch.cat((x, extra), -1)
    residual = module(x)
    assert residual.shape == (len(raw), 2) and bool(torch.isfinite(residual).all())
    return torch.where((target > 0)[:, None], residual, torch.zeros_like(residual))

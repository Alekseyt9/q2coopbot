"""Optional learned movement residual. All neural computation runs on CUDA."""
import math
from combat_memory_route_features import VERSION as FEATURE_VERSION, PARENT_WIDTH, WIDTH

VERSION = 'combat_memory_route_movement_v1'
YAW_VERSION = 'combat_memory_route_movement_yaw_v2'
INPUT_WIDTH = 323


def validate(model):
    spec = model.get('memory_route_movement')
    if spec is None:
        assert model['feature_version'] != FEATURE_VERSION, 'Route features require movement branch'
        return
    assert model['feature_version'] == FEATURE_VERSION and spec['version'] in (VERSION,YAW_VERSION)
    assert model.get('spatial_aim') and not any(model.get(k) for k in ('memory_recovery', 'movement_inertia', 'entity_attention'))
    assert set(spec) == {'version', 'layers'} and len(spec['layers']) == 3
    width = INPUT_WIDTH
    for layer, out in zip(spec['layers'], (32, 16, 3 if spec['version']==YAW_VERSION else 2)):
        assert set(layer) == {'weight', 'bias'} and len(layer['weight']) == len(layer['bias']) == out
        assert all(len(row) == width and all(math.isfinite(v) and abs(v) <= 1e4 for v in [*row, bias]) for row, bias in zip(layer['weight'], layer['bias']))
        width = out


def inputs(features, raw):
    from ppo_combat import torch
    assert features.is_cuda and raw.is_cuda and features.shape[:-1] == raw.shape[:-1]
    assert features.shape[-1] == WIDTH and raw.shape[-1] == 81
    routes = features[..., PARENT_WIDTH+1:].reshape(*features.shape[:-1], 8, 8)
    memory = features[..., 881:1121].reshape(*features.shape[:-1], 8, 30)
    visible = features[..., 73:169].reshape(*features.shape[:-1], 8, 12)[..., 0].ne(0).any(-1)
    found = ((memory[..., 0] == 1) & (routes[..., 0] == 1) & (routes[..., 1] == 1)).any(-1)
    active = (features[..., 0] > 0) & (features[..., PARENT_WIDTH] == 1) & ~visible & found
    x = torch.cat((features[..., :13], features[..., 881:1121], features[..., PARENT_WIDTH:], raw[..., :5].clamp(-20, 20)), -1)
    assert x.shape[-1] == INPUT_WIDTH and bool(torch.isfinite(x).all())
    return x, active


def build(model):
    validate(model)
    if model.get('memory_route_movement') is None:
        return None
    from ppo_combat import network
    return network(model['memory_route_movement']['layers']).to('cuda')


def apply(head, features, raw):
    from ppo_combat import torch
    x, active = inputs(features, raw)
    correction = head(x)
    result = raw.clone()
    result[..., :2] += correction[..., :2]*active[..., None]
    if correction.shape[-1]==3:
        result[..., 2] += correction[..., 2]*active
    assert bool(torch.isfinite(result).all())
    return result


def export(model, head):
    if head is None:
        assert model.get('memory_route_movement') is None
        return
    from ppo_combat import layers
    version=model.get('memory_route_movement',{}).get('version',VERSION)
    model['memory_route_movement'] = dict(version=version, layers=layers(head))
    validate(model)

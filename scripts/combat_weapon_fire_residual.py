"""SG-only learned fire correction; no hidden outcomes or firing rules."""
import copy
from train_combat_bc import torch, nn
from combat_conditional_fire import INPUT_WIDTH, VERSION as BASE_VERSION

VERSION = 'combat_observed_shotgun_fire_residual_v1'


def validate(model):
    spec = model.get('weapon_fire_residual')
    if spec is None:
        return
    assert spec['version'] == VERSION and model.get('conditional_fire')
    from combat_conditional_fire import validate as validate_base
    changed = copy.copy(model)
    changed['conditional_fire'] = dict(version=BASE_VERSION, layers=spec['layers'])
    validate_base(changed)


def initialize(model):
    assert not model.get('weapon_fire_residual') and model.get('conditional_fire')
    from ppo_combat import layers
    head = nn.Sequential(nn.Linear(INPUT_WIDTH, 32, device='cuda'), nn.ReLU(),
                         nn.Linear(32, 16, device='cuda'), nn.ReLU(), nn.Linear(16, 1, device='cuda'))
    with torch.no_grad():
        head[-1].weight.zero_(); head[-1].bias.zero_()
    result = copy.deepcopy(model)
    result['weapon_fire_residual'] = dict(version=VERSION, layers=layers(head))
    validate(result)
    return result


def build(model):
    validate(model)
    if not model.get('weapon_fire_residual'):
        return None
    from ppo_combat import network
    return network(model['weapon_fire_residual']['layers'])


def export(model, head):
    if head is not None:
        from ppo_combat import layers
        model['weapon_fire_residual'] = dict(version=VERSION, layers=layers(head))
        validate(model)


def active(features):
    assert features.is_cuda and features.shape[1] >= 1147
    return features[:, 0].gt(0) & features[:, 1123].eq(1)


def apply(head, features, inputs, legacy):
    if head is None:
        return legacy
    delta = head(inputs)[:, 0]
    assert bool(torch.isfinite(delta).all())
    return torch.where(active(features), legacy+delta, legacy)

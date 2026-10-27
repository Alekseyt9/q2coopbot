"""CUDA fire residual on the actual chosen action.

Recorded action context is detached: this helper must not create extra pathwise
gradients into movement or sampled aim.
"""
from combat_body_features import VERSIONS as BODY_FEATURE_VERSIONS, WIDTH as BODY_FEATURE_WIDTH, legacy as body_legacy
import copy
import math
from train_combat_bc import torch, nn

VERSION = 'combat_action_conditioned_fire_v1'
INPUT_WIDTH = 181
CONTEXT_VERSION = 'combat_action_conditioned_fire_context_v2'
CONTEXT_WIDTH = INPUT_WIDTH + 49 + 37


def validate(model):
    spec = model.get('conditional_fire')
    if spec is None:
        return
    assert model['feature_version'] in BODY_FEATURE_VERSIONS
    assert model.get('aim_mode_head') == 'combat_target_coarse_fine_aim_v1'
    assert model.get('spatial_aim') and model.get('sampled_motion_aim') and not model.get('entity_attention')
    assert spec['version'] in (VERSION, CONTEXT_VERSION) and len(spec['layers']) == 3
    width = CONTEXT_WIDTH if spec['version']==CONTEXT_VERSION else INPUT_WIDTH
    for layer, output in zip(spec['layers'], (32, 16, 1)):
        assert len(layer['bias']) == len(layer['weight']) == output
        for row, bias in zip(layer['weight'], layer['bias']):
            assert len(row) == width and all(math.isfinite(v) and abs(v) <= 1e4 for v in [*row, bias])
        width = output


def inputs(features, target, mode, latent, vertical, weapon):
    features = body_legacy(features)
    assert isinstance(features, torch.Tensor) and features.is_cuda
    assert features.ndim == 2 and features.shape[1] == 1147
    n = len(features)
    assert n > 0 and bool(torch.isfinite(features).all())
    for value, categories in ((target, 9), (mode, 2), (vertical, 3), (weapon, 12)):
        assert isinstance(value, torch.Tensor) and value.is_cuda and value.dtype == torch.long
        assert value.shape == (n,) and bool(((value >= 0) & (value < categories)).all())
    assert isinstance(latent, torch.Tensor) and latent.is_cuda and latent.shape == (n, 4)
    assert bool(torch.isfinite(latent).all())
    from combat_target_head import availability
    from combat_weapon_head import feature_mask
    assert bool(availability(features).gather(1, target[:, None]).all()), 'Unavailable selected target'
    assert bool(feature_mask(features[:, :845]).gather(1, weapon[:, None]).all()), 'Unavailable weapon request'
    x = features.detach()
    slot = (target - 1).clamp_min(0)
    present = (target > 0)[:, None]
    enemy = x[:, 73:169].reshape(n, 8, 12).gather(1, slot[:, None, None].expand(-1, 1, 12))[:, 0] * present
    typed = x[:, 466:786].reshape(n, 8, 40).gather(1, slot[:, None, None].expand(-1, 1, 40))[:, 0] * present
    action = latent.detach().tanh()
    scale = torch.where(mode == 1, 15. / 180., 1.).to(action.dtype)
    physical = torch.cat((action[:, :2], action[:, 2:] * scale[:, None]), -1)
    onehot = lambda value, count: nn.functional.one_hot(value, count).to(x.dtype)
    result = torch.cat((x[:, :73], x[:, 1121:1147], enemy, typed,
        onehot(target, 9), onehot(mode, 2), physical.to(x.dtype),
        onehot(vertical, 3), onehot(weapon, 12)), -1)
    assert result.shape == (n, INPUT_WIDTH) and bool(torch.isfinite(result).all())
    return result


def logits(module, features, raw, target, mode, latent, vertical, weapon):
    assert raw.is_cuda and raw.ndim == 2 and raw.shape == (len(features), 81)
    if module is None:
        return raw[:, 4]
    x = inputs(features, target, mode, latent, vertical, weapon)
    # Keep the legacy 181 input contract used by action geometry and the
    # separate SG residual. Only the versioned main fire branch is expanded.
    branch_input = context_inputs(features,x) if module[0].in_features==CONTEXT_WIDTH else x
    delta = module(branch_input)
    assert delta.shape == (len(raw), 1) and bool(torch.isfinite(delta).all())
    result = raw[:, 4] + delta[:, 0].to(raw.dtype)
    from combat_weapon_fire_residual import apply as apply_weapon_fire
    result = apply_weapon_fire(getattr(module, 'weapon_fire_residual_branch', None), features, x, result)
    assert bool(torch.isfinite(result).all())
    return result


def context_inputs(features, fire_inputs):
    x=body_legacy(features)
    assert fire_inputs.is_cuda and fire_inputs.shape==(len(x),INPUT_WIDTH)
    result=torch.cat((fire_inputs,x[:,169:218].detach(),x[:,259:296].detach()),-1)
    assert result.shape==(len(x),CONTEXT_WIDTH) and bool(torch.isfinite(result).all())
    return result


def build(model):
    validate(model)
    if model.get('conditional_fire') is None:
        return None
    assert torch.cuda.is_available()
    from ppo_combat import network
    return network(model['conditional_fire']['layers']).to('cuda')


def export(model, module):
    if module is None:
        assert not model.get('conditional_fire')
        return
    from ppo_combat import layers
    width=module[0].in_features
    assert width in (INPUT_WIDTH,CONTEXT_WIDTH)
    model['conditional_fire'] = dict(version=CONTEXT_VERSION if width==CONTEXT_WIDTH else VERSION, layers=layers(module))
    validate(model)


def initialize(model):
    assert torch.cuda.is_available() and not model.get('conditional_fire')
    result = copy.deepcopy(model)
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(20261023)
        module = nn.Sequential(nn.Linear(INPUT_WIDTH, 32, device='cuda'), nn.ReLU(),
            nn.Linear(32, 16, device='cuda'), nn.ReLU(), nn.Linear(16, 1, device='cuda'))
        with torch.no_grad():
            module[-1].weight.zero_(); module[-1].bias.zero_()
        export(result, module)
    return result


def freeze_scope(actor, std, optimizer):
    """Train only the new fire branch; preserve old parameters and Adam state."""
    assert actor.conditional_fire_branch is not None
    parameters=list(actor.parameters())+[std]
    selected=set(actor.conditional_fire_branch.parameters())
    optimizer.zero_grad(set_to_none=True)
    frozen={p:p.detach().clone() for p in parameters if p not in selected}
    states={p:copy.deepcopy(optimizer.state.get(p,{})) for p in frozen}
    for p in parameters:
        p.requires_grad_(p in selected)
    def project():
        with torch.no_grad():
            for p,value in frozen.items():
                p.copy_(value)
                if states[p]:
                    optimizer.state[p]=copy.deepcopy(states[p])
                else:
                    optimizer.state.pop(p,None)
    return project,sum(p.numel() for p in selected)

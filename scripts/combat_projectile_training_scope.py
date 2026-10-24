"""Eight-group CUDA input scope retaining the parent's seven Adam groups."""
import copy

from ppo_combat import torch, nn
from combat_decision_scope import DecisionScope, ROW_MODULES, RESIDUALS
from combat_body_features import LEGACY_WIDTH
from combat_projectile_type_features import PARENT_WIDTH, WIDTH
from combat_body_training_scope import RATES as PARENT_RATES

SCOPE = 'combat_projectile_type_input_ppo_v1'
RATES = (*PARENT_RATES, 3e-5)


class ProjectileInputColumns(nn.Module):
    def __init__(self, original):
        super().__init__()
        assert original.in_features == WIDTH and original.weight.is_cuda
        self.original = original
        self.register_buffer('prefix_weight', original.weight[:, :LEGACY_WIDTH].detach().clone())
        self.body = nn.Parameter(original.weight[:, LEGACY_WIDTH:PARENT_WIDTH].detach().clone())
        self.bias = nn.Parameter(original.bias.detach().clone())
        self.projectile = nn.Parameter(original.weight[:, PARENT_WIDTH:].detach().clone())

    def forward(self, features):
        assert features.is_cuda and features.shape[-1] == WIDTH
        return nn.functional.linear(features, torch.cat((self.prefix_weight, self.body, self.projectile), -1), self.bias)


class ProjectileTrainingScope(DecisionScope):
    def __init__(self, actor, std, legacy_optimizer, *, inherited_optimizer=None, saved_optimizer=None):
        assert (inherited_optimizer is None) != (saved_optimizer is None)
        source = copy.deepcopy(inherited_optimizer if inherited_optimizer is not None else saved_optimizer)
        count = 7 if inherited_optimizer is not None else 8
        assert len(source['param_groups']) == count
        initial = copy.deepcopy(source)
        initial['param_groups'] = initial['param_groups'][:5]
        ids = {p for group in initial['param_groups'] for p in group['params']}
        initial['state'] = {k: v for k, v in initial['state'].items() if k in ids}
        super().__init__(actor, std, legacy_optimizer, initial)
        self.input = ProjectileInputColumns(actor.base.encoder[0])
        actor.base.encoder[0] = self.input
        for name, parameter in (('body_input_columns', self.input.body),
                                ('shared_input_bias', self.input.bias),
                                ('projectile_type_input_columns', self.input.projectile)):
            self.optimizer.add_param_group(dict(params=[parameter], lr=3e-5, name=name))
        if inherited_optimizer is not None:
            assert self.optimizer.state_dict()['param_groups'][:7] == source['param_groups']
            source['param_groups'].append(self.optimizer.state_dict()['param_groups'][7])
        self.optimizer.load_state_dict(source)
        assert [group['lr'] for group in self.optimizer.param_groups] == list(RATES)
        self.parameters = [p for group in self.optimizer.param_groups for p in group['params']]
        assert len(set(self.parameters)) == len(self.parameters)

    def materialize(self):
        assert not self.materialized
        with torch.no_grad():
            self.input.original.weight[:, LEGACY_WIDTH:PARENT_WIDTH].copy_(self.input.body)
            self.input.original.weight[:, PARENT_WIDTH:].copy_(self.input.projectile)
            self.input.original.bias.copy_(self.input.bias)
        self.actor.base.encoder[0] = self.input.original
        super().materialize()

    def verify_frozen(self):
        assert self.materialized and torch.equal(self.std, self.std_before)
        current = self.actor.state_dict()
        assert current.keys() == self.before.keys()
        for name, value in current.items():
            original = self.before[name]
            if name.startswith(tuple(module + '.' for module in RESIDUALS)) or name == 'base.encoder.0.bias':
                continue
            if name == 'base.encoder.0.weight':
                assert torch.equal(value[:, :LEGACY_WIDTH], original[:, :LEGACY_WIDTH]), name
            elif name in {f'base.{module}.{field}' for module in ROW_MODULES for field in ('weight', 'bias')}:
                assert torch.equal(value[:4], original[:4]) and torch.equal(value[29:], original[29:]), name
            else:
                assert torch.equal(value, original), name

    def metadata(self, updates):
        result = super().metadata(updates)
        result.update(version=SCOPE, group_lrs=list(RATES), input_prefix=LEGACY_WIDTH,
            parent_width=PARENT_WIDTH, input_width=WIDTH, shared_input_bias=True,
            projectile_columns=[PARENT_WIDTH, WIDTH], updates_completed=updates)
        return result


def widen_value_optimizer(source, value):
    """Preserve every old moment and clock, append zero input moments on CUDA."""
    result = copy.deepcopy(source)
    assert len(result['param_groups']) == 1
    names = list(value.named_parameters())
    ids = result['param_groups'][0]['params']
    assert len(ids) == len(names)
    widened = []
    for identifier, (name, parameter) in zip(ids, names):
        state = result['state'][identifier]
        for key, tensor in list(state.items()):
            if not isinstance(tensor, torch.Tensor) or key == 'step':
                continue
            assert tensor.is_cuda
            if name == 'encoder.0.weight':
                assert tensor.shape == (parameter.shape[0], PARENT_WIDTH)
                expanded = torch.cat((tensor, tensor.new_zeros((len(tensor), WIDTH - PARENT_WIDTH))), -1)
                assert torch.equal(expanded[:, :PARENT_WIDTH], tensor)
                state[key] = expanded
                widened.append(key)
            else:
                assert tensor.shape == parameter.shape, name
    assert set(widened) == {'exp_avg', 'exp_avg_sq'}
    return result

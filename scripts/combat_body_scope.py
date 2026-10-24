"""Dedicated CUDA optimizer for appended body inputs plus inherited decision rows."""
import copy

from ppo_combat import torch, nn
from combat_body_features import LEGACY_WIDTH, WIDTH, SCOPE
from combat_decision_scope import DecisionScope, RATES, RESIDUALS, ROW_MODULES

BODY_RATE = 3e-5
BODY_RATES = (*RATES, BODY_RATE)


class BodyInputColumns(nn.Module):
    def __init__(self, original):
        super().__init__()
        assert isinstance(original, nn.Linear) and original.in_features == WIDTH and original.weight.is_cuda
        self.original = original
        self.register_buffer('prefix_weight', original.weight[:, :LEGACY_WIDTH].detach().clone())
        self.register_buffer('prefix_bias', original.bias.detach().clone())
        self.weight = nn.Parameter(original.weight[:, LEGACY_WIDTH:].detach().clone())

    def forward(self, features):
        assert features.is_cuda and features.shape[-1] == WIDTH
        return nn.functional.linear(features[..., :LEGACY_WIDTH], self.prefix_weight, self.prefix_bias) + nn.functional.linear(features[..., LEGACY_WIDTH:], self.weight)


class BodyScope(DecisionScope):
    def __init__(self, actor, std, legacy_optimizer, *, inherited_optimizer=None, saved_optimizer=None):
        assert (inherited_optimizer is None) != (saved_optimizer is None)
        initial = inherited_optimizer
        if saved_optimizer is not None:
            assert len(saved_optimizer['param_groups']) == 6
            initial = copy.deepcopy(saved_optimizer)
            initial['param_groups'] = initial['param_groups'][:5]
            ids = {parameter for group in initial['param_groups'] for parameter in group['params']}
            initial['state'] = {key: value for key, value in initial['state'].items() if key in ids}
        super().__init__(actor, std, legacy_optimizer, initial)
        self.input = BodyInputColumns(actor.base.encoder[0])
        actor.base.encoder[0] = self.input
        self.optimizer.add_param_group(dict(params=[self.input.weight], lr=BODY_RATE, name='body_input_columns'))
        if saved_optimizer is not None:
            self.optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
        assert [group['lr'] for group in self.optimizer.param_groups] == list(BODY_RATES)
        self.parameters = [parameter for group in self.optimizer.param_groups for parameter in group['params']]
        assert len(set(self.parameters)) == len(self.parameters)

    def materialize(self):
        assert not self.materialized
        with torch.no_grad():
            self.input.original.weight[:, LEGACY_WIDTH:].copy_(self.input.weight)
            self.actor.base.encoder[0] = self.input.original
        super().materialize()

    def verify_frozen(self):
        assert self.materialized and torch.equal(self.std, self.std_before)
        current = self.actor.state_dict()
        assert current.keys() == self.before.keys()
        for name, value in current.items():
            original = self.before[name]
            if name.startswith(tuple(module+'.' for module in RESIDUALS)):
                continue
            if name == 'base.encoder.0.weight':
                assert torch.equal(value[:, :LEGACY_WIDTH], original[:, :LEGACY_WIDTH]), name
            elif name in {f'base.{module}.{field}' for module in ROW_MODULES for field in ('weight', 'bias')}:
                assert torch.equal(value[:4], original[:4]) and torch.equal(value[29:], original[29:]), name
            else:
                assert torch.equal(value, original), name

    def metadata(self, updates):
        result = super().metadata(updates)
        result.update(version=SCOPE, input_prefix=LEGACY_WIDTH, input_width=WIDTH,
                      group_lrs=list(BODY_RATES), body_learning_rate=BODY_RATE,
                      fine_coordinates='body-conditional encoder effect allowed; original coordinate rows frozen')
        return result

"""Symmetric body-input PPO scope: shared encoder bias is trainable in both arms."""
import copy

from ppo_combat import torch, nn
from combat_body_scope import BodyScope, BodyInputColumns, BODY_RATES
from combat_body_features import LEGACY_WIDTH
from combat_decision_scope import RESIDUALS, ROW_MODULES

SCOPE = 'combat_body_input_ppo_v2'
RATES = (*BODY_RATES, 3e-5)


class BodyTrainingColumns(BodyInputColumns):
    def forward(self, features):
        # Use the same contiguous matrix operation as exported inference.
        # Only the appended columns remain trainable through this concat.
        weight = torch.cat((self.prefix_weight, self.weight), -1)
        return nn.functional.linear(features, weight, self.prefix_bias)


class BodyTrainingScope(BodyScope):
    def __init__(self, actor, std, legacy_optimizer, *, inherited_optimizer=None, saved_optimizer=None):
        assert (inherited_optimizer is None) != (saved_optimizer is None)
        initial = inherited_optimizer
        if saved_optimizer is not None:
            assert len(saved_optimizer['param_groups']) == 7
            initial = copy.deepcopy(saved_optimizer)
            initial['param_groups'] = initial['param_groups'][:6]
            ids = {parameter for group in initial['param_groups'] for parameter in group['params']}
            initial['state'] = {key: value for key, value in initial['state'].items() if key in ids}
        super().__init__(actor, std, legacy_optimizer, saved_optimizer=initial)
        previous = self.input
        self.input = BodyTrainingColumns(previous.original)
        self.input.weight = previous.weight  # retain inherited Adam binding
        actor.base.encoder[0] = self.input
        bias = self.input.prefix_bias.detach().clone()
        del self.input._buffers['prefix_bias']
        self.input.register_parameter('prefix_bias', nn.Parameter(bias))
        self.optimizer.add_param_group(dict(params=[self.input.prefix_bias], lr=3e-5, name='shared_input_bias'))
        if saved_optimizer is not None:
            self.optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
        assert [group['lr'] for group in self.optimizer.param_groups] == list(RATES)
        self.parameters = [parameter for group in self.optimizer.param_groups for parameter in group['params']]
        assert len(set(self.parameters)) == len(self.parameters)

    def materialize(self):
        assert not self.materialized
        with torch.no_grad():
            self.input.original.bias.copy_(self.input.prefix_bias)
        super().materialize()

    def verify_frozen(self):
        assert self.materialized and torch.equal(self.std, self.std_before)
        current = self.actor.state_dict()
        assert current.keys() == self.before.keys()
        for name, value in current.items():
            original = self.before[name]
            if name.startswith(tuple(module+'.' for module in RESIDUALS)) or name == 'base.encoder.0.bias':
                continue
            if name == 'base.encoder.0.weight':
                assert torch.equal(value[:, :LEGACY_WIDTH], original[:, :LEGACY_WIDTH]), name
            elif name in {f'base.{module}.{field}' for module in ROW_MODULES for field in ('weight', 'bias')}:
                assert torch.equal(value[:4], original[:4]) and torch.equal(value[29:], original[29:]), name
            else:
                assert torch.equal(value, original), name

    def metadata(self, updates):
        result = super().metadata(updates)
        result.update(version=SCOPE, group_lrs=list(RATES), shared_input_bias=True,
                      bias_scope='same input bias group in observed and ablated arms')
        return result

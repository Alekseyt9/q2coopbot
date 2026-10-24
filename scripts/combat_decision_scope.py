"""CUDA decision-row optimizer with separately persisted Adam clocks."""
import copy

from ppo_combat import torch, nn
from combat_checkpoint_scope import DECISION_SCOPE


RESIDUALS = ('weapon_state_branch', 'motor_branch', 'conditional_fire_branch')
ROW_MODULES = ('head', 'residual')
RATES = (3e-5, 3e-5, .0003, .0003, .0003)


class DecisionRows(nn.Module):
    def __init__(self, original):
        super().__init__()
        assert isinstance(original, nn.Linear) and original.out_features == 81 and original.weight.is_cuda
        self.original = original
        self.weight = nn.Parameter(original.weight[4:29].detach().clone())
        self.bias = nn.Parameter(original.bias[4:29].detach().clone())

    def forward(self, inputs):
        result = self.original(inputs).clone()
        result[..., 4:29] = nn.functional.linear(inputs, self.weight, self.bias)
        return result


class DecisionScope:
    def __init__(self, actor, std, legacy_optimizer, saved_optimizer=None):
        assert next(actor.parameters()).is_cuda and std.is_cuda
        self.actor, self.std = actor, std
        self.before = {name: tensor.detach().clone() for name,tensor in actor.state_dict().items()}
        self.std_before = std.detach().clone()
        self.legacy_optimizer = legacy_optimizer
        self.legacy_before = copy.deepcopy(legacy_optimizer.state_dict())
        actor.requires_grad_(False)
        std.requires_grad_(False)
        self.wrappers, groups, states = {}, [], {}
        self.materialized = False
        for name in ROW_MODULES:
            original = getattr(actor.base, name)
            wrapper = DecisionRows(original)
            setattr(actor.base, name, wrapper)
            self.wrappers[name] = wrapper
            parameters = [wrapper.weight, wrapper.bias]
            groups.append(dict(params=parameters, lr=3e-5, name='decision_rows_' + name))
            for source, destination in zip((original.weight,original.bias), parameters):
                states[destination] = {key:(value[4:29].clone() if isinstance(value,torch.Tensor) and value.shape==source.shape else copy.deepcopy(value))
                                       for key,value in legacy_optimizer.state.get(source,{}).items()}
        for name in RESIDUALS:
            module = getattr(actor,name)
            assert module is not None
            module.requires_grad_(True)
            parameters = list(module.parameters())
            groups.append(dict(params=parameters,lr=.0003,name=name))
            for parameter in parameters:
                states[parameter] = copy.deepcopy(legacy_optimizer.state.get(parameter,{}))
        self.optimizer = torch.optim.Adam(groups)
        if saved_optimizer is None:
            for parameter,state in states.items():
                if state:
                    self.optimizer.state[parameter] = state
        else:
            assert len(saved_optimizer['param_groups']) == 5
            assert [group['name'] for group in saved_optimizer['param_groups']] == [group['name'] for group in groups]
            self.optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
        assert [group['lr'] for group in self.optimizer.param_groups] == list(RATES)
        self.parameters = [parameter for group in self.optimizer.param_groups for parameter in group['params']]
        assert len(set(self.parameters)) == len(self.parameters)

    def materialize(self):
        assert not self.materialized
        with torch.no_grad():
            for name,wrapper in self.wrappers.items():
                wrapper.original.weight[4:29].copy_(wrapper.weight)
                wrapper.original.bias[4:29].copy_(wrapper.bias)
                setattr(self.actor.base,name,wrapper.original)
        self.materialized = True

    def verify_frozen(self):
        assert self.materialized
        assert torch.equal(self.std,self.std_before)
        current = self.actor.state_dict()
        assert current.keys() == self.before.keys()
        for name,value in current.items():
            original = self.before[name]
            if name.startswith(tuple(module+'.' for module in RESIDUALS)):
                continue
            if name in {f'base.{module}.{field}' for module in ROW_MODULES for field in ('weight','bias')}:
                assert torch.equal(value[:4],original[:4]) and torch.equal(value[29:],original[29:]), name
            else:
                assert torch.equal(value,original), name

    def metadata(self, updates):
        return dict(version=DECISION_SCOPE, rows=[4,29], row_modules=list(ROW_MODULES), residuals=list(RESIDUALS),
                    group_lrs=list(RATES), updates_completed=updates, legacy_optimizer_role='frozen_anchor_only')

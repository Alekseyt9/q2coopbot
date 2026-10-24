"""Four-group categorical optimizer; inherited movement and optimizer anchors stay exact."""
import copy

from ppo_combat import torch
from combat_decision_scope import DecisionScope
from fork_conditional_weapon_exploration_cuda import equal_tree


CATEGORICAL_SCOPE = 'combat_categorical_scope_ppo_v1'
RATES = (3e-5, 3e-5, .0003, .0003)
GROUPS = ('decision_rows_head', 'decision_rows_residual', 'weapon_state_branch', 'conditional_fire_branch')


class CategoricalScope(DecisionScope):
    def __init__(self, actor, std, legacy_optimizer, decision_anchor, saved_optimizer=None):
        super().__init__(actor, std, legacy_optimizer, decision_anchor)
        self.decision_anchor_optimizer = self.optimizer
        self.decision_anchor_before = copy.deepcopy(self.optimizer.state_dict())
        self.motor_before = copy.deepcopy(actor.motor_branch.state_dict())
        actor.motor_branch.requires_grad_(False)
        groups = []
        for group in self.optimizer.param_groups:
            if group['name'] != 'motor_branch':
                groups.append(dict({k: copy.deepcopy(v) for k, v in group.items() if k != 'params'}, params=list(group['params'])))
        self.optimizer = torch.optim.Adam(groups)
        if saved_optimizer is None:
            for group in groups:
                for parameter in group['params']:
                    state = self.decision_anchor_optimizer.state.get(parameter)
                    if state:
                        self.optimizer.state[parameter] = copy.deepcopy(state)
        else:
            assert tuple(g['name'] for g in saved_optimizer['param_groups']) == GROUPS
            self.optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
        assert tuple(g['name'] for g in self.optimizer.param_groups) == GROUPS
        assert tuple(g['lr'] for g in self.optimizer.param_groups) == RATES
        self.parameters = [p for group in self.optimizer.param_groups for p in group['params']]

    def verify_frozen(self):
        super().verify_frozen()
        assert equal_tree(self.actor.motor_branch.state_dict(), self.motor_before)
        assert equal_tree(self.decision_anchor_optimizer.state_dict(), self.decision_anchor_before)

    def metadata(self, updates):
        return dict(version=CATEGORICAL_SCOPE, rows=[4, 29], row_modules=['head', 'residual'],
            residuals=['weapon_state_branch', 'conditional_fire_branch'], group_lrs=list(RATES),
            updates_completed=updates, inherited_decision_optimizer_role='frozen_anchor_only',
            legacy_actor_optimizer_role='frozen_anchor_only', motor_frozen=True)

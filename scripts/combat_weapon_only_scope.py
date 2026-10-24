"""CUDA weapon residual scope; all other actor tensors and inherited scopes stay frozen."""
import copy

from ppo_combat import torch
from combat_projectile_training_scope import SCOPE as PARENT_SCOPE
from combat_checkpoint_scope import require_checkpoint_scope

SCOPE = 'combat_observed_weapon_residual_only_ppo_v1'
RATE = .0003


def inherited_optimizer(checkpoint):
    require_checkpoint_scope(checkpoint, PARENT_SCOPE)
    source = checkpoint['projectile_training_optimizer']
    group = copy.deepcopy(source['param_groups'][2])
    assert group['name'] == 'weapon_state_branch' and group['lr'] == RATE
    ids = group['params']
    assert len(ids) == 6
    group.update(params=list(range(len(ids))), name='weapon_residual_only')
    return dict(state={i: copy.deepcopy(source['state'][p]) for i, p in enumerate(ids) if p in source['state']},
                param_groups=[group])


class WeaponOnlyScope:
    def __init__(self, actor, std, saved_optimizer):
        assert next(actor.parameters()).is_cuda and std.is_cuda
        self.actor, self.std = actor, std
        self.before = {k: v.detach().clone() for k, v in actor.state_dict().items()}
        self.std_before = std.detach().clone()
        actor.requires_grad_(False)
        std.requires_grad_(False)
        module = actor.weapon_state_branch
        assert module is not None
        module.requires_grad_(True)
        self.parameters = list(module.parameters())
        assert len(self.parameters) == 6 and len(set(self.parameters)) == 6
        self.optimizer = torch.optim.Adam([dict(params=self.parameters, lr=RATE, name='weapon_residual_only')])
        assert len(saved_optimizer['param_groups']) == 1
        assert saved_optimizer['param_groups'][0]['name'] == 'weapon_residual_only'
        assert saved_optimizer['param_groups'][0]['lr'] == RATE
        assert saved_optimizer['param_groups'][0]['params'] == list(range(6))
        self.optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
        for parameter in self.parameters:
            for key in ('exp_avg', 'exp_avg_sq'):
                value = self.optimizer.state.get(parameter, {}).get(key)
                assert value is None or value.shape == parameter.shape

    def verify_frozen(self):
        assert torch.equal(self.std, self.std_before)
        current = self.actor.state_dict()
        assert current.keys() == self.before.keys()
        for name, value in current.items():
            if not name.startswith('weapon_state_branch.'):
                assert torch.equal(value, self.before[name]), name
        assert all(p.requires_grad == name.startswith('weapon_state_branch.') for name, p in self.actor.named_parameters())

    def metadata(self, updates):
        return dict(version=SCOPE, updates_completed=updates, modules=['weapon_state_branch'],
                    group_lrs=[RATE], other_actor_parameters='frozen_exact',
                    parent_optimizers='retained_frozen_anchors', log_std='frozen_exact')

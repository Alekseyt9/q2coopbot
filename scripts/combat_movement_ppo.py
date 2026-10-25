"""CUDA motor-plus-route PPO scope; BC initialization never imports BC moments."""
import copy
from ppo_combat import torch
from combat_memory_route_ppo import RouteScope
from combat_checkpoint_scope import require_checkpoint_scope
from combat_motor_residual import export as export_motor
from combat_route_task_objective import AUTONOMOUS

SCOPE = 'combat_motor_route_movement_ppo_v1'
CHECKPOINT = 'combat_motor_route_scope_checkpoint_v1'
PREFIXES = ('motor_branch.', 'memory_route_branch.')


class MovementScope(RouteScope):
    def __init__(self, model, *, actor_lr, value_lr):
        super().__init__(model, actor_lr=actor_lr, value_lr=value_lr)
        assert self.actor.motor_branch is not None
        self.actor.motor_branch.requires_grad_(True)
        self.actor_parameters = [p for p in self.actor.parameters() if p.requires_grad]
        self.actor_optimizer = torch.optim.Adam([
            dict(params=list(self.actor.motor_branch.parameters()), name='motor'),
            dict(params=list(self.actor.memory_route_branch.parameters()), name='route')], lr=actor_lr)
        self.frozen = {k:v.detach().clone() for k,v in self.actor.state_dict().items()
                       if not k.startswith(PREFIXES)}
        assert not self.actor_optimizer.state and not self.value_optimizer.state

    def verify_frozen(self):
        current = self.actor.state_dict()
        assert all(torch.equal(current[k],v) for k,v in self.frozen.items()), 'Frozen actor changed'
        assert not self.std.requires_grad and self.std.is_cuda
        assert all(p.requires_grad == name.startswith(PREFIXES) for name,p in self.actor.named_parameters())

    def snapshot(self, behavior_sha256, config, consumed, task_objective=AUTONOMOUS):
        self.verify_frozen()
        assert len(behavior_sha256)==64 and len(consumed)==len(set(consumed))
        return copy.deepcopy(dict(version=CHECKPOINT, checkpoint_training_scope=SCOPE,
            behavior_sha256=behavior_sha256, config=config, consumed=consumed, task_objective=task_objective,
            motor=self.actor.motor_branch.state_dict(), route=self.actor.memory_route_branch.state_dict(),
            value=self.value.state_dict(), actor_optimizer=self.actor_optimizer.state_dict(),
            value_optimizer=self.value_optimizer.state_dict(),
            cuda_rng=torch.cuda.get_rng_state_all(), cpu_rng=torch.get_rng_state()))

    def restore(self, checkpoint, behavior_sha256, config, task_objective=AUTONOMOUS):
        require_checkpoint_scope(checkpoint, SCOPE)
        assert checkpoint['version']==CHECKPOINT and checkpoint['behavior_sha256']==behavior_sha256
        assert checkpoint['config']==config and checkpoint['task_objective']==task_objective
        consumed=checkpoint['consumed']; assert len(consumed)==len(set(consumed))
        self.actor.motor_branch.load_state_dict(checkpoint['motor'])
        self.actor.memory_route_branch.load_state_dict(checkpoint['route'])
        self.value.load_state_dict(checkpoint['value'])
        self.actor_optimizer.load_state_dict(copy.deepcopy(checkpoint['actor_optimizer']))
        self.value_optimizer.load_state_dict(copy.deepcopy(checkpoint['value_optimizer']))
        torch.set_rng_state(checkpoint['cpu_rng'].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng']])
        self.verify_frozen()
        return list(consumed)

    def export(self):
        result=super().export()
        export_motor(result,self.actor.motor_branch)
        assert all(result[k]==v for k,v in self.model.items()
                   if k not in ('motor_residual','memory_route_movement','value','attention'))
        assert result['attention']['actor']==self.model['attention']['actor']
        return result

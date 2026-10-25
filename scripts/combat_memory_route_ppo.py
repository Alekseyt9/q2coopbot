"""Dedicated CUDA optimizer/checkpoint scope for the learned route movement head."""
import copy
from ppo_combat import torch, layers
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model
from combat_memory_route_movement import validate, export as export_routes
from combat_checkpoint_scope import require_checkpoint_scope
from combat_route_task_objective import CONTINUITY

SCOPE = 'combat_memory_route_movement_ppo_v1'
CHECKPOINT = 'combat_memory_route_scope_checkpoint_v1'


class RouteScope:
    def __init__(self, model, *, actor_lr, value_lr):
        validate(model)
        assert model.get('memory_route_movement') and not any(k.startswith('diagnostic_') for k in model)
        assert model.get('attention') and not model.get('memory'), 'Only verified temporal attention migration is supported'
        assert torch.cuda.is_available() and 0 < actor_lr <= .01 and 0 < value_lr <= .01
        self.model = copy.deepcopy(model)
        self.actor = branch(base_model(model, allow_shared=True), 'actor')
        self.value = branch(model, 'value')
        self.actor.requires_grad_(False)
        self.actor.memory_route_branch.requires_grad_(True)
        self.value.requires_grad_(True)
        self.std = torch.tensor(model['log_std'], device='cuda').detach()
        self.actor_parameters = list(self.actor.memory_route_branch.parameters())
        self.actor_optimizer = torch.optim.Adam(self.actor_parameters, lr=actor_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=value_lr)
        self.frozen = {k:v.detach().clone() for k,v in self.actor.state_dict().items() if not k.startswith('memory_route_branch.')}
        assert all(p.is_cuda for module in (self.actor,self.value) for p in module.parameters())

    def verify_frozen(self):
        current = self.actor.state_dict()
        assert all(torch.equal(current[k],v) for k,v in self.frozen.items()), 'Frozen actor changed'
        assert not self.std.requires_grad and self.std.is_cuda
        assert all(not p.requires_grad or name.startswith('memory_route_branch.') for name,p in self.actor.named_parameters())

    def snapshot(self, behavior_sha256, config, consumed, task_objective=CONTINUITY):
        self.verify_frozen()
        assert len(behavior_sha256)==64 and len(consumed)==len(set(consumed))
        return copy.deepcopy(dict(version=CHECKPOINT, checkpoint_training_scope=SCOPE,
            behavior_sha256=behavior_sha256, config=config, consumed=consumed, task_objective=task_objective,
            route=self.actor.memory_route_branch.state_dict(),value=self.value.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(),value_optimizer=self.value_optimizer.state_dict(),
            cuda_rng=torch.cuda.get_rng_state_all(),cpu_rng=torch.get_rng_state()))

    def restore(self, checkpoint, behavior_sha256, config, task_objective=CONTINUITY):
        require_checkpoint_scope(checkpoint, SCOPE)
        assert checkpoint['version']==CHECKPOINT and checkpoint['behavior_sha256']==behavior_sha256 and checkpoint['config']==config
        assert checkpoint.get('task_objective',CONTINUITY)==task_objective, 'Task objective requires an explicit fork'
        consumed = checkpoint['consumed']
        assert len(consumed)==len(set(consumed))
        self.actor.memory_route_branch.load_state_dict(checkpoint['route'])
        self.value.load_state_dict(checkpoint['value'])
        self.actor_optimizer.load_state_dict(copy.deepcopy(checkpoint['actor_optimizer']))
        self.value_optimizer.load_state_dict(copy.deepcopy(checkpoint['value_optimizer']))
        torch.set_rng_state(checkpoint['cpu_rng'].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng']])
        self.verify_frozen()
        return list(consumed)

    def export(self):
        self.verify_frozen()
        result = copy.deepcopy(self.model)
        export_routes(result, self.actor.memory_route_branch)
        actor_layers, actor_attention = self.actor.base.export()
        value_layers, value_attention = self.value.export()
        # Frozen base/attention are checked rather than silently reserialized.
        result['value'] = value_layers
        result['attention']['value'] = value_attention
        assert len(actor_layers[0]['weight'][0])==1554
        validate(result)
        return result

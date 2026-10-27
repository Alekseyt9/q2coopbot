"""Narrow a restored full-joint optimizer without discarding frozen Adam state."""
import copy
from combat_full_joint_ppo import equal_tree,SCOPE
from ppo_combat import torch

ACTOR_SCOPE='conditional_fire_only'


def fork_config(scope,config):
    assert scope.scope==SCOPE and scope.config.get('actor_scope','all')=='all'
    assert config['actor_scope']==ACTOR_SCOPE and config['version']==SCOPE
    changed={k for k in scope.config.keys()|config.keys() if scope.config.get(k)!=config.get(k)}
    assert changed<= {'actor_scope','actor_lr'},'Fire fork may change only actor scope and its learning rate'
    scope.config=copy.deepcopy(config)
    for group in scope.actor_optimizer.param_groups:
        if group['name']=='actor_and_std':group['lr']=config['actor_lr']
    scope.verify()


class FireFreeze:
    def __init__(self,scope):
        assert scope.scope==SCOPE and scope.config.get('actor_scope')==ACTOR_SCOPE
        self.scope=scope
        module=scope.actor.conditional_fire_branch
        assert module is not None and not scope.model.get('wide_aim') and not scope.model.get('movement_options')
        self.parameters=list(module.parameters());self.ids={id(p) for p in self.parameters}
        assert self.ids and self.ids<={id(p) for p in scope.actor_parameters}
        self.frozen={p:p.detach().clone() for p in scope.actor_parameters if id(p) not in self.ids}
        self.moments={p:copy.deepcopy(scope.actor_optimizer.state.get(p,{})) for p in self.frozen}
        scope.actor_optimizer.zero_grad(set_to_none=True)
        for p in scope.actor.parameters():p.requires_grad_(id(p) in self.ids)
        scope.joint.requires_grad_(False)
        # FullJointScope keeps the std parameter registered as trainable for
        # layout compatibility. The narrow likelihood MUST receive std.detach().
        assert scope.std.requires_grad
        self.verify()

    def verify(self):
        assert all(torch.equal(p.detach(),value) for p,value in self.frozen.items()),'Frozen actor parameter changed'
        assert all(equal_tree(self.scope.actor_optimizer.state.get(p,{}),state) for p,state in self.moments.items()),'Frozen Adam state changed'
        assert all(p.grad is None for p in self.frozen),'A frozen actor received a gradient'
        assert all(p.is_cuda for p in self.parameters)
        self.scope.verify()

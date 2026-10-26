"""CUDA full-policy PPO scope, including route and gated fine-aim residuals.

This is a new optimizer boundary. Auxiliary-head checkpoints are deliberately
not interpreted as full-policy Adam state. Native evidence admission belongs
to the trainer; constructing this scope does not make a rollout trainable.
"""
import copy
import math
from ppo_combat import torch, nn, layers
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import JointAim, base_model, validate
from combat_spatial_aim import export as export_branches
from combat_checkpoint_scope import require_checkpoint_scope

SCOPE = 'combat_full_joint_policy_ppo_v1'
CHECKPOINT = 'combat_full_joint_policy_checkpoint_v1'
WIDE_SCOPE = 'combat_full_joint_wide_ppo_v2'
WIDE_CHECKPOINT = 'combat_full_joint_wide_checkpoint_v2'


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b.to(a.device))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_tree(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


class FullJointScope:
    def __init__(self, model, config):
        assert torch.cuda.is_available(), 'CUDA required'
        wide=model.get('wide_aim') is not None
        self.scope=WIDE_SCOPE if wide else SCOPE
        self.checkpoint_version=WIDE_CHECKPOINT if wide else CHECKPOINT
        validate(model, allow_shared=True,allow_wide=wide)  # Diagnostic/target calibration stay excluded.
        assert model.get('attention') and not model.get('memory'), 'Temporal attention scope only'
        assert not any(k.startswith('diagnostic_') for k in model)
        assert not any(model.get(k) for k in ('memory_recovery', 'movement_inertia', 'weapon_fire_residual'))
        assert config['version'] == self.scope
        for key in ('actor_lr', 'route_lr', 'fine_lr', 'candidate_lr', 'value_lr'):
            assert math.isfinite(config[key]) and 0 < config[key] <= .01, key
        self.model, self.config = copy.deepcopy(model), copy.deepcopy(config)
        core = base_model(model, allow_shared=True,allow_wide=wide)
        self.actor, self.value = branch(core, 'actor'), branch(core, 'value')
        self.joint = JointAim(model, allow_shared=True,allow_wide=wide)
        if self.joint.shared is not None:
            # Inference uses float64 geometry; trainable/exported weights remain
            # float32, so rebuilding JSON never discards optimizer precision.
            self.joint.head.requires_grad_(False)
            beta = self.joint.shared_beta.detach().clone()
            del self.joint.shared_beta
            self.joint.register_parameter('shared_beta', nn.Parameter(beta))
        self.std = nn.Parameter(torch.tensor(model['log_std'], device='cuda', dtype=torch.float32))
        groups, special = [], set()
        for attr, rate in (('memory_route_branch', 'route_lr'), ('fine_weapon_aim_branch', 'fine_lr')):
            module = getattr(self.actor, attr, None)
            if module is not None:
                params = list(module.parameters())
                special.update(id(p) for p in params)
                groups.append(dict(name=attr, params=params, lr=config[rate]))
        base = [p for p in self.actor.parameters() if p.requires_grad and id(p) not in special] + [self.std]
        groups.insert(0, dict(name='actor_and_std', params=base, lr=config['actor_lr']))
        candidates = [p for p in self.joint.parameters() if p.requires_grad]
        groups.append(dict(name='aim_candidate_head', params=candidates, lr=config['candidate_lr']))
        self.actor_parameters = [p for g in groups for p in g['params']]
        assert len({id(p) for p in self.actor_parameters}) == len(self.actor_parameters), 'Duplicate optimizer parameter'
        expected = {id(p) for m in (self.actor, self.joint) for p in m.parameters() if p.requires_grad} | {id(self.std)}
        assert {id(p) for p in self.actor_parameters} == expected, 'Unowned trainable parameter'
        self.actor_optimizer = torch.optim.Adam(groups)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config['value_lr'])
        self.frozen_hints = copy.deepcopy(self.joint.hints.state_dict())
        self.verify()

    def verify(self):
        assert all(p.is_cuda for m in (self.actor, self.value, self.joint) for p in m.parameters())
        assert self.std.is_cuda and self.std.requires_grad
        assert not any(p.requires_grad for p in self.joint.hints.parameters())
        assert equal_tree(self.frozen_hints, self.joint.hints.state_dict()), 'Frozen geometry predictor changed'
        assert all(bool(torch.isfinite(p).all()) for p in [*self.actor_parameters, *self.value.parameters()])

    def project(self):
        with torch.no_grad():
            self.std.clamp_(-8, 1)
            if self.joint.shared is not None:
                self.joint.shared_beta.clamp_(-20, 20)

    def export(self):
        self.verify()
        result = copy.deepcopy(self.model)
        for name, module in (('actor', self.actor.base), ('value', self.value)):
            dense, attention = module.export()
            result[name] = dense
            result['attention'][name] = attention
        export_branches(result, self.actor)
        result['log_std'] = self.std.detach().cpu().tolist()
        result['aim_candidate_head']['layers'] = layers(self.joint.head.layers)
        if self.joint.shared is not None:
            result['aim_candidate_head']['shared_scorer']['layers'] = layers(self.joint.shared)
            result['aim_candidate_head']['shared_scorer']['beta'] = float(self.joint.shared_beta.detach())
        if self.joint.wide is not None:
            result['wide_aim']['layers']=layers(self.joint.wide.layers)
        validate(result, allow_shared=True,allow_wide=self.joint.wide is not None)
        return result

    def snapshot(self, weights_sha256, consumed):
        self.verify()
        assert len(weights_sha256) == 64 and len(set(consumed)) == len(consumed)
        return copy.deepcopy(dict(version=self.checkpoint_version, checkpoint_training_scope=self.scope,
            weights_sha256=weights_sha256, config=self.config, consumed_rollouts=list(consumed),
            actor=self.actor.state_dict(), value=self.value.state_dict(), joint=self.joint.state_dict(),
            log_std=self.std.detach(), actor_optimizer=self.actor_optimizer.state_dict(),
            value_optimizer=self.value_optimizer.state_dict(), rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(), optimizer_layout=self.layout()))

    def layout(self):
        names = {id(p): 'actor.'+n for n, p in self.actor.named_parameters()}
        names.update({id(p): 'joint.'+n for n, p in self.joint.named_parameters()})
        names[id(self.std)] = 'log_std'
        return [dict(name=g['name'], parameters=[names[id(p)] for p in g['params']])
                for g in self.actor_optimizer.param_groups]

    def restore(self, cp, weights_sha256):
        require_checkpoint_scope(cp, self.scope)
        assert cp['version'] == self.checkpoint_version and cp['weights_sha256'] == weights_sha256
        assert cp['config'] == self.config and cp['optimizer_layout'] == self.layout(), 'Explicit optimizer migration required'
        for name, state in (('actor', self.actor.state_dict()), ('value', self.value.state_dict()),
                            ('joint', self.joint.state_dict()), ('log_std', self.std.detach())):
            assert equal_tree(state, cp[name]), 'Checkpoint differs from supplied JSON: '+name
        tokens = cp['consumed_rollouts']
        assert len(tokens) == len(set(tokens)), 'Duplicate consumed rollout'
        self.actor_optimizer.load_state_dict(copy.deepcopy(cp['actor_optimizer']))
        self.value_optimizer.load_state_dict(copy.deepcopy(cp['value_optimizer']))
        torch.set_rng_state(cp['rng'].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in cp['cuda_rng']])
        self.verify()
        return list(tokens)

"""Typed CUDA optimizer boundary: option actor/value only, frozen source policy."""
import copy
import hashlib
import json
from ppo_combat import torch
from combat_movement_options import MovementOptions, validate
from combat_full_joint_ppo import equal_tree
from combat_checkpoint_scope import require_checkpoint_scope

SCOPE='combat_movement_options_ppo_v1'
CHECKPOINT='combat_movement_options_checkpoint_v1'


def frozen_source(model):
    result=copy.deepcopy(model);result.pop('movement_options')
    result['action_distribution']='combat_joint_aim_candidates_v1'
    return result


def semantic_sha(model):
    return hashlib.sha256(json.dumps(model,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class MovementOptionScope:
    def __init__(self,model,config):
        validate(model)
        assert model['movement_options']['enabled'] and config['version']==SCOPE
        assert all(0<config[k]<=.01 for k in ('actor_lr','value_lr'))
        self.model,self.config=copy.deepcopy(model),copy.deepcopy(config)
        self.frozen=frozen_source(model);self.source_sha256=semantic_sha(self.frozen)
        self.options=MovementOptions(model['movement_options'])
        self.actor_optimizer=torch.optim.Adam(self.options.actor.parameters(),lr=config['actor_lr'])
        self.value_optimizer=torch.optim.Adam(self.options.value.parameters(),lr=config['value_lr'])
        self.verify()

    def verify(self):
        assert all(p.is_cuda and bool(torch.isfinite(p).all()) for p in self.options.parameters())
        assert frozen_source(self.model)==self.frozen

    def export(self):
        self.verify();result=copy.deepcopy(self.model)
        result['movement_options']=self.options.export(result['movement_options'])
        assert frozen_source(result)==self.frozen
        validate(result)
        return result

    def snapshot(self,weights_sha256,consumed,progress):
        self.verify()
        assert len(weights_sha256)==64 and len(consumed)==len(set(consumed))
        return copy.deepcopy(dict(version=CHECKPOINT,checkpoint_training_scope=SCOPE,
            weights_sha256=weights_sha256,frozen_source_sha256=self.source_sha256,config=self.config,
            options=self.options.state_dict(),actor_optimizer=self.actor_optimizer.state_dict(),
            value_optimizer=self.value_optimizer.state_dict(),consumed_rollouts=consumed,progress=progress,
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))

    def restore(self,checkpoint,weights_sha256):
        require_checkpoint_scope(checkpoint,SCOPE)
        assert checkpoint['version']==CHECKPOINT and checkpoint['config']==self.config
        assert checkpoint['weights_sha256']==weights_sha256 and checkpoint['frozen_source_sha256']==self.source_sha256
        assert equal_tree(checkpoint['options'],self.options.state_dict()),'Checkpoint and JSON disagree'
        tokens=checkpoint['consumed_rollouts'];assert len(tokens)==len(set(tokens))
        self.actor_optimizer.load_state_dict(copy.deepcopy(checkpoint['actor_optimizer']))
        self.value_optimizer.load_state_dict(copy.deepcopy(checkpoint['value_optimizer']))
        torch.set_rng_state(checkpoint['rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in checkpoint['cuda_rng']])
        self.verify()
        return copy.deepcopy(tokens),copy.deepcopy(checkpoint['progress'])

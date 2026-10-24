"""Fork a versioned execution-contract checkpoint without changing learned tensors."""
import argparse
import copy
from pathlib import Path

from evaluate_registered_combat_models import read, sha
from ppo_combat import torch, nn, training_devices
from ppo_recurrent import durable_json, durable_write
from combat_checkpoint_scope import DECISION_SCOPE, require_checkpoint_scope
from combat_execution_contract import LEGACY, CHECKED_GROUND, SCOPE, normalize
from combat_joint_aim_policy import base_model
from migrate_combat_navigation_cuda import branch
from fork_conditional_weapon_exploration_cuda import equal_tree


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--contract',choices=(LEGACY,CHECKED_GROUND),required=True)
    a=ap.parse_args();assert not a.out.exists()
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol=read(a.protocol);assert protocol['version']==SCOPE
    seal=read(a.source/'complete.json');assert seal['state']=='complete'
    for name,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
        assert sha(a.source/name)==seal[key]
    assert seal['weights_sha256']==protocol['initial_source_weights_sha256']
    model=read(a.source/'weights.json');assert normalize(model.get('combat_execution_contract'))==LEGACY
    cp=torch.load(a.source/'checkpoint.pt',map_location='cuda',weights_only=True)
    require_checkpoint_scope(cp,DECISION_SCOPE)
    assert cp['weights_sha256']==seal['weights_sha256'] and len(cp['decision_scope_optimizer']['param_groups'])==5
    assert cp['config']['objective_reward_sha256']==protocol['reward_sha256']
    original=copy.deepcopy(cp)
    core=base_model(model,allow_shared=True)
    actor,value=branch(core,'actor'),branch(core,'value')
    std=nn.Parameter(torch.tensor(model['log_std'],device='cuda'))
    assert equal_tree(actor.state_dict(),cp['actor']) and equal_tree(value.state_dict(),cp['value'])
    assert torch.equal(std,cp['log_std'])
    updated=copy.deepcopy(model);updated['combat_execution_contract']=a.contract
    # Verify the tagged JSON still constructs exactly the same CUDA networks.
    decoded_core=base_model(updated,allow_shared=True)
    decoded_actor,decoded_value=branch(decoded_core,'actor'),branch(decoded_core,'value')
    assert equal_tree(decoded_actor.state_dict(),actor.state_dict()) and equal_tree(decoded_value.state_dict(),value.state_dict())
    a.out.mkdir(parents=True)
    durable_json(a.out/'weights.json',updated)
    fork=copy.deepcopy(cp)
    fork['config']=dict(cp['config'],combat_execution_contract=a.contract)
    fork.update(checkpoint_training_scope=SCOPE,weights_sha256=sha(a.out/'weights.json'),
        execution_contract_scope=dict(version=SCOPE,protocol_sha256=sha(a.protocol),updates_completed=0,
            combat_execution_contract=a.contract,initial_source_weights_sha256=seal['weights_sha256']),
        execution_contract_optimizer=copy.deepcopy(cp['decision_scope_optimizer']),execution_contract_history=[])
    durable_write(a.out/'checkpoint.pt',lambda f:torch.save(fork,f))
    loaded=torch.load(a.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    require_checkpoint_scope(loaded,SCOPE)
    assert equal_tree(fork,loaded) and equal_tree(cp,original)
    for key in cp:
        if key not in ('config','checkpoint_training_scope','weights_sha256'):
            assert equal_tree(cp[key],loaded[key]),key
    assert equal_tree(loaded['execution_contract_optimizer'],cp['decision_scope_optimizer'])
    for old_scope in (None,DECISION_SCOPE):
        try:require_checkpoint_scope(loaded,old_scope)
        except ValueError:pass
        else:raise AssertionError('Old trainer admitted new execution scope')
    report=dict(state='complete',version='combat_execution_contract_initializer_v1',device='cuda',
        combat_execution_contract=a.contract,source_weights_sha256=seal['weights_sha256'],
        source_checkpoint_sha256=seal['checkpoint_sha256'],weights_sha256=sha(a.out/'weights.json'),
        protocol_sha256=sha(a.protocol),source_training_tensors_exact=True,source_optimizers_exact=True,
        source_rng_exact=True,checkpoint_roundtrip_exact=True,source_unchanged=True,
        training_performed=False,scope_updates=0,promotion=False,
        scope='Explicit metadata/optimizer-scope fork. Identical CUDA actor, value, std and inherited optimizer/RNG states. No fitting.')
    durable_json(a.out/'report.json',report)
    durable_json(a.out/'complete.json',dict(state='complete',weights_sha256=sha(a.out/'weights.json'),
        checkpoint_sha256=sha(a.out/'checkpoint.pt'),report_sha256=sha(a.out/'report.json')))
    print(report)


if __name__=='__main__':main()

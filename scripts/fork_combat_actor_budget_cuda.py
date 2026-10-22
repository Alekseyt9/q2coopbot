"""Explicit actor-step-cap fork; preserve weights, reward, Adam and RNG."""
import argparse
import copy
from pathlib import Path
from evaluate_registered_combat_models import read,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--actor-steps',type=int,required=True)
    args=ap.parse_args();assert not args.out.exists() and 1<=args.actor_steps<=1000
    from ppo_combat import torch,training_devices
    training_devices();torch.set_default_device('cuda')
    from ppo_recurrent import durable_json,durable_write
    from fork_conditional_weapon_exploration_cuda import equal_tree
    from migrate_combat_navigation_cuda import branch
    seal=read(args.source/'complete.json');pins={}
    for name,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
        path=args.source/name;assert sha(path)==seal[key];pins[str(path.resolve())]=sha(path)
    cp=torch.load(args.source/'checkpoint.pt',map_location='cuda',weights_only=True)
    model=read(args.source/'weights.json')
    assert cp['weights_sha256']==seal['weights_sha256'] and model.get('conditional_fire')
    for key in ('actor','value'):assert equal_tree(branch(model,key).state_dict(),cp[key])
    assert args.actor_steps!=cp['config']['actor_steps']
    previous=copy.deepcopy(cp['config']);current={**previous,'actor_steps':args.actor_steps}
    fork_record=dict(version='combat_actor_budget_fork_v1',source_checkpoint_sha256=seal['checkpoint_sha256'],
        previous=previous,current=current,changed_fields=['actor_steps'])
    # Keep CPU RNG byte storage for the ordinary resume path; no CPU NN math.
    fork=torch.load(args.source/'checkpoint.pt',map_location='cpu',weights_only=True)
    fork['config']=current
    fork['training_config_forks']=fork.get('training_config_forks',[])+[fork_record]
    args.out.mkdir(parents=True)
    durable_write(args.out/'weights.json',lambda f:f.write((args.source/'weights.json').read_bytes()))
    assert sha(args.out/'weights.json')==seal['weights_sha256']
    durable_write(args.out/'checkpoint.pt',lambda f:torch.save(fork,f))
    loaded=torch.load(args.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    for key in cp:
        if key not in ('config','training_config_forks'):assert equal_tree(cp[key],loaded[key]),key
    assert loaded['config']==current and loaded['training_config_forks'][-1]==fork_record
    durable_json(args.out/'training-config.json',current)
    receipt=dict(state='complete',device='cuda',version='combat_actor_budget_fork_v1',
        actor_step_cap_before=previous['actor_steps'],actor_step_cap_after=current['actor_steps'],
        reward_lr_std_critic_budget_unchanged=True,weights_byte_exact=True,checkpoint_state_except_explicit_budget_fork_exact=True,
        inputs=pins,script_sha256=sha(__file__),training_performed=False,promotion=False,
        scope='Alternative training-budget branch. Same behavior policy and unconsumed-history guards; no replay relabel, optimizer reset or numerical training.')
    for path,digest in pins.items():assert sha(Path(path))==digest
    durable_json(args.out/'report.json',receipt)
    durable_json(args.out/'complete.json',dict(version='combat_update_complete_v1',weights_sha256=sha(args.out/'weights.json'),
        checkpoint_sha256=sha(args.out/'checkpoint.pt'),report_sha256=sha(args.out/'report.json')))
    print(receipt['version'],previous['actor_steps'],'->',current['actor_steps'])


if __name__=='__main__':main()

"""Explicit zero-column fire-input migration with preserved CUDA Adam history."""
import argparse
import copy
from pathlib import Path
from ppo_combat import torch,training_devices
from combat_full_joint_ppo import FullJointScope,equal_tree
from combat_conditional_fire import VERSION,CONTEXT_VERSION,INPUT_WIDTH,CONTEXT_WIDTH
from process_combat_architecture_pool import read,save,sha

STATE_KEY='conditional_fire_branch.0.weight'
PARAMETER_NAME='actor.'+STATE_KEY


def migrate(model,checkpoint):
    assert model['conditional_fire']['version']==VERSION and not checkpoint.get('technical_only')
    assert not any(model.get(k) for k in ('movement_options','weapon_fire_residual','wide_aim'))
    before=FullJointScope(model,checkpoint['config'])
    before.restore(checkpoint,checkpoint['weights_sha256'])
    changed=copy.deepcopy(model)
    changed['conditional_fire']['version']=CONTEXT_VERSION
    for row in changed['conditional_fire']['layers'][0]['weight']:
        assert len(row)==INPUT_WIDTH
        row.extend([0.]*(CONTEXT_WIDTH-INPUT_WIDTH))
    cp=copy.deepcopy(checkpoint)
    def expand(t):
        assert t.is_cuda and t.shape==(32,INPUT_WIDTH)
        padded=torch.nn.functional.pad(t,(0,CONTEXT_WIDTH-INPUT_WIDTH))
        assert torch.equal(padded[:,:INPUT_WIDTH],t) and not bool(padded[:,INPUT_WIDTH:].any())
        return padded
    cp['actor'][STATE_KEY]=expand(cp['actor'][STATE_KEY])
    found=[]
    for layout,group in zip(cp['optimizer_layout'],cp['actor_optimizer']['param_groups']):
        for name,index in zip(layout['parameters'],group['params']):
            if name==PARAMETER_NAME:found.append(index)
    assert len(found)==1
    index=found[0];state=cp['actor_optimizer']['state'].get(index,{})
    for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
        if key in state:state[key]=expand(state[key])
    assert all(equal_tree(v,cp['actor'][k]) for k,v in checkpoint['actor'].items() if k!=STATE_KEY)
    assert all(equal_tree(v,cp['actor_optimizer']['state'][k]) for k,v in checkpoint['actor_optimizer']['state'].items() if k!=index)
    assert equal_tree(checkpoint['actor_optimizer']['param_groups'],cp['actor_optimizer']['param_groups'])
    assert equal_tree(checkpoint['value_optimizer'],cp['value_optimizer'])
    assert equal_tree(checkpoint['joint'],cp['joint']) and equal_tree(checkpoint['log_std'],cp['log_std'])
    return changed,cp


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','checkpoint','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();assert not args.out.exists()
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    model=read(args.model);cp=torch.load(args.checkpoint,map_location='cuda',weights_only=True)
    assert cp['weights_sha256']==sha(args.model)
    changed,migrated=migrate(model,cp)
    args.out.mkdir(parents=True);save(args.out/'weights.json',changed)
    migrated['weights_sha256']=sha(args.out/'weights.json')
    migrated['architecture_migration']=dict(version=CONTEXT_VERSION,source_weights_sha256=sha(args.model),
        source_checkpoint_sha256=sha(args.checkpoint),zero_columns=CONTEXT_WIDTH-INPUT_WIDTH)
    migrated['initialization']='conditional_fire_context_zero_column_migration'
    restored=FullJointScope(changed,migrated['config'])
    restored.restore(migrated,migrated['weights_sha256'])
    torch.save(migrated,args.out/'checkpoint.pt')
    reopened=torch.load(args.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    restored.restore(reopened,sha(args.out/'weights.json'))
    assert equal_tree(migrated,reopened)
    report=dict(state='passed',device='cuda',training_performed=False,
        version='combat_fire_context_checkpoint_migration_v1',input_width=CONTEXT_WIDTH,
        old_columns_and_adam_moments_exact=True,new_columns_and_moments_zero=True,
        other_parameters_and_optimizers_exact=True,actual_checkpoint_reopened=True,
        consumed_rollouts_preserved=True,updates_completed=migrated['updates_completed'],
        weights_sha256=sha(args.out/'weights.json'),checkpoint_sha256=sha(args.out/'checkpoint.pt'),
        inputs={str(p.resolve()):sha(p) for p in (args.model,args.checkpoint,Path(__file__))},
        scope='Architecture migration only. CUDA/native behavior identity and quality require separate evidence; no promotion.')
    save(args.out/'report.json',report)
    save(args.out/'complete.json',dict(state='complete',weights_sha256=report['weights_sha256'],
        checkpoint_sha256=report['checkpoint_sha256'],report_sha256=sha(args.out/'report.json')))
    print({k:v for k,v in report.items() if k!='inputs'})


if __name__=='__main__':main()

"""Verify a decision-scope fork preserves inactive parameters and Adam state."""
import argparse
import pathlib

from ppo_recurrent import torch, training_devices, read, sha, durable_json


def main():
    parser=argparse.ArgumentParser()
    for name in ('before','after','report','out'):
        parser.add_argument('--'+name,type=pathlib.Path,required=True)
    args=parser.parse_args()
    assert not args.out.exists(), 'Fresh audit output required'
    training_devices()
    before=torch.load(args.before,map_location='cuda',weights_only=True)
    after=torch.load(args.after,map_location='cuda',weights_only=True)
    report=read(args.report)
    assert before['version']==after['version']=='combat_architecture_checkpoint_v1'
    assert before['architecture']==after['architecture'] and before['config']==after['config']
    assert after['actor_training']==dict(scope='decisions',learning_rate=report['actor_training']['learning_rate'])
    assert report['device']=='cuda'
    assert report['actor_scope_fork'] or before.get('actor_training')==after['actor_training'], 'Expected explicit scope fork or same-scope continuation'
    assert report['resume_sha256']==sha(args.before) and report['weights_sha256']==after['weights_sha256']
    assert before['weights_sha256']==report['behavior_sha256']
    assert after['consumed_rollouts']==before['consumed_rollouts']+[report['rollout_sha256']]
    assert after['updates_completed']==before['updates_completed']+1
    assert after['total_actor_steps']==before['total_actor_steps']+report['actor_steps']
    assert torch.equal(before['log_std'],after['log_std'])
    assert before['actor'].keys()==after['actor'].keys()
    keys=list(before['actor'])
    selected={name for name in keys if name.rsplit('.',2)[-2] in ('head','residual','output')}
    assert len(selected)==4, 'Expected base and recurrent/attention output weights and biases'
    shared={name for name in keys if name.startswith('target_branch.')}
    selected |= shared
    frozen_count=changed_count=0
    masks={}
    for name,old in before['actor'].items():
        new=after['actor'][name]
        assert old.is_cuda and new.is_cuda and old.dtype==new.dtype and old.shape==new.shape
        if name in selected:
            if name in shared:
                mask=torch.ones_like(old,dtype=torch.bool)
            else:
                assert old.ndim in (1,2) and old.shape[0] in (45,81)
                mask=torch.zeros_like(old,dtype=torch.bool);mask[4]=True;mask[8:29]=True
            assert torch.equal(old[~mask],new[~mask]), name
            changed_count+=int((old[mask]!=new[mask]).sum())
            frozen_count+=int((~mask).sum());masks[name]=mask
        else:
            assert torch.equal(old,new), name
            frozen_count+=old.numel()
    assert changed_count>0
    # These actors contain parameters only, no persistent buffers. Assert the
    # optimizer's parameter order and moment shapes before using that contract.
    old_opt,new_opt=before['actor_optimizer'],after['actor_optimizer']
    old_ids=[p for group in old_opt['param_groups'] for p in group['params']]
    new_ids=[p for group in new_opt['param_groups'] for p in group['params']]
    assert old_ids==new_ids and len(old_ids)==len(keys)+1
    for index,name in zip(old_ids,keys+['log_std']):
        old=old_opt['state'].get(index,{})
        new=new_opt['state'].get(index,{})
        if name not in selected:
            assert old.keys()==new.keys(), name
            for field in old:assert torch.equal(old[field],new[field]), (name,field)
            continue
        if name in shared and not old:
            if report['actor_steps']:
                assert 'step' in new and float(new['step'])==report['actor_steps'], name
            else:
                assert not new, name
            continue
        assert old.keys()==new.keys() and 'step' in old
        assert float(new['step']-old['step'])==report['actor_steps'], name
        for field in ('exp_avg','exp_avg_sq'):
            assert old[field].shape==new[field].shape==masks[name].shape
            assert torch.equal(old[field][~masks[name]],new[field][~masks[name]]), (name,field)
    durable_json(args.out,dict(version='combat_decision_scope_checkpoint_audit_v1',device='cuda',
        before_checkpoint_sha256=sha(args.before),after_checkpoint_sha256=sha(args.after),
        report_sha256=sha(args.report),actor_steps=report['actor_steps'],
        selected_parameter_tensors=sorted(selected),changed_selected_scalars=changed_count,
        preserved_actor_scalars=frozen_count,log_std_preserved=True,
        inactive_optimizer_state_preserved=True,consumed_rollout_append_verified=True,
        scope='Actor movement/vertical/aim/mode/encoder/spatial/std and inactive Adam states exactly preserved. Selected output tensor clocks advance. Critic trains normally; game quality assessed separately.'))
    print(dict(state='complete',changed_selected_scalars=changed_count,preserved_actor_scalars=frozen_count),flush=True)


if __name__=='__main__':main()

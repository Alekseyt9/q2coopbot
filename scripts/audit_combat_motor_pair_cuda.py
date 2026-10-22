"""Verify actual motor-only PPO isolation, shared state and the TRAIN gate."""
import argparse
import json
import os
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();assert not args.out.exists()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    assert torch.cuda.is_available()
    torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    from fork_conditional_weapon_exploration_cuda import equal_tree
    from migrate_combat_navigation_cuda import branch
    from ppo_recurrent import prepare_context,sequence
    from combat_sequence_retention import policy_kl,normal_kl
    root=args.root.resolve();p=read(root/'protocol.json');assert read(root/'training-complete.json')['state']=='complete'
    source=Path(p['source']);original=torch.load(source/'checkpoint.pt',map_location='cuda',weights_only=True)
    checkpoints={};reports={};pins=[]
    for name in p['variants']:
        update=root/name;seal=read(update/'complete.json');report=read(update/'report.json')
        assert report['device']=='cuda' and 0<report['actor_steps']<=p['actor_steps_cap']
        for file,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
            assert sha(update/file)==seal[key];pins.append(dict(path=str(update/file),sha256=seal[key]))
        before=torch.load(root/(name+'-initial.pt'),map_location='cuda',weights_only=True)
        assert sha(root/(name+'-initial.pt'))==report['initial_boundary_sha256']
        for key in ['actor','value','log_std','actor_optimizer','value_optimizer','rng','cuda_rng']:
            assert equal_tree(before[key],original[key]),(name,key)
        assert before['rollout_sha256'] not in original['consumed_rollouts']
        after=torch.load(update/'checkpoint.pt',map_location='cuda',weights_only=True)
        assert after['consumed_rollouts']==original['consumed_rollouts']+[before['rollout_sha256']]
        assert after['total_actor_steps']==original['total_actor_steps']+report['actor_steps']
        assert after['config']==original['config']
        for part in ['actor','value']:assert equal_tree(branch(read(update/'weights.json'),part).state_dict(),after[part])
        checkpoints[name]=after;reports[name]=report
    motor=checkpoints['motor'];assert reports['motor']['actor_training']['scope']=='motor-residual'
    for key in ['value','value_optimizer']:assert equal_tree(motor[key],checkpoints['full'][key]),key
    changed=[]
    for name,value in original['actor'].items():
        current=motor['actor'][name]
        if name.startswith('motor_branch.'):changed.append(not torch.equal(value,current))
        else:assert torch.equal(value,current),name
    assert any(changed) and torch.equal(motor['log_std'],original['log_std'])
    # An omitted Adam slot and an explicit empty dict both mean uninitialized.
    # Require every numerical moment/counter to match; do not demand empty keys.
    empty_slots=0
    for key,value in original['actor_optimizer']['state'].items():
        current=motor['actor_optimizer']['state'].get(key,{})
        assert equal_tree(value,current),key
        empty_slots+=not value and key not in motor['actor_optimizer']['state']
    assert motor['actor_optimizer']['param_groups']==original['actor_optimizer']['param_groups']
    groups=original['actor_optimizer']['param_groups'];assert groups[-1]['name']=='motor_residual'
    new_ids=groups[-1]['params'];assert all(key not in original['actor_optimizer']['state'] for key in new_ids)
    assert all(float(motor['actor_optimizer']['state'][key]['step'])==reports['motor']['actor_steps'] for key in new_ids)
    data=Path(p['corpus'])/'processing-train/merged';meta=read(data/'report.json')
    assert sha(data/'rollout.jsonl')==meta['rollout_sha256'] and sha(data/'sequence.jsonl')==meta['sequence_sha256']
    rows=[json.loads(s) for s in (data/'rollout.jsonl').read_text().splitlines()]
    context=[json.loads(s) for s in (data/'sequence.jsonl').read_text().splitlines()]
    prepared=prepare_context(context,rows,'cuda');x=torch.tensor([r['features'] for r in rows]);z=torch.tensor([r['sample']['latent'][:2] for r in rows])
    teacher_model=read(source/'weights.json');teacher=branch(teacher_model,'actor');old_std=torch.tensor(teacher_model['log_std'])
    statistics={}
    with torch.no_grad():
        old=sequence(teacher,prepared)[0]
        for name in p['variants']:
            model=read(root/name/'weights.json');actor=branch(model,'actor');raw=sequence(actor,prepared)[0];std=torch.tensor(model['log_std'])
            total=policy_kl(raw,std,old,old_std,x,actor.sampled_motion_aim_branch,teacher.sampled_motion_aim_branch,z,actor.motor_branch,teacher.motor_branch)
            body=normal_kl(old[:,:2].double(),raw[:,:2].double(),old_std[:2].double(),std[:2].double())
            statistics[name]=dict(mean_move_shift=float((raw[:,:2].tanh()-old[:,:2].tanh()).abs().mean()),
                total_kl=float(total.mean()),movement_kl=float(body.mean()),nonmovement_kl_max=float((total-body).abs().max()),
                raw_nonmovement_max_error=float((raw[:,2:]-old[:,2:]).abs().max()),
                actor_steps=reports[name]['actor_steps'],approx_kl=reports[name]['final_approx_kl'])
            if name=='motor':assert torch.equal(raw[:,2:],old[:,2:])
    gate=p['mechanical_gate'];measurement=statistics['motor']
    passed=measurement['mean_move_shift']>=gate['minimum_mean_move_shift'] and measurement['nonmovement_kl_max']<=gate['maximum_nonmovement_kl']
    for path,digest in p['inputs'].items():
        archived=Path(path)
        if archived.name=='ppo_recurrent.py':archived=root/'full/trainer.py'
        if archived.name=='combat_motor_residual.py':archived=root/'motor-helper.py'
        assert sha(archived)==digest
    save(args.out,dict(state='complete',device='cuda',initial_state_exact=True,frozen_actor_std_and_adam_exact=True,
         runtime_checkpoint_exact=True,shared_final_critic_and_adam_exact=True,new_motor_adam_clock_exact=True,
         adam_uninitialized_slots_normalized=empty_slots,
         statistics=statistics,mechanical_gate_passed=passed,inputs=pins,script_sha256=sha(__file__),training_performed=False,promotion=False,
         scope='Exact protected parameters/moments and conditional KL on actual TRAIN histories. Native behavior on new trajectories is not proven by this gate.'))
    print(json.dumps(dict(statistics=statistics,mechanical_gate_passed=passed),indent=2))


if __name__=='__main__':main()

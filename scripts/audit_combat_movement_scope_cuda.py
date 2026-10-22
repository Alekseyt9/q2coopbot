"""Verify projected motor-only PPO state and its predeclared mechanical gate."""
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
    from migrate_combat_navigation_cuda import branch
    from fork_conditional_weapon_exploration_cuda import equal_tree
    from ppo_recurrent import prepare_context,sequence
    from combat_weapon_state_residual import optimizer
    from combat_sequence_retention import policy_kl,normal_kl,categorical_kl
    root=args.root.resolve();p=read(root/'protocol.json');assert read(root/'training-complete.json')['state']=='complete'
    source=Path(p['source']);model=read(source/'weights.json')
    original=torch.load(source/'checkpoint.pt',map_location='cuda',weights_only=True)
    checkpoints={};reports={};pins=[]
    for name in p['variants']:
        update=root/name;seal=read(update/'complete.json');report=read(update/'report.json')
        for filename,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
            assert sha(update/filename)==seal[key];pins.append(dict(path=str(update/filename),sha256=seal[key]))
        initial=torch.load(root/(name+'-initial.pt'),map_location='cuda',weights_only=True)
        assert sha(root/(name+'-initial.pt'))==report['initial_boundary_sha256']
        for key in ['actor','value','log_std','actor_optimizer','value_optimizer','rng','cuda_rng']:
            assert equal_tree(initial[key],original[key]),(name,key)
        assert initial['rollout_sha256'] not in original['consumed_rollouts']
        checkpoints[name]=torch.load(update/'checkpoint.pt',map_location='cuda',weights_only=True);reports[name]=report
    assert sha(root/'full/weights.json')==sha(root.parent/'combat-elevation-ppo-v1-20261022/update/weights.json')
    after=checkpoints['movement'];assert reports['movement']['actor_training']['scope']=='movement'
    for key in ['value','value_optimizer']:assert equal_tree(checkpoints['full'][key],after[key]),key
    head_keys={'base.head.weight','base.head.bias','base.residual.weight','base.residual.bias'}
    changed=[]
    for name,value in original['actor'].items():
        current=after['actor'][name]
        if name in head_keys:
            mask=torch.zeros(value.shape[0],dtype=torch.bool);mask[:2]=True;mask[5:8]=True
            assert torch.equal(value[~mask],current[~mask]),name
            changed.append(bool((value[mask]!=current[mask]).any()))
        else:assert torch.equal(value,current),name
    assert any(changed) and torch.equal(original['log_std'][2:],after['log_std'][2:])
    teacher=branch(model,'actor');std=torch.nn.Parameter(original['log_std'].clone());opt=optimizer(teacher,std,.00003,.0003)
    parameter_names={parameter:name for name,parameter in teacher.named_parameters()};parameter_names[std]='log_std'
    for group,saved in zip(opt.param_groups,original['actor_optimizer']['param_groups']):
        for parameter,key in zip(group['params'],saved['params']):
            name=parameter_names[parameter];before_state=original['actor_optimizer']['state'].get(key,{})
            after_state=after['actor_optimizer']['state'].get(key,{})
            if name not in head_keys and name!='log_std':assert equal_tree(before_state,after_state),name
            else:
                mask=torch.zeros_like(parameter,dtype=torch.bool)
                if name=='log_std':mask[:2]=True
                else:mask[:2]=True;mask[5:8]=True
                for field in ['exp_avg','exp_avg_sq']:
                    if field in before_state:assert torch.equal(before_state[field][~mask],after_state[field][~mask]),(name,field)
    data=Path(p['corpus'])/'processing-train/merged'
    rows=[json.loads(s) for s in (data/'rollout.jsonl').read_text().splitlines()]
    context=[json.loads(s) for s in (data/'sequence.jsonl').read_text().splitlines()]
    prepared=prepare_context(context,rows,'cuda');x=torch.tensor([r['features'] for r in rows]);z=torch.tensor([r['sample']['latent'][:2] for r in rows])
    old_std=original['log_std'];statistics={}
    with torch.no_grad():
        old=sequence(teacher,prepared)[0]
        for name in p['variants']:
            current_model=read(root/name/'weights.json');student=branch(current_model,'actor')
            raw=sequence(student,prepared)[0];current_std=torch.tensor(current_model['log_std'])
            total=policy_kl(raw,current_std,old,old_std,x,getattr(student,'sampled_motion_aim_branch',None),getattr(teacher,'sampled_motion_aim_branch',None),z)
            body=normal_kl(old[:,:2].double(),raw[:,:2].double(),old_std[:2].double(),current_std[:2].double())
            body=body+categorical_kl(old[:,5:8].double(),raw[:,5:8].double())
            statistics[name]=dict(mean_absolute_move_shift=float((raw[:,:2].tanh()-old[:,:2].tanh()).abs().mean()),
                                  total_kl=float(total.mean()),body_kl=float(body.mean()),
                                  nonmovement_kl_fraction=float((total-body).mean()/total.mean()),
                                  actor_steps=reports[name]['actor_steps'])
    ratio=statistics['movement']['mean_absolute_move_shift']/statistics['full']['mean_absolute_move_shift']
    gate=p['mechanical_gate'];passed=ratio>=gate['minimum_move_shift_ratio'] and statistics['movement']['nonmovement_kl_fraction']<=gate['maximum_nonmovement_kl_fraction']
    for name,digest in p['inputs'].items():
        path=Path(name)
        if path.name=='ppo_recurrent.py':path=root/'full/trainer.py'
        if path.name=='combat_target_head_scope.py':path=root/'scope-helper.py'
        assert sha(path)==digest
    save(args.out,dict(state='complete',device='cuda',source_boundary_exact=True,full_runtime_exact=True,
         frozen_actor_parameters_and_moments_exact=True,critic_and_adam_exact=True,statistics=statistics,
         movement_shift_ratio=ratio,mechanical_gate_passed=passed,inputs=pins,
         script_sha256=sha(__file__),training_performed=False,promotion=False,
         limitation='Mechanical TRAIN gate, not native quality. Aim outputs may change through frozen controllers that read movement means.'))
    print(json.dumps(dict(statistics=statistics,movement_shift_ratio=ratio,mechanical_gate_passed=passed),indent=2))


if __name__=='__main__':main()

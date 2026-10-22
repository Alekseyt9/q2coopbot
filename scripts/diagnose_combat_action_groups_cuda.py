"""Read-only TRAIN action-group KL and AWR-weight diagnostic on CUDA."""
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
    from ppo_recurrent import prepare_context,sequence
    from ppo_combat import advantages
    from combat_sequence_retention import policy_kl,normal_kl,categorical_kl
    from combat_target_head import availability
    from combat_weapon_head import feature_mask
    from combat_advantage_regression import weights
    root=args.root.resolve();p=read(root/'protocol.json');assert read(root/'pair-audit.json')['state']=='complete'
    source=Path(p['source']);data=Path(p['corpus'])/'processing-train/merged';meta=read(data/'report.json')
    assert sha(data/'rollout.jsonl')==meta['rollout_sha256'] and sha(data/'sequence.jsonl')==meta['sequence_sha256']
    rows=[json.loads(s) for s in (data/'rollout.jsonl').read_text().splitlines()]
    context=[json.loads(s) for s in (data/'sequence.jsonl').read_text().splitlines()]
    prepared=prepare_context(context,rows,'cuda');x=torch.tensor([r['features'] for r in rows]);z=torch.tensor([r['sample']['latent'][:2] for r in rows])
    source_model=read(source/'weights.json');teacher=branch(source_model,'actor');old_std=torch.tensor(source_model['log_std']).double()
    retention=read(root/'retention.json');locations={(r['seed'],r['index']):i for i,r in enumerate(rows)}
    masses=torch.zeros(len(rows),dtype=torch.float64)
    for member in retention['members']:masses[locations[member['seed'],member['index']]]=member['weight']
    assert abs(float(masses.sum())-1)<1e-6
    results={};inputs=[dict(path=str(data/'report.json'),sha256=sha(data/'report.json'))]
    with torch.no_grad():
        old=sequence(teacher,prepared)[0];before=old.double();valid=availability(x)
        target_probs=torch.softmax(before[:,20:29].masked_fill(~valid,float('-inf')),-1)
        for name,update in [('source',source),('ppo',root/'legacy'),('bc',root/'bc'),('awr',root/'awr')]:
            model_path=update/'weights.json';seal=read(update/'complete.json');assert sha(model_path)==seal['weights_sha256']
            model=read(model_path);student=branch(model,'actor');raw=sequence(student,prepared)[0];current=raw.double();std=torch.tensor(model['log_std']).double()
            total=policy_kl(raw,std,old,old_std,x,getattr(student,'sampled_motion_aim_branch',None),getattr(teacher,'sampled_motion_aim_branch',None),z)
            groups=dict(movement=normal_kl(before[:,:2],current[:,:2],old_std[:2],std[:2]),
                        attack=torch.distributions.kl_divergence(torch.distributions.Bernoulli(logits=before[:,4]),torch.distributions.Bernoulli(logits=current[:,4])),
                        vertical=categorical_kl(before[:,5:8],current[:,5:8]),
                        weapon=categorical_kl(before[:,8:20],current[:,8:20],feature_mask(x[:,:845])),
                        target=categorical_kl(before[:,20:29],current[:,20:29],valid),
                        aim_mode=(target_probs*categorical_kl(before[:,63:81].reshape(-1,9,2),current[:,63:81].reshape(-1,9,2))).sum(-1))
            groups['conditional_aim']=total-sum(groups.values())
            assert float(groups['conditional_aim'].min())>-1e-6
            if name=='source':assert float(total.abs().max())<1e-6
            results[name]=dict(total_kl=float(total.mean()),retention_kl=float((total*masses).sum()),
                              groups={k:dict(mean_kl=float(v.mean()),retention_kl=float((v*masses).sum())) for k,v in groups.items()},
                              mean_absolute_move_shift=float((raw[:,:2].tanh()-old[:,:2].tanh()).abs().mean()),
                              log_std=model['log_std'])
            inputs.append(dict(path=str(model_path),sha256=sha(model_path)))
            del student
        value=branch(read(root/'awr/weights.json'),'value');baseline=sequence(value,prepared)[0][:,0]
        _,returns=advantages(rows,.99,1.);importance,stats=weights(torch.tensor(returns),baseline,p['temperature'],p['raw_weight_cap'],1.,torch)
        expected=read(root/'awr/report.json')['policy_update']['weights']
        assert abs(stats['effective_rows']-expected['effective_rows'])<.01
        conditions=dict(no_selected_target=torch.tensor([r['sample'].get('target',0)==0 for r in rows],dtype=torch.bool),
                        attack_no_target=torch.tensor([bool(r['sample']['attack']) and r['sample'].get('target',0)==0 for r in rows],dtype=torch.bool))
        weight_groups={}
        for label,mask in conditions.items():
            weight_groups[label]=dict(rows=int(mask.sum()),mean_weight=float(importance[mask].mean()) if bool(mask.any()) else None)
        event_index=[]
        for case in read(Path(p['corpus'])/'departure-audit.json')['cases']:
            for event in case['events']:
                if event['deep_drop']:event_index.append(locations[case['seed'],event['index']])
        event_tensor=torch.tensor(event_index,dtype=torch.long)
        weight_groups['deep_departures']=dict(rows=len(event_index),weights=importance[event_tensor].cpu().tolist())
    save(args.out,dict(state='complete',device='cuda',split='TRAIN',models=results,awr_weight_groups=weight_groups,
         inputs=inputs,source_self_kl_within_1e_6=True,script_sha256=sha(__file__),training_performed=False,promotion=False,
         scope='Exact conditional policy KL at recorded behavior movement draws, summing targets/modes. Group shares describe final fixed updates, not per-head gradient causality or measured native quality.'))
    print(json.dumps(dict(models=results,awr_weight_groups=weight_groups),indent=2))


if __name__=='__main__':main()

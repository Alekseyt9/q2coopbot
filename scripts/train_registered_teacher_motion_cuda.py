"""CUDA coordinated ground motion/aim BC; no runtime teacher or CPU fallback."""
import argparse
import copy
import gzip
import json
import pathlib
import sys
import time
from process_combat_architecture_pool import read,save,sha
from train_registered_joint_decisions_cuda import groups


def main():
    ap=argparse.ArgumentParser()
    for name in ('source','labels','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    ap.add_argument('--steps',type=int,default=200)
    args=ap.parse_args();assert not args.out.exists() and 1<=args.steps<=200
    import torch
    assert torch.cuda.is_available(),'CUDA required; no CPU fallback'
    torch.set_default_device('cuda');torch.set_num_threads(2);torch.manual_seed(20261022)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    from migrate_combat_navigation_cuda import branch
    from combat_precision_head import selected_means,mode_distribution
    from combat_sequence_retention import policy_kl
    from combat_spatial_aim import export as export_spatial
    from combat_weapon_state_residual import optimizer as actor_optimizer
    from fork_conditional_weapon_exploration_cuda import equal_tree
    from ppo_recurrent import durable_json,durable_write
    pins={}
    def pin(path,expected=None):
        path=pathlib.Path(path).resolve();digest=sha(path);assert expected is None or expected==digest
        pins[str(path)]=digest;return path
    seal=read(pin(args.source/'complete.json'))
    for name,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:pin(args.source/name,seal[key])
    model=read(args.source/'weights.json');assert model['feature_version']=='combat_features_v10' and model.get('attention') and not model.get('shared_target')
    assert model['sampled_motion_aim']['version']=='combat_sampled_motion_aim_v1'
    cp=torch.load(args.source/'checkpoint.pt',map_location='cuda',weights_only=True)
    actor=branch(model,'actor');assert equal_tree(actor.state_dict(),cp['actor']) and equal_tree(branch(model,'value').state_dict(),cp['value'])
    assert cp['weights_sha256']==seal['weights_sha256']
    std=torch.nn.Parameter(cp['log_std'].detach().clone(),requires_grad=False)
    initial={k:v.detach().clone() for k,v in actor.state_dict().items()}
    teacher_motion=copy.deepcopy(actor.sampled_motion_aim_branch)
    for p in teacher_motion.parameters():p.requires_grad_(False)
    selected={};rowmask=torch.zeros(81,dtype=torch.bool,device='cuda');rowmask[:2]=True;rowmask[63:81]=True
    for module in (actor.base.head,actor.base.residual):
        for p in (module.weight,module.bias):
            mask=torch.zeros_like(p,dtype=torch.bool);mask[rowmask]=True;selected[p]=mask
    for p in actor.sampled_motion_aim_branch.parameters():selected[p]=torch.ones_like(p,dtype=torch.bool)
    for p in actor.parameters():
        p.requires_grad_(p in selected)
        if p in selected:p.register_hook(lambda gradient,mask=selected[p]:gradient.masked_fill(~mask,0))
    opt=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],lr=.0003)
    frozen={p:p.detach().clone() for p in actor.parameters()}
    def project():
        with torch.no_grad():
            for p in actor.parameters():
                if p not in selected:p.copy_(frozen[p])
                else:p[~selected[p]]=frozen[p][~selected[p]]
    meta=read(pin(args.labels/'complete.json'));assert meta['state']=='complete' and meta['version']=='registered_teacher_ground_motion_labels_v1' and meta['test_deferred']
    for item in meta['inputs']:pin(item['path'],item['sha256'])
    repo=pathlib.Path(__file__).resolve().parents[1]
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and pathlib.Path(path).suffix=='.py' and pathlib.Path(path).resolve().is_relative_to(repo/'scripts'):pin(path)
    pin(__file__)
    data={};seen=set()
    with torch.no_grad():
        for split,info in meta['splits'].items():
            with gzip.open(pin(info['path'],info['sha256']),'rt',encoding='utf-8') as stream:rows=list(map(json.loads,stream))
            seeds={r['seed'] for r in rows};assert not seen.intersection(seeds);seen.update(seeds)
            segments=groups(rows);length=max(map(len,segments));x=torch.zeros((len(segments),length,1147),device='cuda')
            positions=[];ordered=[]
            for b,segment in enumerate(segments):
                x[b,:len(segment)]=torch.tensor([r['features'] for r in segment],dtype=torch.float32,device='cuda')
                positions.extend((b,t) for t in range(len(segment)));ordered.extend(segment)
            ix=torch.tensor(positions,dtype=torch.long,device='cuda');reference=actor(x)[0][ix[:,0],ix[:,1]].detach()
            features=x[ix[:,0],ix[:,1]];members=[i for i,r in enumerate(ordered) if r['label'] is not None]
            labels=[ordered[i]['label'] for i in members];label_ix=torch.tensor(members,dtype=torch.long,device='cuda')
            movement=torch.tensor([l['movement'] for l in labels],dtype=torch.float32,device='cuda')
            # Exact boundary +/-1 has no finite atanh. Only the conditional
            # branch input is clipped by1e-6; physical MSE still targets +/-1.
            move_latent=movement.clamp(-1+1e-6,1-1e-6).atanh()
            target=torch.tensor([l['target'] for l in labels],dtype=torch.long,device='cuda')
            mode=torch.tensor([l['mode'] for l in labels],dtype=torch.long,device='cuda')
            angles=torch.tensor([l['aim_degrees'] for l in labels],dtype=torch.float32,device='cuda')
            scale=torch.where(mode==1,15.,180.)[:,None]
            moving=torch.tensor([l['kind']=='moving' for l in labels],dtype=torch.bool,device='cuda')
            stride=max(1,(len(ordered)+127)//128);reg=torch.arange(0,len(ordered),stride,device='cuda')
            # This is a sample from the frozen source policy, NOT a teacher
            # command masquerading as an on-policy Gaussian movement draw.
            draw=(reference[reg,:2]+std[:2].exp()*torch.randn((len(reg),2),device='cuda')).detach()
            data[split]=dict(x=x,ix=ix,features=features,reference=reference,label_ix=label_ix,movement=movement,
                move_latent=move_latent,target=target,mode=mode,angles=angles,scale=scale,moving=moving,reg=reg,draw=draw,
                rows=len(ordered),labels=len(labels),segments=len(segments),seeds=sorted(seeds),
                retention_keys=[dict(seed=ordered[i]['seed'],identity=ordered[i]['identity']) for i in range(0,len(ordered),stride)])
    def losses(split):
        d=data[split];raw=actor(d['x'])[0][d['ix'][:,0],d['ix'][:,1]];r=raw[d['label_ix']]
        control_error=(r[:,:2].tanh()-d['movement']).square().mean(-1)
        motion_loss=.5*(control_error[d['moving']].mean()+control_error[~d['moving']].mean())
        means=selected_means(r,d['target'],d['mode'],d['features'][d['label_ix']],d['move_latent'],actor.sampled_motion_aim_branch)
        desired=(d['angles']/d['scale']).atanh()
        aim_loss=torch.nn.functional.smooth_l1_loss(means[:,2:4],desired,beta=.1)
        md=mode_distribution(r,d['target']);ce=-md.log_prob(d['mode'])
        mode_loss=torch.stack([ce[d['mode']==v].mean() for v in d['mode'].unique()]).mean()
        reg=d['reg'];kl=policy_kl(raw[reg],std,d['reference'][reg],std,d['features'][reg],
            actor.sampled_motion_aim_branch,teacher_motion,d['draw']).mean()
        total=motion_loss+aim_loss+.2*mode_loss+.02*kl
        angle_error=(means[:,2:4].tanh()*d['scale']-d['angles']).square().mean(-1)
        metrics=dict(movement_balanced_rmse=float(motion_loss.detach().sqrt()),
            moving_rmse=float(control_error[d['moving']].mean().detach().sqrt()),hold_rmse=float(control_error[~d['moving']].mean().detach().sqrt()),
            aim_rmse_degrees=float(angle_error.mean().detach().sqrt()),
            aim_fine_rmse_degrees=float(angle_error[d['mode']==1].mean().detach().sqrt()),
            aim_coarse_rmse_degrees=float(angle_error[d['mode']==0].mean().detach().sqrt()),
            mode_balanced_accuracy=float(torch.stack([(md.probs.argmax(-1)[d['mode']==v]==v).float().mean() for v in d['mode'].unique()]).mean()),
            retention_policy_kl=float(kl.detach()))
        return total,metrics,raw
    args.out.mkdir(parents=True)
    save(args.out/'protocol.json',dict(version='registered_teacher_ground_motion_bc_v1',source=str(args.source.resolve()),labels=str(args.labels.resolve()),
        steps=args.steps,learning_rate=.0003,seed=20261022,movement_weight=1,aim_latent_huber_weight=1,mode_weight=.2,retention_weight=.02,
        selection_on_development=False,checkpoint='fixed final',teacher_forced_movement_epsilon=1e-6,
        trainable_scope='base motion rows0:2 and mode63:81 plus sampled-motion aim branch; encoder/other parameter rows/value/std frozen',
        retention_scope='Fixed source Gaussian movement draws on <=128 train teacher histories; exact target/mode weighting, no validation in fitting',
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()]))
    with torch.no_grad():before={s:losses(s)[1] for s in data}
    history=[];start=time.perf_counter()
    for step in range(args.steps):
        opt.zero_grad();loss,_,_=losses('train');assert bool(torch.isfinite(loss));loss.backward()
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in actor.parameters())
        torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad],1.);opt.step();project()
        if (step+1)%25==0 or step+1==args.steps:
            with torch.no_grad():history.append(dict(step=step+1,**{s:losses(s)[1] for s in data}))
            save(args.out/'progress.json',dict(stage='cuda-training',step=step+1,steps=args.steps))
    with torch.no_grad():
        after={s:losses(s)[1] for s in data};errors={}
        for split in data:
            raw=losses(split)[2];error=float((raw[:,2:29]-data[split]['reference'][:,2:29]).abs().max());assert error==0.;errors[split]=error
        assert torch.equal(std,cp['log_std'])
    state=actor.state_dict();allowed=('base.head.weight','base.head.bias','base.residual.weight','base.residual.bias')
    for k,v in state.items():
        if k in allowed:assert torch.equal(v[~rowmask],initial[k][~rowmask])
        elif not k.startswith('sampled_motion_aim_branch.'):assert torch.equal(v,initial[k]),k
    updated=copy.deepcopy(model);base,cell=actor.export();updated['actor']=base;updated['attention']['actor']=cell;export_spatial(updated,actor)
    assert equal_tree(branch(updated,'actor').state_dict(),state)
    durable_json(args.out/'weights.json',updated)
    fork=copy.deepcopy(cp);fork.update(actor={k:v.detach().clone() for k,v in state.items()},weights_sha256=sha(args.out/'weights.json'))
    fork['actor_optimizer']=actor_optimizer(actor,std,cp['config']['actor_lr'],cp.get('actor_training',{}).get('weapon_state_residual',{}).get('learning_rate',.0003)).state_dict()
    fork['teacher_motion_bc']=dict(protocol_sha256=sha(args.out/'protocol.json'),steps=args.steps,actor_adam_reset=True,requires_fresh_own_policy_rollouts=True)
    durable_write(args.out/'checkpoint.pt',lambda f:torch.save(fork,f))
    assert equal_tree(torch.load(args.out/'checkpoint.pt',map_location='cuda',weights_only=True),fork)
    for p,h in pins.items():assert sha(p)==h
    durable_write(args.out/'trainer.py',lambda f:f.write(pathlib.Path(__file__).read_bytes()))
    report=dict(state='complete',device='cuda',training_performed=True,promotion=False,steps=args.steps,seconds=time.perf_counter()-start,
        before=before,after=after,history=history,frozen_raw_2_29_max_errors=errors,
        changed_actor_tensors=[k for k,v in state.items() if not torch.equal(v,initial[k])],
        rows={s:{k:d[k] for k in ('rows','labels','segments','seeds','retention_keys')} for s,d in data.items()},
        actor_adam_reset=True,critic_unchanged=True,std_unchanged=True,source_weights_sha256=seal['weights_sha256'],
        source_checkpoint_sha256=seal['checkpoint_sha256'],protocol_sha256=sha(args.out/'protocol.json'),script_sha256=sha(__file__),
        scope='Fixed-budget CUDA coordinated grounded teacher BC. Shared spatial aim weights frozen but effective spatial output can change with movement inputs. Teacher forced sampled-motion input, source Gaussian draws for KL. No vertical training, live quality or model promotion.')
    durable_json(args.out/'report.json',report)
    durable_json(args.out/'complete.json',dict(version='combat_update_complete_v1',weights_sha256=sha(args.out/'weights.json'),checkpoint_sha256=sha(args.out/'checkpoint.pt'),report_sha256=sha(args.out/'report.json')))
    save(args.out/'progress.json',dict(stage='complete',steps=args.steps))
    print(json.dumps(dict(state='complete',before=before,after=after,frozen_errors=errors)))


if __name__=='__main__':main()

"""CUDA-only teacher BC for target/fire/weapon; frozen motion/aim/encoder.

Recompute causal attention on full chronological sequences, preserving gaps.
No teacher trajectories are treated as on-policy PPO experience.
"""
import argparse
import copy
import gzip
import json
import pathlib
import time

from process_combat_architecture_pool import read, save, sha


def groups(rows):
    result = []; previous = None
    for row in rows:
        i = row['identity']
        key = (row['seed'],) + tuple(i[k] for k in ('map','connection','spawncount','actor','life'))
        if previous is None or key != previous[0] or i['frame'] != previous[1] + 1:
            result.append([])
        result[-1].append(row); previous = key, i['frame']
    return result


def main():
    ap = argparse.ArgumentParser()
    for name in ('source','data','out'): ap.add_argument('--'+name,type=pathlib.Path,required=True)
    ap.add_argument('--steps',type=int,default=200)
    args = ap.parse_args(); assert not args.out.exists() and 1 <= args.steps <= 200
    import torch
    assert torch.cuda.is_available(), 'CUDA required; no CPU fallback'
    torch.set_default_device('cuda'); torch.set_num_threads(2); torch.manual_seed(20261022)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    from migrate_combat_navigation_cuda import branch
    from fork_conditional_weapon_exploration_cuda import equal_tree
    from combat_target_head_scope import freeze_decision_scope
    from combat_target_head import distribution as target_distribution
    from combat_weapon_head import distribution as weapon_distribution, feature_mask
    from combat_spatial_aim import export as export_spatial
    from combat_weapon_state_residual import optimizer as actor_optimizer
    from ppo_recurrent import durable_json, durable_write
    pins = {}
    def pin(path, expected=None):
        path=pathlib.Path(path).resolve(); digest=sha(path)
        assert expected is None or expected==digest
        pins[str(path)]=digest; return path
    seal=read(pin(args.source/'complete.json'))
    for name,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]: pin(args.source/name,seal[key])
    assert read(args.source/'report.json')['device']=='cuda'
    model=read(args.source/'weights.json'); assert model['feature_version']=='combat_features_v10' and model.get('attention') and model.get('sampled_motion_aim')
    cp=torch.load(args.source/'checkpoint.pt',map_location='cuda',weights_only=True)
    actor=branch(model,'actor'); value=branch(model,'value')
    assert equal_tree(actor.state_dict(),cp['actor']) and equal_tree(value.state_dict(),cp['value'])
    assert cp['weights_sha256']==seal['weights_sha256']
    std=torch.nn.Parameter(torch.tensor(model['log_std'],dtype=torch.float32,device='cuda'))
    before_state={k:v.detach().clone() for k,v in actor.state_dict().items()}
    # Fresh BC Adam; no PPO momentum is used for supervised updates.
    opt=torch.optim.Adam(list(actor.parameters())+[std],lr=.001)
    project,trainable=freeze_decision_scope(actor,std,opt,'decisions')
    meta=read(pin(args.data/'complete.json')); assert meta['state']=='complete' and meta['version']=='registered_joint_teacher_features_v1' and meta['test_deferred']
    for binding in meta['inputs']: pin(binding['path'],binding['sha256'])
    data={}; seen=set()
    with torch.no_grad():
        for split in ('train','development'):
            info=meta['splits'][split]
            with gzip.open(pin(info['path'],info['sha256']),'rt',encoding='utf-8') as stream: rows=list(map(json.loads,stream))
            seeds={r['seed'] for r in rows}; assert not seen.intersection(seeds); seen.update(seeds)
            segments=groups(rows); length=max(map(len,segments))
            x=torch.zeros((len(segments),length,1147),device='cuda'); positions=[]; ordered=[]
            for b,segment in enumerate(segments):
                x[b,:len(segment)]=torch.tensor([r['features'] for r in segment],dtype=torch.float32,device='cuda')
                positions.extend((b,t) for t in range(len(segment))); ordered.extend(segment)
            ix=torch.tensor(positions,dtype=torch.long,device='cuda')
            raw=actor(x)[0][ix[:,0],ix[:,1]].detach()
            features=x[ix[:,0],ix[:,1]]
            targets=torch.tensor([r['target'] for r in ordered],dtype=torch.long,device='cuda')
            weapons=torch.tensor([r['weapon'] for r in ordered],dtype=torch.long,device='cuda')
            attack=torch.tensor([r['attack'] for r in ordered],dtype=torch.float32,device='cuda')
            masks={k:torch.tensor([r[k+'_mask'] for r in ordered],dtype=torch.bool,device='cuda') for k in ('target','weapon','attack')}
            for r in ordered:
                if any(r[k+'_mask'] for k in masks): assert r['native_matched'] and r['native_exclusive']
            td=target_distribution(raw[:,:45],features); wd=weapon_distribution(raw[:,8:20],feature_mask(features[:,:845]))
            assert bool(torch.isfinite(td.log_prob(targets)[masks['target']]).all())
            assert bool(torch.isfinite(wd.log_prob(weapons)[masks['weapon']]).all())
            data[split]=dict(x=x,ix=ix,features=features,targets=targets,weapons=weapons,attack=attack,masks=masks,
                reference=raw,segments=len(segments),rows=len(ordered),seeds=sorted(seeds))
    selected=torch.zeros(81,dtype=torch.bool,device='cuda'); selected[4]=True; selected[8:29]=True
    def evaluate(split):
        d=data[split]; raw=actor(d['x'])[0][d['ix'][:,0],d['ix'][:,1]]
        td=target_distribution(raw[:,:45],d['features']); wd=weapon_distribution(raw[:,8:20],feature_mask(d['features'][:,:845]))
        t=d['masks']['target']; w=d['masks']['weapon']; a=d['masks']['attack']
        target_loss=-td.log_prob(d['targets'])[t].mean()
        weapon_lp=-wd.log_prob(d['weapons']); present=d['weapons'][w].unique()
        # Keep is common; equal weight per represented request class.
        weapon_loss=torch.stack([weapon_lp[w & (d['weapons']==v)].mean() for v in present]).mean()
        bce=torch.nn.functional.binary_cross_entropy_with_logits(raw[:,4],d['attack'],reduction='none')
        fire_loss=.5*(bce[a & (d['attack']==0)].mean()+bce[a & (d['attack']==1)].mean())
        old=d['reference']; oldtd=target_distribution(old[:,:45],d['features']); oldwd=weapon_distribution(old[:,8:20],feature_mask(d['features'][:,:845]))
        kl=(torch.distributions.kl_divergence(oldtd,td)+torch.distributions.kl_divergence(oldwd,wd)+
            torch.distributions.kl_divergence(torch.distributions.Bernoulli(logits=old[:,4]),torch.distributions.Bernoulli(logits=raw[:,4]))).mean()
        loss=target_loss+weapon_loss+fire_loss+.1*kl
        metrics=dict(target_cross_entropy=float(target_loss.detach()),target_accuracy=float((td.probs.argmax(-1)[t]==d['targets'][t]).float().mean()),
            weapon_macro_cross_entropy=float(weapon_loss.detach()),weapon_macro_accuracy=float(torch.stack([(wd.probs.argmax(-1)[w & (d['weapons']==v)]==v).float().mean() for v in present]).mean()),
            attack_balanced_bce=float(fire_loss.detach()),attack_balanced_accuracy=float(.5*((raw[:,4][a & (d['attack']==0)]<0).float().mean()+(raw[:,4][a & (d['attack']==1)]>=0).float().mean())),
            decision_kl=float(kl.detach()))
        return loss,metrics,raw
    args.out.mkdir(parents=True)
    protocol=dict(version='joint_decision_bc_protocol_v1',source=str(args.source.resolve()),data=str(args.data.resolve()),steps=args.steps,
        learning_rate=.001,retention_weight=.1,seed=20261022,scope='projected attack/weapon/target output rows and shared target branch only',
        development_used_for_selection=False,checkpoint='fixed final',movement_supervision=False,vertical_supervision=False,
        actor_optimizer_reset_for_future_ppo=True,training_performed=False,inputs=[dict(path=p,sha256=h) for p,h in pins.items()])
    save(args.out/'protocol.json',protocol)
    with torch.no_grad(): before={s:evaluate(s)[1] for s in data}
    history=[]; start=time.perf_counter()
    for step in range(args.steps):
        opt.zero_grad(); loss,_,_=evaluate('train'); assert bool(torch.isfinite(loss)); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad],1.)
        opt.step(); project()
        if (step+1)%25==0 or step+1==args.steps:
            with torch.no_grad(): history.append(dict(step=step+1,**{s:evaluate(s)[1] for s in data}))
            save(args.out/'progress.json',dict(stage='cuda-training',step=step+1,steps=args.steps))
    with torch.no_grad():
        after={s:evaluate(s)[1] for s in data}
        errors={}
        for s in data:
            raw=evaluate(s)[2]; error=float((raw[:,~selected]-data[s]['reference'][:,~selected]).abs().max())
            assert error==0.,(s,error); errors[s]=error
        assert torch.equal(std,cp['log_std'])
    updated=copy.deepcopy(model); base,cell=actor.export(); updated['actor']=base; updated['attention']['actor']=cell; export_spatial(updated,actor)
    rebuilt=branch(updated,'actor'); assert equal_tree(rebuilt.state_dict(),actor.state_dict())
    new_state=actor.state_dict(); changed=[k for k,v in new_state.items() if not torch.equal(v,before_state[k])]
    assert changed
    durable_json(args.out/'weights.json',updated)
    fork=copy.deepcopy(cp); fork['actor']={k:v.detach().clone() for k,v in new_state.items()}; fork['weights_sha256']=sha(args.out/'weights.json')
    # Clear all actor Adam state explicitly after BC. Critic/std/data-consumption
    # history remain unchanged; further PPO requires fresh student rollouts.
    fresh=actor_optimizer(actor,std,cp['config']['actor_lr'],cp.get('actor_training',{}).get('weapon_state_residual',{}).get('learning_rate',.0003))
    fork['actor_optimizer']=fresh.state_dict()
    fork['joint_decision_bc']=dict(protocol_sha256=sha(args.out/'protocol.json'),steps=args.steps,actor_adam_reset=True,requires_fresh_own_policy_rollouts=True)
    durable_write(args.out/'checkpoint.pt',lambda output:torch.save(fork,output))
    assert equal_tree(torch.load(args.out/'checkpoint.pt',map_location='cuda',weights_only=True),fork)
    for path,digest in pins.items(): assert sha(path)==digest
    report=dict(device='cuda',training_performed=True,promotion=False,before=before,after=after,history=history,steps=args.steps,seconds=time.perf_counter()-start,
        trainable_parameters=trainable,changed_actor_tensors=changed,frozen_nondecision_raw_max_error=errors,source=str(args.source.resolve()),
        teacher_cases=meta['cases'],data_counts={s:{k:data[s][k] for k in ('rows','segments','seeds')} for s in data},
        actor_adam_reset=True,critic_unchanged=True,std_unchanged=True,weights_sha256=sha(args.out/'weights.json'),script_sha256=sha(__file__),
        scope='CUDA-only fixed endpoint supervised target/fire/weapon pilot on fresh rules train data. Development metrics only, no selection. Frozen encoder/movement/vertical/aim/mode/spatial/weapon-state/std/value; target changes can indirectly change live trajectories and history. No live quality claim.')
    durable_json(args.out/'report.json',report)
    durable_json(args.out/'complete.json',dict(version='combat_update_complete_v1',weights_sha256=sha(args.out/'weights.json'),checkpoint_sha256=sha(args.out/'checkpoint.pt'),report_sha256=sha(args.out/'report.json')))
    save(args.out/'progress.json',dict(stage='complete',steps=args.steps))
    print(json.dumps(dict(state='complete',before=before,after=after,frozen_errors=errors)))


if __name__=='__main__': main()

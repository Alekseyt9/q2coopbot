"""Offline CUDA residual-value probe; never exports or updates a combat policy."""
import argparse,copy,hashlib,json
from pathlib import Path
from collections import Counter
from ppo_combat import torch,nn,training_devices,advantages
from train_combat_full_joint_ppo_cuda import admit
from combat_wide_aim import events,geometry
from combat_conditional_fire import inputs as fire_inputs
from evaluate_registered_combat_models import read,save,sha

VERSION='combat_pre_gate_baseline_probe_v1'

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batch',type=Path,action='append',required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--ridge',action='store_true',help='Post-hoc low-capacity probe; regularization selected only inside TRAIN maps')
    a=ap.parse_args();out=a.out.resolve();assert not out.exists()
    training_devices();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path);assert expected is None or digest==expected,str(path)
        pins[str(path)]=digest;return path
    pin(Path(__file__))
    collected=[];maps=set();seeds=set()
    for generation,batch in enumerate(a.batch):
        protocol=read(pin(batch/'protocol.json'));source=Path(protocol['source']);model=read(pin(source/'weights.json'));config=protocol['config']
        rows,_,_,case_seeds,_=admit(read(pin(batch/'train-request.json')),sha(source/'weights.json'),[],[],pin)
        assert not seeds&case_seeds;seeds|=case_seeds
        plan=read(batch/'train-plan.json');names={t['seeds'][0]:t['episode']['map'] for t in plan['tasks']};maps.update(names.values())
        collected.append((generation,rows,model,config,names))
    assert len(seeds)==48 and len(maps)==32
    ordered=sorted(maps,key=lambda name:hashlib.sha256(('wide-baseline-20261026:'+name).encode()).hexdigest())
    held=set(ordered[:8]);out.mkdir(parents=True)
    save(out/'protocol.json',dict(version=VERSION,state='frozen',holdout_maps=sorted(held),train_maps=sorted(maps-held),
        split='Fixed SHA map order before labels/fitting; all rows of same map remain on same side. Actor has previously trained these maps: internal baseline holdout, not globally unseen policy maps.',
        seeds=[41,42,43],steps=300,batch_size=512,learning_rate=.0003,weight_decay=.0001,architecture='same input width ->64 ReLU->32 ReLU->1; state-only context channels zeroed; identical initial weights and minibatches',
        target='GAE return minus native old value; common old value and behavior-generation indicator are state inputs. Train-only normalizers, whole-case weighting.',
        extra_inputs='181 pre-gate conditional-fire context from observation, selected target/mode, pre-wide source latent, vertical/weapon; current kick4 and observed anchor2. No current wide choice, final wide latent, attack, next_features, reward or future as inputs.',
        advance='Median eligible heldout case-weighted MSE ratio conditional/state <=.95, conditional lower in at least2/3seeds, median gradient-score second moment no higher, conditional MSE below old zero residual.',
        actor_training=False,diagnostic_only=True,promotion=False,inputs=pins))
    states=[];contexts=[];targets=[];active=[];choices=[];probabilities=[];row_maps=[];row_seeds=[]
    for generation,rows,model,config,names in collected:
        x=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float64)
        cat=lambda k:torch.tensor([r['sample'].get(k,0) for r in rows],device='cuda',dtype=torch.long)
        target,mode,vertical,weapon=(cat(k) for k in ('target','aim_mode','vertical','weapon'))
        d=events(model,rows,x,target,mode,vertical,weapon)
        # Both helpers see only the original pre-wide event. Choice and attack
        # are retained for diagnostic metric weighting, never for predictors.
        anchor,_=geometry(x,target,mode)
        ctx=torch.cat((fire_inputs(x,target,mode,d['source'],vertical,weapon),x[:,810:814],anchor/180),-1)
        assert ctx.shape==(len(rows),187)
        old_value=torch.tensor([r['sample']['value'] for r in rows],device='cuda',dtype=torch.float64)
        states.append(torch.cat((x,old_value[:,None],torch.full((len(rows),1),generation,device='cuda',dtype=torch.float64)),1).float())
        contexts.append(ctx.float());targets.extend(advantages(rows,config['gamma'],config['lambda'])[0]);active.append(d['active']);choices.append(d['choice'])
        probabilities.extend([1/(1+__import__('math').exp(-r['sample'].get('wide_aim',{}).get('logit',0.))) for r in rows])
        row_maps.extend(names[r['seed']] for r in rows);row_seeds.extend(r['seed'] for r in rows)
    s=torch.cat(states);ctx=torch.cat(contexts);y=torch.tensor(targets,device='cuda',dtype=torch.float32)
    mask=torch.cat(active);choice=torch.cat(choices).float();p=torch.tensor(probabilities,device='cuda');score2=(choice-p).square()
    train=torch.tensor([m not in held for m in row_maps],device='cuda');test=~train
    assert int((mask&train).sum())>=100 and int((mask&test).sum())>=30
    n=len(y);counts=Counter(seed for seed,tr in zip(row_seeds,train.tolist()) if tr)
    w=torch.tensor([n/(len(counts)*counts[seed]) if seed in counts else 0. for seed in row_seeds],device='cuda')
    normal=lambda x:((x-x[train].mean(0))/(x[train].std(0,unbiased=False).clamp_min(.001))).clamp(-10,10)
    sn,cn=normal(s),normal(ctx);mean=y[train].mean();scale=y[train].std(unbiased=False).clamp_min(.001);yn=(y-mean)/scale
    datasets={'state_only':torch.cat((sn,torch.zeros_like(cn)),1),'action_conditioned':torch.cat((sn,cn),1)}
    train_indices=torch.where(train)[0]
    def metric(pred,select):
        err=(y-pred).square();per=[];second=[]
        for seed in sorted(set(row_seeds)):
            selected=select&torch.tensor([v==seed for v in row_seeds],device='cuda')
            if bool(selected.any()):per.append(err[selected].mean());second.append((err[selected]*score2[selected]).mean())
        return dict(cases=len(per),rows=int(select.sum()),mse=float(torch.stack(per).mean()),score_second_moment=float(torch.stack(second).mean()))
    reference={label:metric(pred,mask&test) for label,pred in (('native_value_zero_residual',torch.zeros_like(y)),('train_constant',mean.expand_as(y)))}
    if a.ridge:
        spec=read(out/'protocol.json')
        spec.update(version='combat_pre_gate_ridge_baseline_probe_v1',steps=0,seeds=[],architecture='Linear CUDA weighted dual ridge; common state plus zero187 or pre-gate187 context; train-fold-only normalization and target centering',
            regularization=[.001,.01,.1,1.,10.],inner_folds='Three fixed SHA-map folds within24 outer TRAIN maps, no holdout targets/normalizers used for selection',
            advance='Conditional eligible heldout case-weighted MSE <=.95*state-only and below old zero baseline; score second moment no higher than state-only',
            scope='Post-hoc capacity refinement after MLP overfit. Same internal outer validation maps reused, not independent final confirmation.')
        save(out/'protocol.json',spec)
        inner_order=sorted(maps-held,key=lambda m:hashlib.sha256(('wide-ridge-inner:'+m).encode()).hexdigest())
        inner_groups=[set(inner_order[i::3]) for i in range(3)]
        raw={'state_only':torch.cat((s,torch.zeros_like(ctx)),1),'action_conditioned':torch.cat((s,ctx),1)}
        def predict(raw_x,fit_mask,ridge):
            fit_ids=torch.where(fit_mask)[0];xx=raw_x[fit_ids]
            mu=xx.mean(0);sd=xx.std(0,unbiased=False).clamp_min(.001)
            xx=((xx-mu)/sd).clamp(-10,10)/(raw_x.shape[1]**.5)
            case_counts=Counter(row_seeds[i] for i in fit_ids.tolist())
            weights=torch.tensor([1/(len(case_counts)*case_counts[row_seeds[i]]) for i in fit_ids.tolist()],device='cuda')
            yy=y[fit_ids];center=(weights*yy).sum();sw=weights.sqrt()
            weighted_x=xx*sw[:,None]
            matrix=weighted_x@weighted_x.T+ridge*torch.eye(len(fit_ids),device='cuda')
            dual=torch.linalg.solve(matrix,(yy-center)*sw)
            coefficient=weighted_x.T@dual
            all_x=((raw_x-mu)/sd).clamp(-10,10)/(raw_x.shape[1]**.5)
            return all_x@coefficient+center,dict(mean=mu,std=sd,coefficient=coefficient,center=center)
        results={}
        for name,raw_x in raw.items():
            scores=[]
            for ridge in (.001,.01,.1,1.,10.):
                losses=[]
                for group in inner_groups:
                    inner_val=torch.tensor([m in group for m in row_maps],device='cuda')&mask
                    inner_train=train&mask&~inner_val
                    assert int(inner_val.sum())>0 and int(inner_train.sum())>30
                    pred,_=predict(raw_x,inner_train,ridge);losses.append(metric(pred,inner_val)['mse'])
                scores.append(dict(ridge=ridge,inner_fold_mse=losses,mean_inner_mse=sum(losses)/3))
            best=min(scores,key=lambda r:(r['mean_inner_mse'],-r['ridge']))
            pred,fit=predict(raw_x,train&mask,best['ridge'])
            results[name]=dict(selected_ridge=best['ridge'],selection=scores,train=metric(pred,train&mask),heldout=metric(pred,test&mask))
            torch.save(dict(version=spec['version'],diagnostic_only=True,actor_compatible=False,fit=fit),out/(name+'-ridge.pt'))
        left,right=(results[k]['heldout'] for k in ('state_only','action_conditioned'))
        ratio=right['mse']/left['mse'];second=right['score_second_moment']/left['score_second_moment']
        advance=ratio<=.95 and second<=1 and right['mse']<reference['native_value_zero_residual']['mse']
        report=dict(state='complete',version=spec['version'],device='cuda',diagnostic_only=True,actor_training=False,promotion=False,posthoc_refinement=True,
            rows=n,active_rows=int(mask.sum()),reference=reference,results=results,mse_ratio=ratio,score_second_moment_ratio=second,advance=advance,
            scope=spec['scope'],holdout_maps=sorted(held),inner_map_folds=[sorted(g) for g in inner_groups])
        for path,digest in pins.items():assert sha(path)==digest
        save(out/'report.json',report);save(out/'complete.json',dict(state='complete',report_sha256=sha(out/'report.json'),protocol_sha256=sha(out/'protocol.json'),diagnostic_only=True,actor_training=False,promotion=False))
        save(out/'progress.json',dict(stage='complete'));print(json.dumps(report));return
    results=[]
    for seed in (41,42,43):
        torch.manual_seed(seed)
        make=lambda:nn.Sequential(nn.Linear(datasets['state_only'].shape[1],64,device='cuda'),nn.ReLU(),nn.Linear(64,32,device='cuda'),nn.ReLU(),nn.Linear(32,1,device='cuda'))
        first=make()
        with torch.no_grad():first[-1].weight.zero_();first[-1].bias.zero_()
        networks={'state_only':first,'action_conditioned':copy.deepcopy(first)}
        optimizers={k:torch.optim.AdamW(v.parameters(),lr=.0003,weight_decay=.0001) for k,v in networks.items()}
        generator=torch.Generator(device='cuda').manual_seed(seed+1000)
        for step in range(300):
            ids=train_indices[torch.randint(len(train_indices),(512,),device='cuda',generator=generator)]
            for name,net in networks.items():
                loss=((net(datasets[name][ids])[:,0]-yn[ids]).square()*w[ids]).mean();assert bool(torch.isfinite(loss))
                optimizers[name].zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);optimizers[name].step()
            if step%100==99:save(out/'progress.json',dict(stage='fitting',seed=seed,step=step+1))
        rr=dict(seed=seed,variants={})
        for name,net in networks.items():
            with torch.no_grad():pred=net(datasets[name])[:,0]*scale+mean
            rr['variants'][name]=dict(train=metric(pred,mask&train),heldout=metric(pred,mask&test))
            torch.save(dict(version=VERSION,diagnostic_only=True,actor_compatible=False,seed=seed,state_dict=net.state_dict()),out/f'{name}-{seed}.pt')
        results.append(rr)
    ratios=[r['variants']['action_conditioned']['heldout']['mse']/r['variants']['state_only']['heldout']['mse'] for r in results]
    second_ratios=[r['variants']['action_conditioned']['heldout']['score_second_moment']/r['variants']['state_only']['heldout']['score_second_moment'] for r in results]
    median=lambda a:sorted(a)[len(a)//2]
    conditional_mse=median([r['variants']['action_conditioned']['heldout']['mse'] for r in results])
    advance=median(ratios)<=.95 and sum(v<1 for v in ratios)>=2 and median(second_ratios)<=1 and conditional_mse<reference['native_value_zero_residual']['mse']
    for path,digest in pins.items():assert sha(path)==digest
    report=dict(state='complete',version=VERSION,device='cuda',diagnostic_only=True,actor_training=False,promotion=False,
        rows=n,active_rows=int(mask.sum()),train_cases=len(counts),holdout_cases=len(set(row_seeds))-len(counts),holdout_maps=sorted(held),reference=reference,results=results,mse_ratios=ratios,score_second_moment_ratios=second_ratios,advance=advance,
        scope='Internal map-held-out prediction probe on consumed ordinary TRAIN data. Actor unchanged. Offline baseline loss or score second moment is not native combat benefit; no policy or ordinary PPO checkpoint exported.')
    save(out/'report.json',report);save(out/'complete.json',dict(state='complete',report_sha256=sha(out/'report.json'),protocol_sha256=sha(out/'protocol.json'),diagnostic_only=True,actor_training=False,promotion=False))
    save(out/'progress.json',dict(stage='complete'));print(json.dumps(report))

if __name__=='__main__':main()

"""Shared observed-enemy H1 damage forecasting, fitting/evaluation on CUDA."""
import argparse
import copy
import math
from pathlib import Path
import numpy as np
from evaluate_registered_combat_models import read,save,sha
from train_combat_local_effect_predictor_cuda import torch,nn,training_devices,world_weights,encode as flat_encode,Forecaster as Flat


class SharedThreat(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(185,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,2)).cuda()

    def forward(self,x):
        assert x.is_cuda and x.shape[-1]==185
        return self.network(x)


def inputs(d,no_phase=False):
    assert all(v.is_cuda for v in d.values())
    f=d['features'];n=len(f);basic=f[:,73:169].reshape(n,8,12);typed=f[:,466:786].reshape(n,8,40).clone()
    mask=basic[:,:,0]>0;kind=typed[:,:,:21].argmax(-1)
    # Features already express enemies in ego axes. Rotate own world velocity
    # likewise and omit absolute ego yaw from the shared body.
    velocity=torch.stack((f[:,10]*f[:,7]+f[:,11]*f[:,6],-f[:,10]*f[:,6]+f[:,11]*f[:,7],f[:,12]),-1)
    own=torch.cat((f[:,:6],f[:,8:10],velocity,f[:,13:73],f[:,1121:1147]),-1)
    assert own.shape[1]==97
    c=d['categories'];scale=torch.where(c[:,1]==1,15./180.,1.)
    physical=[]
    for key in ('source_action','action'):
        z=d[key].tanh();physical.append(torch.cat((z[:,:2],z[:,2:]*scale[:,None]),-1))
    onehot=lambda col,count:nn.functional.one_hot(c[:,col],count).float()
    action=torch.cat((*physical,d['attack'][:,None],onehot(1,2),onehot(2,3),onehot(3,12)),-1)
    selected=(c[:,0,None]==torch.arange(1,9,device='cuda')[None,:]).float()
    no_target=(c[:,0]==0).float()[:,None,None].expand(-1,8,-1)
    phase_frame=typed[:,:,35:36]*512.;frequencies=torch.tensor([2*math.pi/v for v in (4,8,16,32)],device='cuda')
    phase=torch.cat(((phase_frame*frequencies).sin(),(phase_frame*frequencies).cos()),-1)*typed[:,:,34:35]
    if no_phase:typed[:,:,34:36]=0;phase=torch.zeros_like(phase)
    x=torch.cat((basic,typed,own[:,None,:].expand(-1,8,-1),action[:,None,:].expand(-1,8,-1),selected[:,:,None],no_target,phase),-1).clamp(-4,4)
    assert x.shape==(n,8,185) and bool(torch.isfinite(x).all())
    return x,mask,kind


def aggregate(output,mask,health,unknown):
    probability=output[:,:,0].sigmoid()*mask
    amount=nn.functional.softplus(output[:,:,1])*20.*mask
    total_probability=1.-(1.-probability).prod(-1)*(1.-unknown['probability'])
    total_amount=torch.minimum(amount.sum(-1)+unknown['amount'],health.clamp_min(0))
    return probability,amount,total_probability.clamp(1e-6,1-1e-6),total_amount


def constants(d,w,mask,kind):
    node_w=w[:,None]*mask/mask.sum(-1,keepdim=True).clamp_min(1)
    target=d['incoming_by_enemy'];event=(target>0).float()
    global_p=(event*node_w).sum()/node_w.sum();global_amount=(target*node_w).sum()/node_w.sum()
    probs=[];means=[]
    for k in range(21):
        weight=node_w*(kind==k);den=weight.sum()
        probs.append((event*weight).sum()/den if float(den)>0 else global_p)
        means.append((target*weight).sum()/den if float(den)>0 else global_amount)
    unknown=dict(probability=((d['unknown_incoming']>0)*w).sum(),amount=(d['unknown_incoming']*w).sum())
    return torch.stack(probs).clamp(1e-5,1-1e-5),torch.stack(means).clamp_min(2e-4),unknown,global_p,global_amount


def metrics(output,d,w,mask,unknown,aggregate_fn=aggregate):
    node_p,node_amount,pred_p,pred_amount=aggregate_fn(output,mask,d['features'][:,0]*100,unknown)
    actual=d['effects'][:,0,1];event=(actual>0).float();node_actual=d['incoming_by_enemy']
    node_w=w[:,None]*mask/mask.sum(-1,keepdim=True).clamp_min(1)
    return dict(received_bce=float((nn.functional.binary_cross_entropy(pred_p,event,reduction='none')*w).sum()),
        received_rmse=float(((pred_amount-actual).square()*w).sum().sqrt()),
        node_bce=float((nn.functional.binary_cross_entropy_with_logits(output[:,:,0],(node_actual>0).float(),reduction='none')*node_w).sum()/node_w.sum()),
        node_rmse=float((((node_amount-node_actual).square()*node_w).sum()/node_w.sum()).sqrt()),
        observed_mean=float((actual*w).sum()),predicted_mean=float((pred_amount*w).sum()))


def challenge(output,d,mask,unknown,roots,aggregate_fn=aggregate):
    _,_,p,amount=aggregate_fn(output,mask,d['features'][:,0]*100,unknown)
    index={(r['context_id'],r['stream_seed'],r['duration'],r['label']):i for i,r in enumerate(roots)}
    same=[];deltas={1:[],5:[]}
    for r in roots:
        i=index[r['context_id'],r['stream_seed'],r['duration'],r['label']]
        if r['duration']==1:
            j=index[r['context_id'],r['stream_seed'],5,r['label']];same.append((amount[i]-amount[j]).abs())
        if r['label']!='control':
            j=index[r['context_id'],r['stream_seed'],r['duration'],'control'];deltas[r['duration']].append(amount[i]-amount[j])
    error=torch.stack(same).max();assert float(error)<1e-6,error
    return dict(known_zero_h1_received=bool((d['effects'][:,0,1]==0).all()),same_action_duration_max_error=float(error),
        amounts_mean=float(amount.mean()),event_probability_mean=float(p.mean()),
        alternative_control={str(k):dict(dependent_pairs=len(v),predicted_delta_rmse=float(torch.stack(v).square().mean().sqrt()),
            maximum_absolute_delta=float(torch.stack(v).abs().max())) for k,v in deltas.items()},
        scope='Four inspected held-out worlds, actual H1effects zero. No claim that nonzero evasive action effects can be ranked.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--seal-only',action='store_true',help='Verify already completed CUDA fits and close metadata without optimizer steps')
    a=ap.parse_args();assert a.out.exists() if a.seal_only else not a.out.exists()
    assert not (a.out/'complete.json').exists() and not (a.out/'report.json').exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path);h=pins[key] if key in pins else sha(path)
        assert expected is None or expected==h,str(path);pins[key]=h;return path
    p=read(pin(a.protocol));assert p['version']=='combat_enemy_threat_predictor_experiment_v1'
    closed=read(pin(a.dataset/'complete.json'));ds=read(pin(a.dataset/'report.json',closed['report_sha256']))
    assert ds['version']=='combat_enemy_threat_dataset_v1' and ds['protocol_sha256']==sha(a.protocol) and ds['all_h1_totals_exact']
    for path,h in ds['inputs'].items():pin(path,h)
    worlds={k:set(v) for k,v in ds['worlds'].items()}
    assert not(worlds['train'] & worlds['development'] or worlds['train'] & worlds['challenge'] or worlds['development'] & worlds['challenge'])
    base=Path(p['base_dataset']);bc=read(pin(base/'complete.json'));bd=read(pin(base/'report.json',bc['report_sha256']))
    root_chunk=next(c for c in bd['chunks'] if c['split']=='challenge');roots=read(pin(root_chunk['case']['root_records'],root_chunk['case']['root_records_sha256']))
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data={};x={};masks={};kinds={};weights={}
    for split in ('train','development','challenge'):
        chunks=[]
        for c in ds['chunks']:
            if c['split']==split:
                with np.load(pin(c['path'],c['sha256']),allow_pickle=False) as f:chunks.append({k:f[k].copy() for k in f.files})
        data[split]={k:torch.as_tensor(np.concatenate([c[k] for c in chunks]),device='cuda') for k in chunks[0]}
        x[split],masks[split],kinds[split]=inputs(data[split]);weights[split]=world_weights(data[split])
        assert bool(data[split]['known'][:,0].all()) and bool(torch.equal(data[split]['incoming_by_enemy'].sum(-1)+data[split]['unknown_incoming'],data[split]['effects'][:,0,1]))
    probs,means,unknown,global_p,global_mean=constants(data['train'],weights['train'],masks['train'],kinds['train'])
    type_metrics={}
    for split in ('train','development'):
        kind=kinds[split];out=torch.stack((torch.logit(probs[kind]),torch.log(torch.expm1(means[kind]/20.))),-1)
        type_metrics[split]=metrics(out,data[split],weights[split],masks[split],unknown)
    flat_root=Path(p['flat_experiment']);fc=read(pin(flat_root/'complete.json'));fr=read(pin(flat_root/'report.json',fc['report_sha256']))
    cp=pin(flat_root/'state_action/checkpoint.pt',fr['results']['state_action']['checkpoint_sha256'])
    flat=Flat();flat.load_state_dict(torch.load(cp,map_location='cuda',weights_only=False)['state_dict']);flat.eval()
    with torch.no_grad():
        out=flat(flat_encode(data['development']));pred_p=out[:,0,0].sigmoid();pred_hp=nn.functional.softplus(out[:,0,4])*20.
        actual=data['development']['effects'][:,0,1];w=weights['development']
        flat_metrics=dict(received_bce=float((nn.functional.binary_cross_entropy(pred_p,(actual>0).float(),reduction='none')*w).sum()),
            received_rmse=float(((pred_hp-actual).square()*w).sum().sqrt()))
    a.out.mkdir(parents=True,exist_ok=a.seal_only);cfg=p['training'];results={};initial=None
    for name in p['models']:
        if a.seal_only:
            folder=a.out/name;saved=read(pin(folder/'report.json'))
            state=torch.load(pin(folder/'checkpoint.pt',saved['checkpoint_sha256']),map_location='cuda',weights_only=False)
            assert state['version']=='combat_shared_enemy_threat_v1' and state['steps_completed']==cfg['steps']==saved['steps_completed']
            assert state['protocol_sha256']==sha(a.protocol) and state['dataset_sha256']==closed['report_sha256']
            assert saved['gpu_reload_exact'] and saved['device']=='cuda' and all(v.is_cuda for v in state['state_dict'].values())
            completed=SharedThreat();completed.load_state_dict(state['state_dict']);completed.eval()
            xx=x if name=='shared_phase_action' else {split:inputs(d,no_phase=True)[0] for split,d in data.items()}
            with torch.no_grad():
                for split in ('train','development'):
                    score=metrics(completed(xx[split]),data[split],weights[split],masks[split],unknown)
                    actual=torch.tensor([score[k] for k in score],device='cuda')
                    expected=torch.tensor([saved['metrics'][split][k] for k in score],device='cuda')
                    assert float((actual-expected).abs().max())<1e-5
                assert challenge(completed(xx['challenge']),data['challenge'],masks['challenge'],unknown,roots)==saved['challenge']
            results[name]=saved
            continue
        torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);model=SharedThreat()
        with torch.no_grad():
            model.network[-1].weight.zero_()
            model.network[-1].bias.copy_(torch.stack((torch.logit(global_p.clamp(1e-5,1-1e-5)),torch.log(torch.expm1((global_mean/20).clamp_min(1e-5))))))
        assert sum(v.numel() for v in model.parameters())==p['architecture']['parameters']
        if initial is None:initial=copy.deepcopy(model.state_dict())
        else:assert all(torch.equal(v,initial[k]) for k,v in model.state_dict().items())
        xx=x if name=='shared_phase_action' else {split:inputs(d,no_phase=True)[0] for split,d in data.items()}
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        torch.manual_seed(cfg['seed']+1);torch.cuda.manual_seed_all(cfg['seed']+1)
        for step in range(cfg['steps']):
            ix=torch.multinomial(weights['train'],cfg['batch_size'],replacement=True);mask=masks['train'][ix]
            output=model(xx['train'][ix]);d=data['train'];y=d['incoming_by_enemy'][ix];total=d['effects'][ix,0,1]
            npred,nmean,rprob,rmean=aggregate(output,mask,d['features'][ix,0]*100,unknown)
            node_loss=nn.functional.binary_cross_entropy_with_logits(output[:,:,0],(y>0).float(),reduction='none')+((nmean-y)/20.).square()
            node_loss=(node_loss*mask).sum(-1)/mask.sum(-1).clamp_min(1)
            root_loss=nn.functional.binary_cross_entropy(rprob,(total>0).float(),reduction='none')+((rmean-total)/20.).square()
            loss=.5*(node_loss+root_loss).mean();assert bool(torch.isfinite(loss))
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);optimizer.step()
            if (step+1)%100==0:save(a.out/'progress.json',dict(stage='cuda-fitting',model=name,steps=step+1,total_steps=cfg['steps'],loss=float(loss.detach())))
        model.eval()
        with torch.no_grad():
            output={split:model(v) for split,v in xx.items()}
            scores={split:metrics(output[split],data[split],weights[split],masks[split],unknown) for split in ('train','development')}
            counterfactual=challenge(output['challenge'],data['challenge'],masks['challenge'],unknown,roots)
        folder=a.out/name;folder.mkdir();checkpoint=folder/'checkpoint.pt'
        torch.save(dict(version='combat_shared_enemy_threat_v1',state_dict=model.state_dict(),optimizer=optimizer.state_dict(),unknown=unknown,
                        steps_completed=cfg['steps'],protocol_sha256=sha(a.protocol),dataset_sha256=closed['report_sha256'],cuda_rng=torch.cuda.get_rng_state()),checkpoint)
        restored=SharedThreat();state=torch.load(checkpoint,map_location='cuda',weights_only=False);restored.load_state_dict(state['state_dict']);restored.eval()
        with torch.no_grad():assert torch.equal(restored(xx['development']),output['development'])
        result=dict(model=name,device='cuda',parameters=sum(t.numel() for t in model.parameters()),steps_completed=cfg['steps'],metrics=scores,
            challenge=counterfactual,checkpoint_sha256=sha(checkpoint),gpu_reload_exact=True)
        save(folder/'report.json',result);results[name]=result
    full=results['shared_phase_action']['metrics']['development'];q=p['predictive_admission'];checks={};ratios={}
    for field,short in [('received_bce','bce'),('received_rmse','rmse')]:
        for label,baseline in [('type_constant',type_metrics['development']),('frozen_flat_h1',flat_metrics)]:
            key=short+'_to_'+label;ratios[key]=full[field]/baseline[field];checks[key]=ratios[key]<=q['development_'+short+'_ratio_to_'+label+'_max']
    for path,h in pins.items():assert sha(path)==h,path
    report=dict(state='complete',version='combat_enemy_threat_predictor_report_v1',device='cuda',results=results,type_baseline=type_metrics,
        frozen_flat_development_h1=flat_metrics,unknown_train_constants={k:float(v) for k,v in unknown.items()},predictive_ratios=ratios,
        predictive_checks=checks,predictive_admitted=all(checks.values()),worlds=ds['worlds'],rows=ds['rows'],inputs=pins,
        recovered_completed_fits=a.seal_only,gradient_steps_added_on_recovery=0 if a.seal_only else None,
        training_performed=True,actor_training=False,new_games=False,ordinary_training_eligible=False,runtime_action_admitted=False,promotion=False,
        scope='Fixed shared-node H1 threat forecast. Predictive gate supports only fresh forecast validation/under-fire TRAIN collection; action choice and full battle superiority remain unproven.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    save(a.out/'progress.json',dict(stage='complete',predictive_admitted=report['predictive_admitted'],runtime_action_admitted=False))
    print(dict(predictive_ratios=ratios,checks=checks,predictive_admitted=report['predictive_admitted']))


if __name__=='__main__':main()

"""Fixed-budget CUDA effect forecasting; source combat policy remains frozen."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from evaluate_registered_combat_models import read,save,sha
from ppo_combat import torch,training_devices
from train_combat_bc import nn


class Forecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(1185,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,24)).cuda()

    def forward(self,x):
        assert x.is_cuda and x.shape[1]==1185
        return self.network(x).reshape(-1,4,6)


def encode(d):
    assert all(v.is_cuda for v in d.values())
    c=d['categories'];assert c.shape[1]==4
    scale=torch.where(c[:,1]==1,15./180.,1.)
    physical=[]
    for key in ('source_action','action'):
        z=d[key].tanh();physical.append(torch.cat((z[:,:2],z[:,2:]*scale[:,None]),-1))
    onehot=[]
    for index,count in enumerate((9,2,3,12)):
        assert bool(((c[:,index]>=0)&(c[:,index]<count)).all())
        onehot.append(nn.functional.one_hot(c[:,index],count).float())
    option=torch.cat((d['option'][:,:2]/2,d['option'][:,2:]/10),-1)
    x=torch.cat((d['features'].clamp(-4,4),*physical,d['attack'][:,None],*onehot,option),-1)
    assert x.shape[1]==1185 and bool(torch.isfinite(x).all())
    return x


def world_weights(d):
    # One total weight per world; repeated branch rows at a world/frame share it.
    groups,inverse,counts=torch.unique(torch.stack((d['world'],d['frame'].long()),-1),dim=0,return_inverse=True,return_counts=True)
    worlds,world_index,frame_counts=torch.unique(groups[:,0],return_inverse=True,return_counts=True)
    weights=1./(counts[inverse].float()*frame_counts[world_index[inverse]].float())
    return weights/weights.sum()


def targets(d):
    e=d['effects']
    events=torch.stack((e[:,:,1]>0,e[:,:,0]>0,e[:,:,4]>0,e[:,:,5]>0),-1).float()
    amounts=torch.stack((e[:,:,1],e[:,:,0]),-1)
    return events,amounts


def constants(d,w):
    event,amount=targets(d);weight=d['known'].float()*w[:,None]
    denominator=weight.sum(0)[:,None]
    probabilities=((event*weight[:,:,None]).sum(0)/denominator).clamp(1e-5,1-1e-5)
    means=((amount*weight[:,:,None]).sum(0)/denominator).clamp_min(2e-4)
    return torch.cat((torch.logit(probabilities),torch.log(torch.expm1(means/20.))),-1)


def prediction(output):
    return output[:,:,:4].sigmoid(),nn.functional.softplus(output[:,:,4:])*20.


def metrics(output,d,w):
    event,amount=targets(d);probability,predicted=prediction(output)
    mask=d['known'].float()*w[:,None];den=mask.sum()
    bce=nn.functional.binary_cross_entropy_with_logits(output[:,:,:4],event,reduction='none')
    mse=(predicted-amount).square()
    def summarize(m):
        total=m.sum()
        return dict(received_bce=float((bce[:,:,0]*m).sum()/total),dealt_bce=float((bce[:,:,1]*m).sum()/total),
            received_rmse=float(((mse[:,:,0]*m).sum()/total).sqrt()),dealt_rmse=float(((mse[:,:,1]*m).sum()/total).sqrt()),
            received_observed=float((amount[:,:,0]*m).sum()/total),received_predicted=float((predicted[:,:,0]*m).sum()/total))
    result=dict(aggregate=summarize(mask),horizons={})
    for i,h in enumerate((1,3,5,10)):
        only=torch.zeros_like(mask);only[:,i]=mask[:,i];result['horizons'][str(h)]=summarize(only)
    return result


def challenge_metrics(output,d,roots):
    _,pred=prediction(output);actual=targets(d)[1]
    index={(r['context_id'],r['stream_seed'],r['duration'],r['label']):i for i,r in enumerate(roots)}
    worlds=sorted({r['context_id'] for r in roots});seeds=sorted({r['stream_seed'] for r in roots})
    result={}
    for kind in ('alternative_control','long_short'):
        by_horizon={}
        for hindex,h in enumerate((1,3,5,10)):
            means_pred=[];means_actual=[];all_pred=[];all_actual=[]
            for cid in worlds:
                for label in ('left','right','forward','backward'):
                    for duration in ((1,5) if kind=='alternative_control' else (5,)):
                        pairs=[]
                        for seed in seeds:
                            left=index[cid,seed,duration,label]
                            right=index[cid,seed,duration,'control'] if kind=='alternative_control' else index[cid,seed,1,label]
                            if bool(d['known'][left,hindex] and d['known'][right,hindex]):pairs.append((left,right))
                        if len(pairs)!=4:continue
                        li=torch.tensor([x for x,y in pairs],device='cuda');ri=torch.tensor([y for x,y in pairs],device='cuda')
                        pd=pred[li,hindex]-pred[ri,hindex];yd=actual[li,hindex]-actual[ri,hindex]
                        all_pred.append(pd);all_actual.append(yd);means_pred.append(pd.mean(0));means_actual.append(yd.mean(0))
            rows={}
            for scope,pp,yy in [('per_stream',torch.cat(all_pred),torch.cat(all_actual)),('four_stream_mean',torch.stack(means_pred),torch.stack(means_actual))]:
                rmse=(pp-yy).square().mean(0).sqrt();zero=yy.square().mean(0).sqrt()
                rows[scope]={name:dict(rmse=float(rmse[i]),zero_effect_rmse=float(zero[i]),ratio=float(rmse[i]/zero[i]) if float(zero[i])>0 else None) for i,name in enumerate(('received','dealt'))}
                rows[scope]['comparisons']=len(pp)
            by_horizon[str(h)]=rows
        result[kind]=dict(worlds=len(worlds),horizons=by_horizon)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path);h=pins[key] if key in pins else sha(path)
        assert expected is None or expected==h,str(path);pins[key]=h;return path
    p=read(pin(a.protocol));assert p['version']=='combat_local_effect_predictor_experiment_v1'
    closed=read(pin(a.dataset/'complete.json'));ds=read(pin(a.dataset/'report.json',closed['report_sha256']))
    assert ds['version']=='combat_local_effect_prediction_dataset_v1' and ds['protocol_sha256']==sha(a.protocol)
    assert ds['source_weights_sha256']==p['source_weights_sha256']
    for path,h in ds['inputs'].items():pin(path,h)
    train_worlds=set(ds['worlds']['train']);dev_worlds=set(ds['worlds']['development']);challenge_worlds=set(ds['worlds']['challenge'])
    assert not(train_worlds & dev_worlds or train_worlds & challenge_worlds or dev_worlds & challenge_worlds)
    assert len(train_worlds)==20 and len(dev_worlds)==8 and len(challenge_worlds)==4
    cr=Path(p['challenge']);cc=read(pin(cr/'complete.json'));cm=read(pin(cr/'report.json',cc['report_sha256']))
    roots=read(pin(cr/'records.json',cm['artifacts']['records']))
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data={};encoded={};weights={}
    for split in ('train','development','challenge'):
        chunks=[]
        for item in ds['chunks']:
            if item['split']!=split:continue
            with np.load(pin(item['path'],item['sha256']),allow_pickle=False) as arrays:
                chunks.append({k:arrays[k].copy() for k in arrays.files})
        raw={k:np.concatenate([chunk[k] for chunk in chunks]) for k in chunks[0]}
        data[split]={k:torch.as_tensor(v,device='cuda') for k,v in raw.items()}
        assert len(raw['features'])==ds['rows'][split]
        encoded[split]=encode(data[split]);weights[split]=world_weights(data[split])
    assert len(roots)==len(encoded['challenge'])==160
    for i,r in enumerate(roots):assert int(data['challenge']['world'][i])==r['world_seed'] and int(data['challenge']['frame'][i])==r['frame']
    baseline=constants(data['train'],weights['train']);baseline_metrics={}
    for split in ('train','development'):
        baseline_metrics[split]=metrics(baseline[None].expand(len(encoded[split]),-1,-1),data[split],weights[split])
    a.out.mkdir(parents=True);cfg=p['training'];event,amount=targets(data['train']);known=data['train']['known'].float()
    results={};initial_state=None
    for name in p['models']:
        torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
        model=Forecaster()
        with torch.no_grad():
            model.network[-1].weight.zero_();model.network[-1].bias.copy_(baseline.reshape(-1))
        if initial_state is None:initial_state=copy.deepcopy(model.state_dict())
        else:assert all(torch.equal(v,initial_state[k]) for k,v in model.state_dict().items())
        inputs=encoded if name=='state_action' else {k:torch.cat((x[:,:1147],torch.zeros_like(x[:,1147:])),1) for k,x in encoded.items()}
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        assert all(t.is_cuda for t in model.parameters())
        torch.manual_seed(cfg['seed']+1);torch.cuda.manual_seed_all(cfg['seed']+1)
        for step in range(cfg['steps']):
            indices=torch.multinomial(weights['train'],cfg['batch_size'],replacement=True)
            output=model(inputs['train'][indices]);mask=known[indices]
            bce=nn.functional.binary_cross_entropy_with_logits(output[:,:,:4],event[indices],reduction='none')
            squared=(nn.functional.softplus(output[:,:,4:])-amount[indices]/20.).square()
            loss=(torch.cat((bce,squared),-1).mean(-1)*mask).sum()/mask.sum()
            assert bool(torch.isfinite(loss));optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);optimizer.step()
            if (step+1)%100==0:save(a.out/'progress.json',dict(stage='cuda-training',model=name,steps=step+1,total_steps=cfg['steps'],loss=float(loss.detach())))
        model.eval()
        with torch.no_grad():
            outputs={split:model(x) for split,x in inputs.items()}
            scores={split:metrics(outputs[split],data[split],weights[split]) for split in ('train','development')}
            counterfactual=challenge_metrics(outputs['challenge'],data['challenge'],roots)
        folder=a.out/name;folder.mkdir()
        checkpoint=folder/'checkpoint.pt'
        torch.save(dict(version='combat_local_effect_forecaster_v1',model=name,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
                        protocol_sha256=sha(a.protocol),dataset_sha256=closed['report_sha256'],steps_completed=cfg['steps'],cuda_rng=torch.cuda.get_rng_state()),checkpoint)
        restored=Forecaster();state=torch.load(checkpoint,map_location='cuda',weights_only=False);restored.load_state_dict(state['state_dict']);restored.eval()
        assert all(torch.equal(v,state['state_dict'][k]) for k,v in restored.state_dict().items())
        with torch.no_grad():assert torch.equal(restored(inputs['development']),outputs['development'])
        result=dict(model=name,device='cuda',steps_completed=cfg['steps'],parameters=sum(t.numel() for t in model.parameters()),
            metrics=scores,counterfactual=counterfactual,checkpoint_sha256=sha(checkpoint),gpu_reload_exact=True,
            prediction_scope='Cumulative native effects until H or verified terminal; nonterminal handoff/unknown windows masked, not zero-filled.')
        save(folder/'report.json',result);results[name]=result
    full=results['state_action'];q=p['runtime_admission'];dev=full['metrics']['development']['aggregate'];const=baseline_metrics['development']['aggregate']
    ratios={k:dev[k]/const[k] if const[k]>0 else None for k in ('received_bce','received_rmse','dealt_rmse')}
    checks={k:ratios[k] is not None and ratios[k]<=q['development_'+k+'_ratio_to_constant_max'] for k in ratios}
    for kind in ('alternative_control','long_short'):
        group=full['counterfactual'][kind];c=group['horizons']['10']['four_stream_mean']
        r=[c[name]['ratio'] for name in ('received','dealt')]
        checks[kind]=group['worlds']==q['required_challenge_worlds'] and all(v is not None for v in r) and min(r)<=q['challenge_mean_h10_delta_rmse_ratio_at_least_one_component_max'] and max(r)<=q['challenge_mean_h10_other_component_ratio_max']
    for path,h in pins.items():assert sha(path)==h,path
    report=dict(state='complete',version='combat_local_effect_predictor_experiment_report_v1',device='cuda',results=results,
        baseline=baseline_metrics,development_ratios=ratios,runtime_admission_checks=checks,runtime_admitted=all(checks.values()),
        gate_binding='Development aggregate across all four known horizons. Both alternative/control and long/short averaged-H10 counterfactual checks must pass; no choosing the easier one.',
        world_counts={k:len(v) for k,v in ds['worlds'].items()},rows=ds['rows'],inputs=pins,training_performed=True,actor_training=False,
        ordinary_training_eligible=False,promotion=False,new_games=False,source_policy_unchanged=True,
        scope='Fixed-budget auxiliary forecast comparison. Development exploratory; four previously inspected challenge worlds excluded from fitting. Predictive admission is not improved combat or promotion.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    save(a.out/'progress.json',dict(stage='complete',runtime_admitted=report['runtime_admitted'],training_performed=True,actor_training=False))
    print(dict(development_ratios=ratios,checks=checks,runtime_admitted=report['runtime_admitted']))


if __name__=='__main__':main()

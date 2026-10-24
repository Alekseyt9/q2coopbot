"""CUDA two-part observed-enemy damage model, without changing the combat actor."""
import argparse
from pathlib import Path
import numpy as np
from evaluate_registered_combat_models import read,save,sha
from train_combat_enemy_threat_cuda import torch,nn,training_devices,SharedThreat,inputs,world_weights,constants,metrics,challenge


def aggregate(output,mask,health,unknown):
    probability=output[:,:,0].sigmoid()*mask
    conditional=nn.functional.softplus(output[:,:,1])*20.
    expected=probability*conditional
    total_probability=(1.-(1.-probability).prod(-1)*(1.-unknown['probability'])).clamp(1e-6,1-1e-6)
    total_amount=torch.minimum(expected.sum(-1)+unknown['amount'],health.clamp_min(0))
    return probability,expected,total_probability,total_amount


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path);h=pins[key] if key in pins else sha(path)
        assert expected is None or expected==h,str(path);pins[key]=h;return path
    p=read(pin(a.protocol));assert p['version']=='combat_enemy_hurdle_predictor_experiment_v1'
    parent=read(pin(p['parent_protocol']));root=Path(p['parent_experiment']);pc=read(pin(root/'complete.json'))
    predecessor=read(pin(root/'report.json',pc['report_sha256']));assert predecessor['device']=='cuda'
    path=Path(p['dataset']);closed=read(pin(path/'complete.json'));ds=read(pin(path/'report.json',closed['report_sha256']))
    assert ds['protocol_sha256']==sha(p['parent_protocol']) and ds['all_h1_totals_exact']
    for path,h in ds['inputs'].items():pin(path,h)
    base=Path(parent['base_dataset']);bc=read(pin(base/'complete.json'));bd=read(pin(base/'report.json',bc['report_sha256']))
    c=next(c for c in bd['chunks'] if c['split']=='challenge');roots=read(pin(c['case']['root_records'],c['case']['root_records_sha256']))
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data={};xx={};masks={};kinds={};weights={}
    for split in ('train','development','challenge'):
        chunks=[]
        for c in ds['chunks']:
            if c['split']==split:
                with np.load(pin(c['path'],c['sha256']),allow_pickle=False) as f:chunks.append({k:f[k].copy() for k in f.files})
        data[split]={k:torch.as_tensor(np.concatenate([c[k] for c in chunks]),device='cuda') for k in chunks[0]}
        xx[split],masks[split],kinds[split]=inputs(data[split]);weights[split]=world_weights(data[split])
    _,_,unknown,global_p,global_mean=constants(data['train'],weights['train'],masks['train'],kinds['train'])
    cfg=p['training'];torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);model=SharedThreat()
    assert sum(v.numel() for v in model.parameters())==6514
    with torch.no_grad():
        model.network[-1].weight.zero_()
        conditional_mean=(global_mean/global_p.clamp_min(1e-5)/20.).clamp_min(1e-5)
        model.network[-1].bias.copy_(torch.stack((torch.logit(global_p.clamp(1e-5,1-1e-5)),torch.log(torch.expm1(conditional_mean)))))
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    a.out.mkdir(parents=True);torch.manual_seed(cfg['seed']+1);torch.cuda.manual_seed_all(cfg['seed']+1)
    for step in range(cfg['steps']):
        ix=torch.multinomial(weights['train'],cfg['batch_size'],replacement=True);mask=masks['train'][ix];d=data['train']
        output=model(xx['train'][ix]);y=d['incoming_by_enemy'][ix];total=d['effects'][ix,0,1]
        node_p,node_mean,root_p,root_mean=aggregate(output,mask,d['features'][ix,0]*100,unknown)
        node_weight=mask/mask.sum(-1,keepdim=True).clamp_min(1)
        event_loss=(nn.functional.binary_cross_entropy_with_logits(output[:,:,0],(y>0).float(),reduction='none')*node_weight).sum(-1).mean()
        positive=node_weight*(y>0)
        amount_loss=(((nn.functional.softplus(output[:,:,1])-y/20.).square())*positive).sum()/positive.sum().clamp_min(1)
        root_event=nn.functional.binary_cross_entropy(root_p,(total>0).float())
        root_amount=((root_mean-total)/20.).square().mean();loss=(event_loss+amount_loss+root_event+root_amount)/4
        assert bool(torch.isfinite(loss));optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);optimizer.step()
        if (step+1)%100==0:save(a.out/'progress.json',dict(stage='cuda-fitting',steps=step+1,total_steps=cfg['steps'],loss=float(loss.detach())))
    model.eval()
    with torch.no_grad():
        outputs={split:model(v) for split,v in xx.items()}
        scores={split:metrics(outputs[split],data[split],weights[split],masks[split],unknown,aggregate_fn=aggregate) for split in ('train','development')}
        counterfactual=challenge(outputs['challenge'],data['challenge'],masks['challenge'],unknown,roots,aggregate_fn=aggregate)
    checkpoint=a.out/'checkpoint.pt'
    torch.save(dict(version='combat_enemy_hurdle_threat_v1',state_dict=model.state_dict(),optimizer=optimizer.state_dict(),unknown=unknown,
        amount_semantics='conditional_positive_damage',steps_completed=cfg['steps'],protocol_sha256=sha(a.protocol),dataset_sha256=closed['report_sha256'],cuda_rng=torch.cuda.get_rng_state()),checkpoint)
    restored=SharedThreat();restored.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=False)['state_dict']);restored.eval()
    with torch.no_grad():assert torch.equal(restored(xx['development']),outputs['development'])
    q=parent['predictive_admission'];ratios={};checks={};full=scores['development']
    baselines=dict(type_constant=predecessor['type_baseline']['development'],frozen_flat_h1=predecessor['frozen_flat_development_h1'])
    for field,short in [('received_bce','bce'),('received_rmse','rmse')]:
        for label,baseline in baselines.items():
            key=short+'_to_'+label;ratios[key]=full[field]/baseline[field]
            checks[key]=ratios[key]<=q['development_'+short+'_ratio_to_'+label+'_max']
    for path,h in pins.items():assert sha(path)==h,path
    report=dict(state='complete',version='combat_enemy_hurdle_predictor_report_v1',device='cuda',parameters=6514,steps_completed=cfg['steps'],
        metrics=scores,challenge=counterfactual,unknown_train_constants={k:float(v) for k,v in unknown.items()},predictive_ratios=ratios,predictive_checks=checks,
        predictive_admitted=all(checks.values()),checkpoint_sha256=sha(checkpoint),gpu_reload_exact=True,worlds=ds['worlds'],rows=ds['rows'],inputs=pins,
        training_performed=True,actor_training=False,new_games=False,ordinary_training_eligible=False,runtime_action_admitted=False,promotion=False,
        scope='Exploratory fixed two-part damage forecast with unchanged admission thresholds. A predictive pass permits fresh forecast/under-fire data checks only, never combat policy promotion.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    save(a.out/'progress.json',dict(stage='complete',predictive_admitted=report['predictive_admitted'],runtime_action_admitted=False))
    print(dict(metrics=scores['development'],predictive_ratios=ratios,predictive_admitted=report['predictive_admitted']))


if __name__=='__main__':main()

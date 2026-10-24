"""Frozen threat forecasts on new alternate-site worlds; no fitting or promotion."""
import argparse
from pathlib import Path
import numpy as np
from evaluate_registered_combat_models import read,save,sha
from train_combat_enemy_threat_cuda import torch,training_devices,SharedThreat,inputs,world_weights,constants,aggregate
from train_combat_enemy_hurdle_cuda import aggregate as hurdle_aggregate
from train_combat_local_effect_predictor_cuda import Forecaster as Flat,encode as flat_encode


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path);h=pins[key] if key in pins else sha(path)
        assert expected is None or expected==h,str(path);pins[key]=h;return path
    protocol=read(pin(a.protocol));assert protocol['version']=='combat_enemy_threat_fresh_corpus_v1'
    cs=read(pin(a.dataset/'complete.json'));ds=read(pin(a.dataset/'report.json',cs['report_sha256']))
    assert ds['version']=='combat_enemy_threat_fresh_dataset_v1' and ds['protocol_sha256']==sha(a.protocol) and ds['cases']==32
    parent_path=Path('workspace/artifacts/combat-enemy-threat-predictors-v1-20261023')
    pc=read(pin(parent_path/'complete.json'));parent=read(pin(parent_path/'report.json',pc['report_sha256']))
    hurdle_path=Path('workspace/artifacts/combat-enemy-hurdle-predictor-v1-20261023')
    hc=read(pin(hurdle_path/'complete.json'));hurdle=read(pin(hurdle_path/'report.json',hc['report_sha256']))
    flat_path=Path('workspace/artifacts/combat-local-effect-predictors-v1-20261023')
    fc=read(pin(flat_path/'complete.json'));flat_report=read(pin(flat_path/'report.json',fc['report_sha256']))
    train_path=Path('workspace/artifacts/combat-enemy-threat-dataset-v1-20261023')
    tc=read(pin(train_path/'complete.json'));training=read(pin(train_path/'report.json',tc['report_sha256']))
    old_worlds={w for values in training['worlds'].values() for w in values}
    assert old_worlds.isdisjoint(ds['worlds']) and not parent['predictive_admitted'] and not hurdle['predictive_admitted']
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def load(chunks):
        arrays=[]
        for c in chunks:
            with np.load(pin(c['path'],c['sha256']),allow_pickle=False) as f:arrays.append({k:f[k].copy() for k in f.files})
        return {k:torch.as_tensor(np.concatenate([v[k] for v in arrays]),device='cuda') for k in arrays[0]}
    train=load([c for c in training['chunks'] if c['split']=='train']);d=load(ds['chunks'])
    assert bool(d['known'][:,0].all())
    tx,tm,tk=inputs(train);x,mask,kind=inputs(d);weights=world_weights(d)
    probs,means,unknown,_,_=constants(train,world_weights(train),tm,tk)
    actual=d['effects'][:,0,1];event=(actual>0).float();predictions={};node_predictions={}
    with torch.no_grad():
        type_output=torch.stack((torch.logit(probs[kind]),torch.log(torch.expm1(means[kind]/20.))),-1)
        npred,nmean,prob,amount=aggregate(type_output,mask,d['features'][:,0]*100,unknown)
        predictions['type_constant']=(prob,amount);node_predictions['type_constant']=nmean
        for name in ('shared_phase_action','shared_no_phase'):
            model=SharedThreat();checkpoint=pin(parent_path/name/'checkpoint.pt',parent['results'][name]['checkpoint_sha256'])
            model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=False)['state_dict']);model.eval()
            xx=x if name=='shared_phase_action' else inputs(d,no_phase=True)[0]
            npred,nmean,prob,amount=aggregate(model(xx),mask,d['features'][:,0]*100,unknown)
            predictions[name]=(prob,amount);node_predictions[name]=nmean
        model=SharedThreat();checkpoint=pin(hurdle_path/'checkpoint.pt',hurdle['checkpoint_sha256'])
        state=torch.load(checkpoint,map_location='cuda',weights_only=False);assert state['version']=='combat_enemy_hurdle_threat_v1'
        model.load_state_dict(state['state_dict']);model.eval()
        npred,nmean,prob,amount=hurdle_aggregate(model(x),mask,d['features'][:,0]*100,unknown)
        predictions['hurdle_phase_action']=(prob,amount);node_predictions['hurdle_phase_action']=nmean
        flat=Flat();checkpoint=pin(flat_path/'state_action/checkpoint.pt',flat_report['results']['state_action']['checkpoint_sha256'])
        flat.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=False)['state_dict']);flat.eval();out=flat(flat_encode(d))
        predictions['frozen_flat']=(out[:,0,0].sigmoid(),torch.nn.functional.softplus(out[:,0,4])*20.)
    worlds=torch.unique(d['world']);assert len(worlds)==32
    torch.manual_seed(2026102312);torch.cuda.manual_seed_all(2026102312)
    bootstrap=torch.randint(0,len(worlds),(4096,len(worlds)),device='cuda')
    losses={};scores={};per_world={}
    map_by_world={c['case']['seed']:c['case']['map'] for c in ds['chunks']}
    map_masks={mp:torch.isin(d['world'],torch.tensor([world for world,name in map_by_world.items() if name==mp],device='cuda')) for mp in sorted(set(map_by_world.values()))}
    def score(prob,amount,w):
        w=w/w.sum();bce=torch.nn.functional.binary_cross_entropy(prob.clamp(1e-6,1-1e-6),event,reduction='none');squared=(amount-actual).square()
        return dict(received_bce=float((bce*w).sum()),received_rmse=float((squared*w).sum().sqrt()),
            observed_mean=float((actual*w).sum()),predicted_mean=float((amount*w).sum()),event_brier=float(((prob-event).square()*w).sum())),bce,squared
    for name,(prob,amount) in predictions.items():
        metrics,bce,squared=score(prob,amount,weights);losses[name]=(bce,squared)
        metrics['per_map']={mp:score(prob,amount,weights*map_mask)[0] for mp,map_mask in map_masks.items()}
        metrics['unknown_damage_rows_squared_error_share']=float((squared*weights*(d['unknown_incoming']>0)).sum()/(squared*weights).sum())
        if name in node_predictions:
            visible_actual=d['incoming_by_enemy'].sum(-1);visible_prediction=node_predictions[name].sum(-1)
            metrics['visible_received_rmse']=float(((visible_prediction-visible_actual).square()*weights).sum().sqrt())
        scores[name]=metrics
        per_world[name]=dict(bce=torch.stack([(bce[d['world']==w]).mean() for w in worlds]),mse=torch.stack([(squared[d['world']==w]).mean() for w in worlds]))
    intervals={}
    for name in predictions:
        if name=='type_constant':continue
        intervals[name]={}
        for metric in ('bce','mse'):
            numerator=per_world[name][metric][bootstrap].mean(-1);denominator=per_world['type_constant'][metric][bootstrap].mean(-1)
            ratio=numerator/denominator
            if metric=='mse':ratio=ratio.sqrt()
            q=torch.quantile(ratio,torch.tensor([.025,.975],device='cuda'))
            intervals[name]['rmse_ratio' if metric=='mse' else 'bce_ratio']=dict(point=scores[name]['received_rmse' if metric=='mse' else 'received_bce']/scores['type_constant']['received_rmse' if metric=='mse' else 'received_bce'],lower=float(q[0]),upper=float(q[1]))
    contexts=read(pin(a.dataset/'contexts.json',ds['contexts_sha256']));unknown_rows=read(pin(a.dataset/'unknown-attacks.json',ds['unknown_attacks_sha256']))
    counts=dict(unknown_events=len(unknown_rows),unknown_hp=sum(r['health_damage'] for r in unknown_rows),
        with_observed_projectile_events=sum(bool(r['observed_projectile_classes']) for r in unknown_rows),
        with_observed_projectile_hp=sum(r['health_damage'] for r in unknown_rows if r['observed_projectile_classes']),
        in_memory_events=sum(r['attacker_in_memory'] for r in unknown_rows))
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True);report=dict(state='complete',version='combat_enemy_threat_fresh_evaluation_v1',device='cuda',worlds=32,rows=len(actual),
        metrics=scores,ratios_to_type_constant_world_bootstrap=intervals,bootstrap_scope='4096 paired resamples of whole worlds, conditional on these eight registry families/four maps; not an interval over all campaign geometries.',
        unknown_coverage=counts,contexts=len(contexts['contexts']),context_maps=sorted({r['instance']['map'] for r in contexts['contexts']}),
        inputs=pins,training_performed=False,actor_training=False,new_games=False,old_predictive_gate_failures_retained=True,
        runtime_action_admitted=False,promotion=False,
        scope='Frozen forecast transfer and coverage audit at alternate sites. All new worlds excluded from fitting. No runtime controller change, checkpoint selection, threshold relaxation, or learned combat superiority claim.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(worlds=32,rows=len(actual),metrics={name:{k:v for k,v in value.items() if k in ('received_bce','received_rmse','predicted_mean','observed_mean')} for name,value in scores.items()},unknown=counts,contexts=len(contexts['contexts'])))


if __name__=='__main__':main()

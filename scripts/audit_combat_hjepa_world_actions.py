"""Frozen one-step world forecasts: proposed command, applied command, persistence."""
import argparse
import collections
import json
import math
import pathlib
import torch
from train_combat_hjepa import sha,save,serialized_forward


def primitive(a):
    vertical=['release','jump','crouch']
    if a['vertical'] not in vertical:raise ValueError('unknown primitive pose')
    return [a['forward'],a['side'],a['yaw_delta_degrees']/180,a['pitch_delta_degrees']/180,
            float(a['attack']),*[float(a['vertical']==v) for v in vertical]]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=pathlib.Path,required=True)
    ap.add_argument('--model',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    a=ap.parse_args()
    meta_path=pathlib.Path(str(a.data)+'.meta.json');meta=json.loads(meta_path.read_text())
    if meta['version']!='hjepa_data_v1' or meta['feature_version']!='combat_features_v6':
        raise ValueError('world feature contract mismatch')
    if meta.get('data_sha256') and sha(a.data)!=meta['data_sha256']:raise ValueError('dataset changed')
    source_steps={}
    for name,digest in meta['sources_sha256'].items():
        path=pathlib.Path(name)
        if sha(path)!=digest:raise ValueError('historical native source changed: '+name)
        if path.name!='steps.jsonl':continue
        indexed={}
        for line in path.read_text().splitlines():
            step=json.loads(line);key=step['observation']['identity']['frame']
            if key in indexed:raise ValueError('ambiguous source frame')
            indexed[key]=step
        source_steps[digest]=indexed
    rows=[json.loads(line) for line in a.data.read_text().splitlines()]
    if len(rows)!=meta['rows']:raise ValueError('dataset row count changed')
    raw=[];applied=[];groups=collections.defaultdict(list);reasons=collections.Counter()
    for index,row in enumerate(rows):
        step=source_steps[row['source_sha256']][row['frame']]
        av=primitive(step['applied_action']);rv=primitive(step['action'])
        if any(abs(x-y)>1e-12 for x,y in zip(av,row['action'])):
            raise ValueError('training action does not match native AppliedAction')
        if not step.get('native_step') or not step['server_execution']['matched']:
            raise ValueError('missing matched native action')
        raw.append(rv);applied.append(av)
        guarded=bool(step.get('interventions'))
        changed=any(abs(x-y)>.01 for x,y in zip(rv[:4],av[:4])) or rv[4:]!=av[4:]
        groups['all'].append(index)
        groups['guarded' if guarded else 'unclipped'].append(index)
        groups['primitive_changed' if changed else 'primitive_same_or_quantized'].append(index)
        groups['owner_'+step['owner']].append(index)
        if step['owner']=='provider':groups['provider_guarded' if guarded else 'provider_unclipped'].append(index)
        for intervention in step.get('interventions',[]):reasons[intervention['reason']]+=1
    model=json.loads(a.model.read_text());level=model['levels'][0]
    if level['stride']!=1 or model['feature_version']!='combat_features_v6':raise ValueError('one-step v6 model required')
    torch.set_num_threads(2)
    x=torch.tensor([r['features'] for r in rows],dtype=torch.float64)
    nx=torch.tensor([r['next_features'] for r in rows],dtype=torch.float64)
    av=torch.tensor(applied,dtype=torch.float64);rv=torch.tensor(raw,dtype=torch.float64)
    with torch.no_grad():
        z=serialized_forward(x,level['encoder']);target=serialized_forward(nx,level['encoder'])
        pa=serialized_forward(torch.cat((z,av),-1),level['predictor'])
        pr=serialized_forward(torch.cat((z,rv),-1),level['predictor'])
        errors=dict(applied=(pa-target).square().mean(-1),proposed=(pr-target).square().mean(-1),
                    persistence=(z-target).square().mean(-1),forecast_input_sensitivity=(pa-pr).square().mean(-1))
    if not all(torch.isfinite(v).all() for v in errors.values()):raise ValueError('nonfinite world diagnostic')
    summaries={}
    for group,indexes in groups.items():
        summaries[group]=dict(rows=len(indexes),**{k:float(v[indexes].mean()) for k,v in errors.items()},
            proposed_better_than_applied=sum(float(errors['proposed'][i])<float(errors['applied'][i]) for i in indexes),
            applied_better_than_persistence=sum(float(errors['applied'][i])<float(errors['persistence'][i]) for i in indexes))
    save(a.out,dict(version='hjepa_one_step_action_contract_audit_v1',promotion=False,
        model_sha256=sha(a.model),data_sha256=sha(a.data),meta_sha256=sha(meta_path),
        source_training_action_verified='AppliedAction',groups=summaries,guard_reasons=dict(reasons),
        scope='Frozen input substitution on native-verified transitions, only stride1 latent prediction. '
              'Raw and applied inputs share the same observed next target; not a physical intervention '
              'or proof that changing input contract improves combat. Persistence is a predictive baseline, '
              'not an alternative controller. No fitting or future policy inputs.'))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()

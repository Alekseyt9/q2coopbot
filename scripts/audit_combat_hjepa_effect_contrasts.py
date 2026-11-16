"""Score a frozen effect predictor on new controlled constant-action contrasts."""
import argparse
import collections
import json
import pathlib
import torch
from train_combat_hjepa import sha, save, serialized_forward


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--contrasts',type=pathlib.Path,required=True)
    ap.add_argument('--model',type=pathlib.Path,required=True)
    ap.add_argument('--fit-report',type=pathlib.Path,required=True)
    ap.add_argument('--training-meta',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    a=ap.parse_args();report=json.loads((a.contrasts/'report.json').read_text())
    if report['rows_sha256']!=sha(a.contrasts/'rows.jsonl'):raise ValueError('controlled rows changed')
    for name,digest in report['sources_sha256'].items():
        if sha(pathlib.Path(name))!=digest:raise ValueError('controlled evidence changed')
    model=json.loads(a.model.read_text())
    if model['version']!='hjepa_native_effect_predictor_v1':raise ValueError('wrong effect predictor')
    fit=json.loads(a.fit_report.read_text());training=json.loads(a.training_meta.read_text())
    if fit['model_sha256']!=sha(a.model) or fit['data_meta_sha256']!=sha(a.training_meta):raise ValueError('predictor training provenance changed')
    rows=[json.loads(l) for l in (a.contrasts/'rows.jsonl').open()]
    if {r['seed'] for r in rows}&{e['seed'] for e in training['episode_ledgers']}:raise ValueError('controlled worlds overlap fitted evidence')
    torch.set_num_threads(2);weights=torch.tensor(model['reward_weights'],dtype=torch.float64)
    results={}
    with torch.no_grad():
        for i,h in enumerate([1,4,16]):
            samples=[r for r in rows if r['horizon']==h]
            features=torch.tensor([r['features'] for r in samples],dtype=torch.float64)
            actions=torch.tensor([r['planned_actions'] for r in samples],dtype=torch.float64).flatten(1)
            target=torch.tensor([r['effects'] for r in samples],dtype=torch.float64)
            z=features[:,:845]
            for encoder in model['encoders'][:i+1]:z=serialized_forward(z,encoder)
            level=model['levels'][i]
            mean=torch.tensor(level['latent_mean'],dtype=torch.float64);scale=torch.tensor(level['latent_scale'],dtype=torch.float64)
            x=torch.cat(((z-mean)/scale,features[:,845:],actions),-1)
            prediction=serialized_forward(x,level['heads']['full'])
            index={(r['seed'],r['variant']):j for j,r in enumerate(samples)}
            deltas=[];actual=[];details=[]
            for row in samples:
                variant=row['variant']
                if variant=='idle':continue
                reference='idle' if variant=='fire' else 'fire'
                if (row['seed'],reference) not in index:continue
                j=index[row['seed'],variant];k=index[row['seed'],reference]
                deltas.append(prediction[j]-prediction[k]);actual.append(target[j]-target[k])
                details.append(dict(seed=row['seed'],variant=variant,reference=reference,
                                    predicted_reward_difference=float((prediction[j]-prediction[k])@weights),
                                    actual_reward_difference=float((target[j]-target[k])@weights)))
            delta=torch.stack(deltas);truth=torch.stack(actual);pr=delta@weights;tr=truth@weights;nonzero=tr.abs()>1e-9
            results[h]=dict(paired_comparisons=len(details),worlds=len({r['seed'] for r in details}),
                           contrast_effect_mse=float((delta-truth).square().mean()),zero_contrast_effect_mse=float(truth.square().mean()),
                           contrast_reward_mse=float((pr-tr).square().mean()),zero_contrast_reward_mse=float(tr.square().mean()),
                           nonzero_actual_reward_contrasts=int(nonzero.sum()),
                           nonzero_reward_sign_accuracy=float(((pr[nonzero]>0)==(tr[nonzero]>0)).double().mean()) if nonzero.any() else None,
                           details=details)
    save(a.out,dict(version='hjepa_controlled_effect_prediction_audit_v1',promotion=False,model_sha256=sha(a.model),
         fit_report_sha256=sha(a.fit_report),training_meta_sha256=sha(a.training_meta),source_worlds_disjoint=True,
         contrast_report_sha256=sha(a.contrasts/'report.json'),horizons=results,
         scope='Frozen inference on new TRAIN470 interventions, no fitting. Differences pair identical causal initial state, so action-ablated and constant mean predictors imply zero contrast. Commands planned before outcomes; environment guards remain active. Comparisons share16worlds and are descriptive, not independent samples or learned-combat validation.'))
    print(json.dumps({h:{k:v for k,v in r.items() if k!='details'} for h,r in results.items()},indent=2))


if __name__=='__main__':main()

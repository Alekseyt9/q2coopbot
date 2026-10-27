"""Evaluate pre-frozen predictions on all newly collected continuation cases."""
import argparse
import json
from pathlib import Path

from ppo_combat import torch,training_devices
from process_combat_architecture_pool import read,save,sha


def run(frozen,cases,out):
    frozen,cases,out=frozen.resolve(),cases.resolve(),out.resolve();assert not out.exists()
    prepared=read(frozen/'complete.json');closed=read(cases/'complete.json');protocol=read(cases/'protocol.json')
    assert prepared['state']==closed['state']=='complete' and closed['native_cases']==closed['cuda_cases']==72
    assert sha(frozen/'predictions.json')==prepared['predictions_sha256']==protocol['frozen_predictions_sha256']
    assert sha(frozen/'protocol.json')==prepared['protocol_sha256'] and sha(frozen/'frozen-readout.pt')==prepared['readout_sha256']
    assert sha(cases/'results.json')==closed['results_sha256']
    predictions=read(frozen/'predictions.json')['contexts'];results=read(cases/'results.json')['cases']
    assert len(predictions)==len(results)==8 and {p['record_id'] for p in predictions}=={r['context']['record_id'] for r in results}
    pins={str(p):sha(p) for p in (frozen/'predictions.json',frozen/'complete.json',frozen/'protocol.json',frozen/'frozen-readout.pt',
        cases/'results.json',cases/'complete.json',cases/'protocol.json',Path(__file__).resolve())}
    exports={(e['context_id'],e['future'],e['mode']):e for e in read(cases/'exports.json')['entries']}
    retained=[];excluded=[]
    for prediction in predictions:
        case=next(r for r in results if r['context']['record_id']==prediction['record_id'])
        assert all(prediction[k]==case['context'][k] for k in ('map','seed','frame'))
        assert case['historical_root_exact'] and all(c['root_and_prefix_exact'] and c['matching_clamp_life_terminal_exact'] for c in case['checks'])
        # Recheck all original/fork arms and their sealed CUDA likelihood proof.
        for outcome in case['outcomes']:
            entry=exports[(case['context']['id'],outcome['future'],outcome['mode'])]
            directory=Path(outcome['data']);assert str(directory)==entry['data']
            meta=read(directory/'report.json');proof=read(directory/'cuda-verification.json')
            assert proof['state']=='passed' and proof['device']=='cuda' and proof['model_sha256']==entry['model_sha256']
            for p,h in ((directory/'cuda-verification.json',meta['cuda_verification_sha256']),(directory/'rollout.jsonl',meta['rollout_sha256'])):
                assert sha(p)==h;pins[str(p)]=h
            pins[str(directory/'report.json')]=sha(directory/'report.json')
            with (directory/'rollout.jsonl').open(encoding='utf-8-sig') as stream:
                root=next(r for r in map(json.loads,stream) if r['frame']==prediction['frame'])
            if outcome['mode']=='baseline':
                assert abs(root['sample']['wide_aim']['logit']-prediction['source_logit'])<1e-4,'Frozen root context no longer matches native gate'
            else:
                choice=int(outcome['mode'][-1])
                assert root['sample']['wide_aim']['logit']==40*choice-20
                assert root['sample']['wide_aim']['choice']==choice
        assert {d['future'] for d in case['differences']}=={'original','forkA','forkB'}
        if not all(d['eligible'] for d in case['differences']):
            excluded.append(dict(record_id=prediction['record_id'],map=prediction['map'],reason='mixed_control',differences=case['differences']))
        else:retained.append((prediction,[d['delta'] for d in case['differences']]))
    training_devices();torch.set_num_threads(2);dtype=torch.float64;device='cuda'
    assert retained,'No autonomous confirmation context'
    y=torch.tensor([d for _,d in retained],device=device,dtype=dtype).mean(-1)
    pred=torch.tensor([p['prediction'] for p,_ in retained],device=device,dtype=dtype)
    constant=torch.tensor([p['constant_prediction'] for p,_ in retained],device=device,dtype=dtype)
    scale=torch.tensor([p['train_scale'] for p,_ in retained],device=device,dtype=dtype)
    base=torch.tensor([p['source_logit'] for p,_ in retained],device=device,dtype=dtype)
    gains={name:((base+.5*(value/scale).tanh()).sigmoid()-base.sigmoid())*y for name,value in (('frozen16',pred),('constant',constant))}
    scores={name:dict(mse=float((value-y).square().mean()),root_surrogate_gain=float(gains[name].mean())) for name,value in (('frozen16',pred),('constant',constant))}
    excess=gains['frozen16']-gains['constant'];maps=sorted({p['map'] for p,_ in retained})
    sums=[];counts=[]
    for name in maps:
        members=[i for i,(p,_) in enumerate(retained) if p['map']==name]
        sums.append(excess[members].sum());counts.append(len(members))
    sums=torch.stack(sums);counts=torch.tensor(counts,device=device,dtype=dtype)
    generator=torch.Generator(device=device);generator.manual_seed(20261027)
    draws=torch.randint(len(maps),(4000,len(maps)),device=device,generator=generator)
    boot=sums[draws].sum(-1)/counts[draws].sum(-1)
    interval=torch.quantile(boot,torch.tensor([.025,.975],device=device,dtype=dtype)).tolist()
    ratio=scores['frozen16']['mse']/scores['constant']['mse']
    passed=len(retained)>=6 and ratio<=.95 and scores['frozen16']['root_surrogate_gain']>0 and float(excess.mean())>0 and interval[0]>0
    for p,h in pins.items():assert sha(p)==h,p
    out.mkdir(parents=True)
    save(out/'report.json',dict(state='complete',device=device,all_contexts=8,autonomous_contexts=len(retained),excluded=excluded,
        scores=scores,mse_ratio=ratio,excess_root_surrogate_gain=float(excess.mean()),descriptive_map_bootstrap_interval=interval,
        confirmation_gate_passed=passed,inputs=pins,
        per_context=[dict(record_id=p['record_id'],map=p['map'],seed=p['seed'],empirical_mean=float(y[i]),prediction=float(pred[i]),constant_prediction=float(constant[i])) for i,(p,_) in enumerate(retained)],
        scope='Frozen secondary hypothesis on eight additional historical TRAIN worlds. Root counterfactual surrogate only; future controller remains source. No retuning,actor fitting or native candidate-policy quality claim.',
        actor_compatible=False,policy_training_performed=False,promotion=False))
    print(out/'report.json',scores,'ratio',ratio,'interval',interval,'gate',passed,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('frozen','cases','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();run(a.frozen,a.cases,a.out)

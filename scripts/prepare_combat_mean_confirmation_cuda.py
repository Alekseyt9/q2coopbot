"""Freeze a secondary discovery hypothesis before new continuation collection."""
import argparse
import hashlib
import json
from pathlib import Path

from combat_continuation_evidence import load_index
from combat_native_teacher_registry import load_index as teachers
from combat_continuation_selection import select_unseen_worlds
from combat_teacher_folds import map_folds,world_weights
from combat_teacher_logistic_cuda import normalizer
from probe_combat_mean_advantage_cuda import ridge_fit,self_test
from ppo_combat import torch,training_devices
from process_combat_architecture_pool import read,save,sha


def input_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def run(index,teacher_index,source_path,discovery,out):
    repo=Path(__file__).resolve().parents[1];assert not out.exists()
    records=load_index(repo,index);train=[r for r in records if r['summary']['all_streams_autonomous']]
    selected=select_unseen_worlds(teachers(repo,teacher_index),records,8)
    assert len(records)==20 and len(train)==14
    report=read(discovery);assert report['state']=='complete' and report['device']=='cuda' and not report['learnability_gate_passed']
    assert report['scores']['frozen16']['mse']<report['scores']['constant']['mse']
    source=read(source_path);source_sha=sha(source_path)
    assert all(r['source_model_sha256']==source_sha for r in records+selected)
    pins={str(p.resolve()):sha(p) for p in (index,teacher_index,source_path,discovery)}
    for idx in (index,teacher_index):
        for corpus in read(idx)['corpora']:
            path=repo/corpus['manifest'];assert sha(path)==corpus['manifest_sha256'];pins[str(path)]=sha(path)
            for entry in read(path)['entries']:assert sha(entry['path'])==entry['sha256'];pins[entry['path']]=entry['sha256']
    for name in ('prepare_combat_mean_confirmation_cuda.py','combat_continuation_selection.py','probe_combat_mean_advantage_cuda.py',
                 'combat_continuation_evidence.py','combat_teacher_folds.py','combat_teacher_logistic_cuda.py'):
        path=repo/'scripts'/name;pins[str(path)]=sha(path)
    out.mkdir(parents=True)
    save(out/'protocol.json',dict(version='combat_frozen_mean_confirmation_v1',source_model_sha256=source_sha,
        discovery_report_sha256=sha(discovery),hypothesis='frozen16 secondary selected after physical24 primary failed; requires new confirmation,not a primary discovery win',
        train_contexts=14,known_contexts=20,confirmation_worlds=8,
        selection='Next SHA-ranked distinct worlds excluding all20 known worlds; no outcome/preference/prediction filter',
        contexts=[dict(record_id=r['id'],map=r['map'],seed=r['seed'],frame=r['frame']) for r in selected],
        lambda_selection='Four TRAIN whole-map folds;grid(.1,1,10,100),TRAIN-only standardization; minimumMSE,tie favorslarger ridge',
        correction='.5*tanh(predicted_mean/TRAIN_RMS_floor1) added to root source logit; source future policy unchanged',
        confirmation_gate='Atleast6/8fullyautonomous contexts;MSE<=.95constant;root surrogategain>0 and>constant;descriptive fixed-prediction map-bootstrap excess2.5percentile>0',
        scope='New roots for this regression,but historical teacher TRAIN corpus; not globally unseen maps or native candidate-policy quality',
        actor_compatible=False,policy_training_performed=False,promotion=False,inputs=pins))
    training_devices();torch.set_num_threads(2);self_test();dtype=torch.float64;device='cuda'
    def hidden(rr):
        value=torch.tensor([r['input']['pre_gate_context187'] for r in rr],device=device,dtype=dtype)
        for layer in source['wide_aim']['layers'][:2]:
            value=torch.nn.functional.relu(torch.nn.functional.linear(value,
                torch.tensor(layer['weight'],device=device,dtype=dtype),torch.tensor(layer['bias'],device=device,dtype=dtype)))
        return value
    x,new=hidden(train),hidden(selected)
    y=torch.stack([torch.tensor([s['outcomes']['choice1']['utility']-s['outcomes']['choice0']['utility'] for s in r['streams']],device=device,dtype=dtype).mean() for r in train])
    weights=lambda ids:torch.tensor(world_weights([train[i] for i in ids]),device=device,dtype=dtype)
    def matrix(ti,vi):
        mean,std=normalizer(x[ti],weights(ti));aug=lambda v:torch.cat(((v-mean)/std,torch.ones((len(v),1),device=device,dtype=dtype)),-1)
        return aug(x[ti]),aug(x[vi]),mean,std
    folds=map_folds(train);scores=[]
    for ridge in (.1,1.,10.,100.):
        prediction=torch.full_like(y,float('nan'))
        for fold in range(4):
            ti=[i for i,f in enumerate(folds) if f!=fold];vi=[i for i,f in enumerate(folds) if f==fold]
            assert {train[i]['map'] for i in ti}.isdisjoint({train[i]['map'] for i in vi})
            tr,va,_,_=matrix(ti,vi);prediction[vi]=va@ridge_fit(tr,y[ti],weights(ti),ridge)
        scores.append(dict(ridge=ridge,mse=float(((prediction-y).square()*weights(list(range(len(train))))).mean())))
    penalty=min(scores,key=lambda s:(s['mse'],-s['ridge']))['ridge']
    ids=list(range(len(train)));tr,_,mean,std=matrix(ids,[]);theta=ridge_fit(tr,y,weights(ids),penalty)
    design=torch.cat(((new-mean)/std,torch.ones((len(new),1),device=device,dtype=dtype)),-1)
    prediction=design@theta;w=weights(ids);constant=(w*y).sum()/w.sum();scale=((w*y.square()).sum()/w.sum()).sqrt().clamp_min(1.)
    layer=source['wide_aim']['layers'][2]
    logit=torch.nn.functional.linear(new,torch.tensor(layer['weight'],device=device,dtype=dtype),torch.tensor(layer['bias'],device=device,dtype=dtype))[:,0]
    assert bool(torch.isfinite(prediction).all())
    path=out/'frozen-readout.pt';torch.save(dict(theta=theta,mean=mean,std=std,ridge=penalty,constant=constant,scale=scale,
        actor_compatible=False,source_model_sha256=source_sha,protocol_sha256=sha(out/'protocol.json')),path)
    save(out/'predictions.json',dict(version='combat_frozen_mean_predictions_v1',contexts=[dict(record_id=r['id'],map=r['map'],seed=r['seed'],frame=r['frame'],
        input_sha256=input_sha(r['input']),prediction=float(prediction[i]),constant_prediction=float(constant),train_scale=float(scale),source_logit=float(logit[i])) for i,r in enumerate(selected)],
        cv_scores=scores,ridge=penalty,actor_compatible=False))
    for p,h in pins.items():assert sha(p)==h,p
    save(out/'complete.json',dict(state='complete',protocol_sha256=sha(out/'protocol.json'),predictions_sha256=sha(out/'predictions.json'),
        readout_sha256=sha(path),train_contexts=14,confirmation_worlds=8,device='cuda',actor_compatible=False,policy_training_performed=False,promotion=False))
    print(out/'complete.json',scores,'chosen',penalty,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('index','teacher-index','source','discovery','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();run(a.index,a.teacher_index,a.source,a.discovery,a.out.resolve())

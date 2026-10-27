"""Nested map validation of empirical paired mean-advantage regression."""
import argparse
import shutil
from pathlib import Path

from combat_continuation_evidence import load_index
from combat_teacher_folds import map_folds,world_weights
from combat_teacher_logistic_cuda import normalizer
from combat_teacher_physical_features import features
from ppo_combat import torch,training_devices
from process_combat_architecture_pool import read,save,sha


def ridge_fit(design,target,weights,penalty):
    assert all(x.is_cuda and x.dtype==torch.float64 for x in (design,target,weights))
    assert penalty>=0 and bool((weights>0).all())
    w=weights/weights.sum()
    diagonal=torch.full((design.shape[1],),penalty,device='cuda',dtype=torch.float64);diagonal[-1]=0.
    gram=design.T@(w[:,None]*design)+torch.diag(diagonal)
    rhs=design.T@(w*target)
    theta=torch.linalg.solve(gram,rhs)
    residual=design.T@(w*(design@theta-target))+diagonal*theta
    assert bool(torch.isfinite(theta).all()) and float(residual.abs().max())<1e-8
    return theta


def self_test():
    x=torch.tensor([[-1.,1.],[0.,1.],[1.,1.]],device='cuda',dtype=torch.float64)
    y=torch.tensor([1.,3.,5.],device='cuda',dtype=torch.float64);w=torch.ones_like(y)
    for penalty,expected in ((0.,[2.,3.]),(1.,[.8,3.])):
        result=ridge_fit(x,y,w,penalty)
        assert torch.allclose(result,torch.tensor(expected,device='cuda',dtype=torch.float64),atol=1e-12,rtol=0)
    result=ridge_fit(torch.ones((2,1),device='cuda',dtype=torch.float64),
        torch.tensor([1.,3.],device='cuda',dtype=torch.float64),torch.tensor([1.,3.],device='cuda',dtype=torch.float64),100.)
    assert abs(float(result[0])-2.5)<1e-12


def run(index,source_path,out):
    repo=Path(__file__).resolve().parents[1];assert not out.exists()
    records=load_index(repo,index)
    selected=[r for r in records if r['summary']['all_streams_autonomous']]
    assert len(records)==20 and len(selected)==14 and len({r['map'] for r in selected})>=10
    source=read(source_path);assert all(r['source_model_sha256']==sha(source_path) for r in records)
    pins={str(index.resolve()):sha(index),str(source_path.resolve()):sha(source_path)}
    out.mkdir(parents=True);(out/'frozen-scripts').mkdir()
    for name in ('probe_combat_mean_advantage_cuda.py','combat_continuation_evidence.py','combat_teacher_folds.py',
                 'combat_teacher_logistic_cuda.py','combat_teacher_physical_features.py','combat_action_effect.py',
                 'combat_action_geometry.py','combat_body_features.py','ppo_combat.py','process_combat_architecture_pool.py'):
        path=repo/'scripts'/name;pins[str(path)]=sha(path);shutil.copyfile(path,out/'frozen-scripts'/name)
    for corpus in read(index)['corpora']:
        path=repo/corpus['manifest'];assert sha(path)==corpus['manifest_sha256'];pins[str(path)]=sha(path)
        for entry in read(path)['entries']:assert sha(entry['path'])==entry['sha256'];pins[entry['path']]=entry['sha256']
    protocol=dict(version='combat_paired_mean_advantage_probe_v1',source_model_sha256=sha(source_path),
        contexts=20,fit_contexts=14,excluded_mixed_contexts=6,primary='physical24',secondary='frozen16',baseline='constant',
        ridge_candidates=[.1,1.,10.,100.],outer_map_folds=4,inner_train_map_folds=3,std_floor=.05,
        target='CUDA mean of all3/7 paired finite observed return deltas; ties retained. Each world has equal total weight,not one weight per stream. No hard binary target.',
        diagnostics='Report estimated within-world sample variance/n; no inverse-variance weights from3samples. Fixed-OOF map-cluster bootstrap4000 draws is descriptive,not coverage or policy-quality proof.',
        bounded_correction='delta_logit=.5*tanh(predicted_mean/TRAIN_target_RMS_floor1),same transform for constant and contextual prediction; source future policy heldfixed in evidence.',
        gate='Primary OOF MSE<=.95constant;positive observed root surrogate gain vs source andconstant;map-bootstrap2.5percentile of excessoverconstant>0',
        optimization='CUDA float64 weighted ridge closed-form; unpenalized intercept; TRAIN-only normalization and inner ridge selection.',
        scope='Exploratory reused TRAIN roots;14autonomous contexts with3/7streams. Six mixed contexts remain in source corpus but cannot supervise autonomous full-return. Not a new actor or native quality comparison.',
        actor_compatible=False,promotion=False,inputs=pins)
    save(out/'protocol.json',protocol)
    training_devices();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    self_test();dtype=torch.float64;device='cuda'
    context=torch.tensor([r['input']['pre_gate_context187'] for r in selected],device=device,dtype=dtype)
    targets=[];noise=[]
    for record in selected:
        delta=torch.tensor([s['outcomes']['choice1']['utility']-s['outcomes']['choice0']['utility'] for s in record['streams']],device=device,dtype=dtype)
        targets.append(delta.mean());noise.append(delta.var(correction=1)/len(delta))
    y=torch.stack(targets);noise=torch.stack(noise)
    with torch.no_grad():
        physical=features(context);hidden=context
        for layer in source['wide_aim']['layers'][:2]:
            hidden=torch.nn.functional.relu(torch.nn.functional.linear(hidden,
                torch.tensor(layer['weight'],device=device,dtype=dtype),torch.tensor(layer['bias'],device=device,dtype=dtype)))
        layer=source['wide_aim']['layers'][2]
        base=torch.nn.functional.linear(hidden,torch.tensor(layer['weight'],device=device,dtype=dtype),
            torch.tensor(layer['bias'],device=device,dtype=dtype))[:,0]
    designs={'constant':physical[:,:0],'physical24':physical,'frozen16':hidden}
    ids=map_folds(selected);predictions={k:torch.full_like(y,float('nan')) for k in designs};scales=torch.full_like(y,float('nan'))
    def weights(indices):return torch.tensor(world_weights([selected[i] for i in indices]),device=device,dtype=dtype)
    def design(kind,train,other):
        x=designs[kind];mean,std=normalizer(x[train],weights(train))
        matrix=lambda indices:torch.cat(((x[indices]-mean)/std,torch.ones((len(indices),1),device=device,dtype=dtype)),-1)
        return matrix(train),matrix(other),mean,std
    def mse(predicted,target,w):return float(((predicted-target).square()*w).sum()/w.sum())
    fold_reports=[]
    try:
        for outer in range(4):
            train=[i for i,f in enumerate(ids) if f!=outer];test=[i for i,f in enumerate(ids) if f==outer]
            assert {selected[i]['map'] for i in train}.isdisjoint({selected[i]['map'] for i in test})
            inner_ids=map_folds([selected[i] for i in train],3)
            scale=((y[train].square()*weights(train)).sum()/weights(train).sum()).sqrt().clamp_min(1.)
            scales[test]=scale
            fold_report=dict(fold=outer,heldout_maps=sorted({selected[i]['map'] for i in test}),train_rows=len(train),heldout_rows=len(test),fits={})
            for kind in designs:
                penalty=0.;selection=[]
                if kind!='constant':
                    for candidate in protocol['ridge_candidates']:
                        inner_prediction=torch.full((len(train),),float('nan'),device=device,dtype=dtype)
                        for inner in range(3):
                            ti=[i for i,f in zip(train,inner_ids) if f!=inner];vi=[i for i,f in zip(train,inner_ids) if f==inner]
                            assert {selected[i]['map'] for i in ti}.isdisjoint({selected[i]['map'] for i in vi})
                            tr,va,_,_=design(kind,ti,vi);theta=ridge_fit(tr,y[ti],weights(ti),candidate)
                            positions=[i for i,f in enumerate(inner_ids) if f==inner];inner_prediction[positions]=va@theta
                        assert bool(torch.isfinite(inner_prediction).all())
                        selection.append(dict(ridge=candidate,mse=mse(inner_prediction,y[train],weights(train))))
                    penalty=min(selection,key=lambda r:(r['mse'],-r['ridge']))['ridge']
                tr,te,mean,std=design(kind,train,test);theta=ridge_fit(tr,y[train],weights(train),penalty)
                predictions[kind][test]=te@theta
                path=out/f'fold-{outer}-{kind}.pt'
                torch.save(dict(actor_compatible=False,theta=theta,mean=mean,std=std,ridge=penalty,
                    protocol_sha256=sha(out/'protocol.json'),kind=kind),path)
                fold_report['fits'][kind]=dict(ridge=penalty,inner_scores=selection,parameters=len(theta),checkpoint_sha256=sha(path))
            fold_reports.append(fold_report);save(out/'progress.json',dict(stage='map_cv',completed_folds=outer+1))
        w=weights(list(range(len(selected))));p0=base.sigmoid()
        assert all(bool(torch.isfinite(p).all()) for p in predictions.values()) and bool(torch.isfinite(scales).all())
        gains={kind:((base+.5*(pred/scales).tanh()).sigmoid()-p0)*y for kind,pred in predictions.items()}
        scores={kind:dict(mse=mse(pred,y,w),root_surrogate_gain=float((gains[kind]*w).sum()/w.sum())) for kind,pred in predictions.items()}
        maps=sorted({r['map'] for r in selected});sums=[];counts=[]
        excess=gains['physical24']-gains['constant']
        for name in maps:
            members=[i for i,r in enumerate(selected) if r['map']==name]
            sums.append((excess[members]*w[members]).sum());counts.append(w[members].sum())
        sums,counts=torch.stack(sums),torch.stack(counts)
        generator=torch.Generator(device=device);generator.manual_seed(20261026)
        draws=torch.randint(len(maps),(4000,len(maps)),generator=generator,device=device)
        boot=sums[draws].sum(-1)/counts[draws].sum(-1)
        interval=torch.quantile(boot,torch.tensor([.025,.975],device=device,dtype=dtype)).tolist()
        ratio=scores['physical24']['mse']/scores['constant']['mse']
        excess_mean=float((excess*w).sum()/w.sum())
        passed=ratio<=.95 and scores['physical24']['root_surrogate_gain']>0 and excess_mean>0 and interval[0]>0
        for path,digest in pins.items():assert sha(path)==digest,path
        save(out/'report.json',dict(state='complete',device=device,self_test_passed=True,scores=scores,
            primary_mse_ratio=ratio,primary_excess_root_surrogate_gain=excess_mean,
            fixed_oof_map_bootstrap_excess_interval=interval,learnability_gate_passed=passed,
            contexts=len(records),fit_contexts=len(selected),fit_maps=len(maps),
            estimated_mean_target_noise_variance=float((noise*w).sum()/w.sum()),folds=fold_reports,
            predictions=[dict(id=r['id'],map=r['map'],seed=r['seed'],fold=ids[i],streams=len(r['streams']),
                empirical_mean=float(y[i]),mean_estimation_variance=float(noise[i]),source_probability=float(p0[i]),
                predicted={k:float(p[i]) for k,p in predictions.items()}) for i,r in enumerate(selected)],
            excluded=[dict(id=r['id'],map=r['map'],reason='mixed_control_in_full_continuations') for r in records if not r['summary']['all_streams_autonomous']],
            policy_training_performed=False,actor_compatible=False,promotion=False,scope=protocol['scope']))
        save(out/'progress.json',dict(stage='complete'));print(out/'report.json',flush=True)
        print(scores,'ratio',ratio,'excess',excess_mean,'interval',interval,'gate',passed,flush=True)
    except BaseException as error:
        save(out/'progress.json',dict(stage='failed',error=str(error)));raise


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('index','source','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();run(a.index,a.source,a.out.resolve())

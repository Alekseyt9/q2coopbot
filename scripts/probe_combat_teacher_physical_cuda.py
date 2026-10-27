"""Nested map-CV of causal physical gate features; no game-policy export."""
import argparse
import shutil
from pathlib import Path

from combat_native_teacher_registry import load_index
from combat_teacher_folds import map_folds,world_weights
from combat_teacher_logistic_cuda import fit,normalizer,metrics
from combat_teacher_physical_features import VERSION,NAMES,features
from combat_wide_aim import validate
from ppo_combat import torch,training_devices
from process_combat_architecture_pool import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('index','source','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();repo=Path(__file__).resolve().parents[1];out=a.out.resolve()
    assert not out.exists()
    records=load_index(repo,a.index);labelled=[r for r in records if r['target']['preference'] is not None]
    source=read(a.source);validate(source);source_sha=sha(a.source)
    assert all(r['source_model_sha256']==source_sha for r in records)
    assert len(labelled)>=40 and len({r['map'] for r in labelled})>=16
    assert min(sum(r['target']['preference']==c for r in labelled) for c in (0,1))>=10
    pins={str(p.resolve()):sha(p) for p in (a.index,a.source,Path(__file__))}
    out.mkdir(parents=True);(out/'frozen-scripts').mkdir()
    for name in ('probe_combat_teacher_physical_cuda.py','combat_teacher_physical_features.py',
        'combat_teacher_logistic_cuda.py','combat_teacher_folds.py','combat_native_teacher_registry.py',
        'combat_action_effect.py','combat_action_geometry.py','combat_body_features.py','combat_wide_aim.py'):
        p=repo/'scripts'/name;pins[str(p)]=sha(p);shutil.copyfile(p,out/'frozen-scripts'/name)
    for corpus in read(a.index)['corpora']:
        p=(repo/corpus['manifest']).resolve();pins[str(p)]=sha(p)
        for entry in read(p)['entries']:
            p=Path(entry['path']);pins[str(p)]=entry['sha256']
    protocol=dict(version='combat_teacher_physical_probe_v1',feature_version=VERSION,
        names=NAMES,source_sha256=source_sha,records=len(records),labels=len(labelled),
        primary='physical24',secondary='geometry16',baseline='bias_only',
        parameters=dict(physical24=25,geometry16=17,bias_only=1),outer_folds=4,inner_folds=3,
        ridge_candidates=[.01,.1,1.,10.],standard_deviation_floor=.05,
        optimization='CUDA float64 convex logistic Newton,TRAIN gradient stopping,60 iteration cap,unpenalized intercept',
        standardization='World-balanced mean/std fitted only on current TRAIN partition,including every inner fold',
        advance_gate='Primary OOF world-weighted log loss<=.95*bias_only and balanced accuracy>=.55',
        scope='Reused TRAIN-teacher map folds,architecture hypothesis after prior negative readout. Nested ridge selection uses only inner TRAIN maps. Not independent confirmation or combat quality.',
        inputs='Only stored pre-gate context187. Current observed centered muzzle/bbox geometry and requested-motion contrasts,not forecasts,ray traces or future outcomes.',
        diagnostic_only=True,actor_compatible=False,promotion=False,inputs_sha256=pins)
    save(out/'protocol.json',protocol)
    training_devices();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dtype=torch.float64;device='cuda'
    context=torch.tensor([r['input']['pre_gate_context187'] for r in labelled],device=device,dtype=dtype)
    y=torch.tensor([r['target']['preference'] for r in labelled],device=device,dtype=dtype)
    with torch.no_grad():
        x=features(context);hidden=context
        for layer in source['wide_aim']['layers'][:2]:
            hidden=torch.nn.functional.relu(torch.nn.functional.linear(hidden,
                torch.tensor(layer['weight'],device=device,dtype=dtype),torch.tensor(layer['bias'],device=device,dtype=dtype)))
        last=source['wide_aim']['layers'][2]
        base=torch.nn.functional.linear(hidden,torch.tensor(last['weight'],device=device,dtype=dtype),
                                        torch.tensor(last['bias'],device=device,dtype=dtype))[:,0]
    fold_ids=map_folds(labelled)
    save(out/'splits.json',dict(records=[dict(id=r['id'],map=r['map'],seed=r['seed'],fold=f) for r,f in zip(labelled,fold_ids)]))
    save(out/'features.json',dict(feature_version=VERSION,names=NAMES,
        records=[dict(id=r['id'],values=v) for r,v in zip(labelled,x.tolist())],
        diagnostic_only=True,future_outcome_inputs=False))
    predictions={key:torch.empty_like(y) for key in ('bias_only','geometry16','physical24')}
    fold_reports=[]

    def weights(ids):
        return torch.tensor(world_weights([labelled[i] for i in ids]),device=device,dtype=dtype)

    def design_for(kind,train,other):
        width={'bias_only':0,'geometry16':16,'physical24':24}[kind]
        mean,std=normalizer(x[train,:width],weights(train))
        def matrix(ids):
            return torch.cat(((x[ids,:width]-mean)/std,torch.ones((len(ids),1),device=device,dtype=dtype)),-1)
        return matrix(train),matrix(other),mean,std

    try:
        for outer in range(4):
            train=[i for i,f in enumerate(fold_ids) if f!=outer]
            test=[i for i,f in enumerate(fold_ids) if f==outer]
            assert {labelled[i]['map'] for i in train}.isdisjoint({labelled[i]['map'] for i in test})
            inner_ids=map_folds([labelled[i] for i in train],3)
            fold_report=dict(fold=outer,train_rows=len(train),heldout_rows=len(test),
                heldout_maps=sorted({labelled[i]['map'] for i in test}),fits={})
            for kind in ('bias_only','geometry16','physical24'):
                ridge=0.;selection=[]
                if kind!='bias_only':
                    for candidate in protocol['ridge_candidates']:
                        inner_prediction=torch.empty(len(train),device=device,dtype=dtype)
                        for inner in range(3):
                            ti=[i for i,f in zip(train,inner_ids) if f!=inner]
                            vi=[i for i,f in zip(train,inner_ids) if f==inner]
                            assert {labelled[i]['map'] for i in ti}.isdisjoint({labelled[i]['map'] for i in vi})
                            tr,va,_,_=design_for(kind,ti,vi)
                            theta,_=fit(base[ti],tr,y[ti],weights(ti),candidate)
                            positions=[j for j,f in enumerate(inner_ids) if f==inner]
                            inner_prediction[positions]=(base[vi]+va@theta).clamp(-20,20)
                        score=metrics(inner_prediction,y[train],weights(train))
                        selection.append(dict(ridge=candidate,log_loss=score['log_loss']))
                    ridge=min(selection,key=lambda v:(v['log_loss'],-v['ridge']))['ridge']
                tr,te,mean,std=design_for(kind,train,test)
                theta,optimization=fit(base[train],tr,y[train],weights(train),ridge)
                predictions[kind][test]=(base[test]+te@theta).clamp(-20,20)
                checkpoint=out/f'fold-{outer}-{kind}.pt'
                torch.save(dict(version='combat_teacher_physical_diagnostic_checkpoint_v1',actor_compatible=False,
                    protocol_sha256=sha(out/'protocol.json'),theta=theta,normalizer_mean=mean,normalizer_std=std,
                    ridge=ridge,kind=kind,feature_version=VERSION),checkpoint)
                fold_report['fits'][kind]=dict(ridge=ridge,inner_scores=selection,optimization=optimization,
                    checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),train_parameters=len(theta))
            fold_reports.append(fold_report)
            save(out/'progress.json',dict(stage='nested_map_cv',completed_folds=outer+1))
        w=weights(list(range(len(labelled))))
        scores={key:metrics(value,y,w) for key,value in predictions.items()}
        ratio=scores['physical24']['log_loss']/scores['bias_only']['log_loss']
        advance=ratio<=.95 and scores['physical24']['balanced_accuracy']>=.55
        for path,digest in pins.items():assert sha(path)==digest,path
        save(out/'report.json',dict(state='complete',device=device,records=len(records),labels=len(labelled),
            maps=len({r['map'] for r in labelled}),scores=scores,physical_to_bias_loss_ratio=ratio,
            advance_to_native=advance,folds=fold_reports,diagnostic_only=True,actor_compatible=False,
            training_performed=True,policy_training_performed=False,promotion=False,scope=protocol['scope'],
            predictions=[dict(id=r['id'],map=r['map'],seed=r['seed'],frame=r['frame'],label=r['target']['preference'],fold=f,
                probabilities={key:float(v[i].sigmoid()) for key,v in predictions.items()}) for i,(r,f) in enumerate(zip(labelled,fold_ids))]))
        save(out/'progress.json',dict(stage='complete'));print(out/'report.json')
    except BaseException as error:
        save(out/'progress.json',dict(stage='failed',error=str(error)));raise


if __name__=='__main__':main()

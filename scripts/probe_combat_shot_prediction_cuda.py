"""Predict immediate native MG damage from causal observation/action inputs.

Whole-battle holdouts inside TRAIN only. Never changes the combat policy.
"""
import argparse
import collections
import gzip
import json
import pathlib
from evaluate_registered_combat_models import read,save,sha


def rows(path):
    with pathlib.Path(path).open(encoding='utf-8-sig') as stream:return [json.loads(s) for s in stream]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    args=ap.parse_args();root=args.corpus.resolve();out=args.out.resolve();assert not out.exists()
    pins={}
    def pin(path,expected=None):
        path=pathlib.Path(path).resolve();digest=sha(path);assert expected is None or digest==expected
        pins[str(path)]=digest;return path
    seal=read(pin(root/'complete.json'));assert seal['state']=='complete' and not seal['training_performed']
    protocol=read(pin(root/'protocol.json',seal['protocol_sha256']));assert protocol['train_only']
    pool=read(pin(root/'pool/report.json',seal['pool_sha256']));assert pool['state']=='complete' and pool['source_unchanged']
    assert len(pool['plans'])==1 and len(pool['jobs'])==32 and not any(j['error'] for j in pool['jobs'])
    plan=read(pin(pool['plans'][0]['path'],pool['plans'][0]['sha256']))
    assert all(t['split']=='train' and t['modes']==['learned'] and len(t['seeds'])==4 for t in plan['tasks'])
    meta=read(pin(root/'processing-train/merged/report.json'))
    observed=rows(pin(root/'processing-train/merged/rollout.jsonl',meta['rollout_sha256']))
    lookup={(v['seed'],v['index']):v for v in observed};assert len(lookup)==len(observed)==meta['rows']
    from report_combat_machinegun_hits import scan
    from report_registered_machinegun_outcomes import outcome
    records=[];samples=[];features=[]
    held_seeds={max(t['seeds']) for t in plan['tasks']}
    for job in sorted(pool['jobs'],key=lambda v:v['job']):
        assert job['mode']=='learned' and job['seed'] in plan['tasks'][job['task_index']]['seeds']
        task=plan['tasks'][job['task_index']];episode=task['episode']['id']
        capture=read(pin(pathlib.Path(job['root'])/'report.json'));assert capture['capture_complete'] and capture['provenance_valid']
        native=root/f"processing-train/case-{job['job']:04d}/native/replay-0"
        steps=rows(pin(native/'steps.jsonl'));windows={}
        for step in steps:
            if (job['seed'],step['index']) not in lookup:continue
            dispatch=step.get('server_execution') or {};n=step.get('native_step')
            if not n or step['owner']!='provider' or not dispatch.get('matched') or not dispatch.get('window_exclusive') or dispatch.get('recovery_commands',0):continue
            key=(n['spawncount'],n['begin_frame'],n['actor'],n['sequence']);assert key not in windows
            windows[key]=step
        log=pin(pathlib.Path(capture['results'][0]['root'])/'server.log')
        with log.open(encoding='utf-8-sig') as stream:shots=[s for s in scan(stream,set(windows)) if s['eligible']]
        for shot in shots:
            step=windows[shot['window']];row=lookup[(job['seed'],step['index'])]
            assert row['sample']['attack'] and row['frame']==step['observation']['identity']['frame']
            label=outcome(shot)
            records.append(dict(seed=job['seed'],index=step['index'],episode=episode,task=job['task_index'],
                shot=int(shot['fields']['shot']),outcome=label,positive=label=='live_monster_damage',
                selected_target=row['sample'].get('target',0)>0,seed_holdout=job['seed'] in held_seeds,
                site_holdout='-site-02-' in episode))
            features.append(row['features']);samples.append(row['sample'])
    assert len({(v['seed'],v['shot']) for v in records})==len(records)>0
    out.mkdir(parents=True)
    save(out/'protocol.json',dict(version='combat_shot_prediction_probe_v1',input='conditional_fire181_causal_inputs',
        label='immediate_live_monster_health_damage_given_an_actual_native_MG_shot',steps=200,lr=.001,weight_decay=.0001,seed=20261022,
        splits=['last_seed_per_family','held_out_site02'],models=['train_constant','selected_target_strata','linear181','mlp181x32x16x1'],
        normalization='train_only_mean_std',checkpoint_selection=False,policy_training=False,runtime_integrated=False,promotion=False,
        scope='Only existing TRAIN fights; whole battles held out. No no-shot counterfactual labels. Predicting immediate contact is not choosing the optimal fire action.'))
    with gzip.open(out/'members.jsonl.gz','wt',encoding='utf-8') as stream:
        for record in records:stream.write(json.dumps(record)+'\n')
    from ppo_combat import torch,nn,training_devices
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    from combat_conditional_fire import inputs
    x=torch.tensor(features,dtype=torch.float32);latent=torch.tensor([s['latent'] for s in samples],dtype=torch.float32)
    cat=lambda name:torch.tensor([s.get(name,0) for s in samples],dtype=torch.long)
    values=inputs(x,cat('target'),cat('aim_mode'),latent,cat('vertical'),cat('weapon'))
    y=torch.tensor([v['positive'] for v in records],dtype=torch.float32)
    selected=torch.tensor([v['selected_target'] for v in records],dtype=torch.bool)
    torch.save(dict(version='combat_shot_prediction_data_v1',features=values,labels=y,selected_target=selected),out/'dataset.pt')
    def measure(probability,mask):
        if not bool(mask.any()):return None
        p=probability[mask].clamp(1e-6,1-1e-6);target=y[mask]
        positive,negative=p[target==1],p[target==0]
        auc=float(((positive[:,None]>negative[None,:]).float()+.5*(positive[:,None]==negative[None,:])).mean()) if len(positive) and len(negative) else None
        return dict(rows=int(mask.sum()),positive=int(target.sum()),log_loss=float(nn.functional.binary_cross_entropy(p,target)),
            brier=float((p-target).square().mean()),auc=auc)
    reports={}
    for split,field in [('last_seed_per_family','seed_holdout'),('held_out_site02','site_holdout')]:
        test=torch.tensor([v[field] for v in records],dtype=torch.bool);train=~test
        assert bool(train.any()) and bool(test.any())
        train_seeds={v['seed'] for v in records if not v[field]};test_seeds={v['seed'] for v in records if v[field]}
        assert train_seeds.isdisjoint(test_seeds)
        mean=values[train].mean(0);std=values[train].std(0,unbiased=False);std=torch.where(std<1e-3,torch.ones_like(std),std)
        normalized=(values-mean)/std
        constant=torch.full_like(y,float((y[train].sum()+1)/(train.sum()+2)))
        strata=constant.clone()
        for category in (False,True):
            mask=train&(selected==category)
            rate=(y[mask].sum()+1)/(mask.sum()+2)
            strata[selected==category]=rate
        probabilities={'train_constant':constant,'selected_target_strata':strata}
        for name in ('linear181','mlp181x32x16x1'):
            torch.manual_seed(20261022)
            model=nn.Linear(181,1,device='cuda') if name=='linear181' else nn.Sequential(nn.Linear(181,32,device='cuda'),nn.ReLU(),nn.Linear(32,16,device='cuda'),nn.ReLU(),nn.Linear(16,1,device='cuda'))
            optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
            for _ in range(200):
                loss=nn.functional.binary_cross_entropy_with_logits(model(normalized[train])[:,0],y[train])
                assert bool(torch.isfinite(loss));optimizer.zero_grad();loss.backward();optimizer.step()
            with torch.no_grad():probabilities[name]=model(normalized)[:,0].sigmoid()
            torch.save(dict(model=model.state_dict(),mean=mean,std=std,steps=200,split=split,
                input_width=181,parameters=sum(p.numel() for p in model.parameters())),out/(split+'-'+name+'.pt'))
        reports[split]=dict(train_seeds=sorted(train_seeds),test_seeds=sorted(test_seeds),models={name:dict(
            train=measure(p,train),held_out=measure(p,test),held_out_selected_target=measure(p,test&selected)) for name,p in probabilities.items()})
    for path,digest in pins.items():assert sha(pathlib.Path(path))==digest
    outcomes=collections.Counter(v['outcome'] for v in records)
    save(out/'report.json',dict(state='complete',version='combat_shot_prediction_probe_v1',device='cuda',shots=len(records),outcomes=dict(outcomes),
        splits=reports,inputs=pins,script_sha256=sha(__file__),dataset_sha256=sha(out/'dataset.pt'),members_sha256=sha(out/'members.jsonl.gz'),
        policy_training=False,runtime_integrated=False,promotion=False,
        scope='Diagnostic supervised immediate shot outcome prediction. Only causal actor inputs; future/native fields are labels. Whole TRAIN fights and site02 held out. Fixed200steps without checkpoint selection. No combat-policy improvement or safe fire-threshold claim.'))
    save(out/'complete.json',dict(state='complete',report_sha256=sha(out/'report.json'),policy_training=False,promotion=False))
    print(json.dumps(dict(shots=len(records),outcomes=dict(outcomes),held_out={split:{name:stats['held_out'] for name,stats in v['models'].items()} for split,v in reports.items()})))


if __name__=='__main__':main()

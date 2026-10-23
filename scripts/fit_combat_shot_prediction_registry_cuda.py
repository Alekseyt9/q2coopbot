"""Fit fixed predictive models from a sealed compact TRAIN shot dataset."""
import argparse
import gzip
import json
import time
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True)
    args=ap.parse_args();path=args.protocol.resolve();out=path.parent;p=read(path)
    assert p['version']=='combat_shot_prediction_registry_probe_v2'
    assert p['steps']==200 and p['lr']==.001 and p['weight_decay']==.0001 and p['seed']==20261022
    assert p['models']==['train_constant','selected_target_strata','linear181','mlp181x32x16x1']
    assert not p['policy_training'] and not p['checkpoint_selection'] and not p['runtime_integrated'] and not p['promotion']
    assert not any(out.glob('*.pt')) and not (out/'report.json').exists(),'Fresh fit required'
    data=Path(p['data']);seal=read(data/'complete.json');report=read(data/'report.json')
    assert seal['state']==report['state']=='complete' and not seal['ppo_ready']
    assert sha(data/'report.json')==seal['report_sha256']
    assert sha(data/'dataset.pt')==report['dataset_sha256']==seal['dataset_sha256']
    assert sha(data/'members.jsonl.gz')==report['members_sha256']
    pins={str(v.resolve()):sha(v) for v in (path,data/'complete.json',data/'report.json',data/'dataset.pt',data/'members.jsonl.gz')}
    with gzip.open(data/'members.jsonl.gz','rt',encoding='utf-8') as stream:members=[json.loads(s) for s in stream]
    from ppo_combat import torch,nn,training_devices
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    tensors=torch.load(data/'dataset.pt',map_location='cuda',weights_only=True)
    assert tensors['version']=='combat_shot_prediction_data_v2'
    x,y,selected=(tensors[k] for k in ('features','labels','selected_target'))
    assert x.shape==(len(members),181) and len(members)==report['shots']
    assert bool(torch.isfinite(x).all()) and bool(((y==0)|(y==1)).all())
    assert torch.equal(y,torch.tensor([v['positive'] for v in members],dtype=torch.float32))
    assert torch.equal(selected,torch.tensor([v['selected_target'] for v in members],dtype=torch.bool))
    assert torch.equal(selected,x[:,151]==0),'Selected-target metadata differs from causal one-hot input'
    def measure(probability,mask):
        if not bool(mask.any()):return None
        predicted=probability[mask].clamp(1e-6,1-1e-6);target=y[mask]
        yes,no=predicted[target==1],predicted[target==0]
        auc=float(((yes[:,None]>no[None,:]).float()+.5*(yes[:,None]==no[None,:])).mean()) if len(yes) and len(no) else None
        return dict(rows=int(mask.sum()),positive=int(target.sum()),mean_probability=float(predicted.mean()),
            log_loss=float(nn.functional.binary_cross_entropy(predicted,target)),brier=float((predicted-target).square().mean()),auc=auc)
    reports={};folds={};checkpoint_pins={};start=time.perf_counter()
    for split,field in [('seed_mod4_eq3','seed_holdout'),('held_out_site02','site_holdout')]:
        test=tensors[field];train=~test
        assert torch.equal(test,torch.tensor([v[field] for v in members],dtype=torch.bool))
        train_groups={(v['episode'],v['seed']) for v in members if not v[field]}
        test_groups={(v['episode'],v['seed']) for v in members if v[field]}
        assert train_groups and test_groups and train_groups.isdisjoint(test_groups)
        folds[split]=dict(train=sorted(train_groups),held_out=sorted(test_groups))
        mean=x[train].mean(0);std=x[train].std(0,unbiased=False);std=torch.where(std<1e-3,torch.ones_like(std),std)
        values=(x-mean)/std
        assert bool(torch.isfinite(values).all())
        constant=torch.full_like(y,float((y[train].sum()+1)/(train.sum()+2)))
        strata=constant.clone()
        for category in (False,True):
            mask=train&(selected==category);strata[selected==category]=(y[mask].sum()+1)/(mask.sum()+2)
        probabilities={'train_constant':constant,'selected_target_strata':strata}
        for name in ('linear181','mlp181x32x16x1'):
            save(out/'progress.json',dict(stage='fitting',split=split,model=name))
            torch.manual_seed(p['seed'])
            model=nn.Linear(181,1,device='cuda') if name=='linear181' else nn.Sequential(nn.Linear(181,32,device='cuda'),nn.ReLU(),nn.Linear(32,16,device='cuda'),nn.ReLU(),nn.Linear(16,1,device='cuda'))
            opt=torch.optim.Adam(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
            for _ in range(p['steps']):
                loss=nn.functional.binary_cross_entropy_with_logits(model(values[train])[:,0],y[train])
                assert bool(torch.isfinite(loss));opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():probabilities[name]=model(values)[:,0].sigmoid()
            file=out/(split+'-'+name+'.pt')
            torch.save(dict(version='combat_shot_prediction_checkpoint_v1',kind=name,model=model.state_dict(),optimizer=opt.state_dict(),
                mean=mean,std=std,input_width=181,parameters=sum(v.numel() for v in model.parameters()),steps=p['steps'],split=split,
                dataset_sha256=report['dataset_sha256'],protocol_sha256=pins[str(path)]),file)
            checkpoint_pins[str(file)]=sha(file)
            print(split,name,'complete',flush=True)
        reports[split]=dict(train_battles=len(train_groups),held_out_battles=len(test_groups),models={name:dict(
            train=measure(v,train),held_out=measure(v,test),held_out_selected_target=measure(v,test&selected),held_out_no_target=measure(v,test&~selected)) for name,v in probabilities.items()})
    torch.cuda.synchronize()
    for name,digest in pins.items():assert sha(Path(name))==digest,'Pinned input changed'
    save(out/'folds.json',folds)
    result=dict(state='complete',version='combat_shot_prediction_registry_probe_v2',device='cuda',shots=len(members),source_cases=report['source_cases'],
        battles_with_shots=report['battles_with_shots'],duplicate_battles=report['duplicate_battles'],splits=reports,seconds=time.perf_counter()-start,
        inputs=pins,checkpoints=checkpoint_pins,folds_sha256=sha(out/'folds.json'),script_sha256=sha(__file__),policy_training=False,runtime_integrated=False,promotion=False,
        scope='Predictive development only on a pooled TRAIN registry; whole episode/seed groups and site02 held out. Same fixed200steps and simple baselines. Shot frequency weighting; not independent per-shot statistical confidence or optimal fire-action supervision.')
    save(out/'report.json',result)
    save(out/'complete.json',dict(state='complete',report_sha256=sha(out/'report.json'),policy_training=False,promotion=False))
    save(out/'progress.json',dict(stage='complete'))
    print(json.dumps(dict(shots=len(members),seconds=result['seconds'],held_out={split:{name:v['held_out'] for name,v in item['models'].items()} for split,item in reports.items()})))


if __name__=='__main__':main()

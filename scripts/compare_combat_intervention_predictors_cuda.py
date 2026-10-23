"""Fixed-budget predictor training on old, randomized and mixed TRAIN experience."""
import argparse
import gzip
import json
import time
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True)
    args=ap.parse_args();path=args.protocol.resolve();out=path.parent;p=read(path);pins={}
    assert p['version']=='combat_interventional_predictor_probe_v1'
    assert p['regimes']==['legacy_only','intervention_only','pooled','balanced']
    assert (p['steps'],p['lr'],p['weight_decay'],p['seed'])==(200,.001,.0001,20261022)
    assert not any(p[k] for k in ('policy_training','runtime_integrated','promotion'))
    assert not (out/'report.json').exists() and not any(out.glob('*.pt'))
    def pin(file,expected=None):
        file=Path(file).resolve();digest=sha(file);assert expected is None or digest==expected,str(file)
        pins[str(file)]=digest;return file
    pin(path);registry=read(pin(p['registry']))
    assert registry['purpose']=='predictive_train_only' and not registry['ppo_ready']
    folders=[Path(p['legacy_data'])]+[Path(e['dataset']) for e in registry['entries']]
    metadata=[];members=[]
    for folder in folders:
        seal=read(pin(folder/'complete.json'));report=read(pin(folder/'report.json',seal['report_sha256']))
        assert report['state']=='complete' and not seal['ppo_ready']
        pin(folder/'dataset.pt',seal['dataset_sha256']);assert report['dataset_sha256']==seal['dataset_sha256']
        with gzip.open(pin(folder/'members.jsonl.gz',report['members_sha256']),'rt',encoding='utf-8') as f:group=[json.loads(line) for line in f]
        assert len(group)==report['shots'];metadata.append(report);members.append(group)
    old_groups={(m['episode'],m['seed']) for m in members[0]}
    new_groups={(m['episode'],m['seed']) for group in members[1:] for m in group}
    assert old_groups.isdisjoint(new_groups)
    from ppo_combat import torch,nn,training_devices
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from combat_action_geometry import SMOOTH_VERSION,features_for_version
    pin(Path(__file__).with_name('combat_action_geometry.py'))
    sets=[torch.load(folder/'dataset.pt',map_location='cuda',weights_only=True) for folder in folders]
    for t in sets:
        assert t['geometry_version']==SMOOTH_VERSION
        assert torch.equal(features_for_version(t['features_with_recoil'],SMOOTH_VERSION),t['action_geometry'])
    x=torch.cat([torch.cat((t['features_with_recoil'],t['action_geometry']),-1) for t in sets])
    y=torch.cat([t['labels'] for t in sets]);selected=torch.cat([t['selected_target'] for t in sets])
    domain=torch.arange(len(y),device='cuda')>=len(members[0])
    perturbed=torch.cat([torch.zeros(len(members[0]),dtype=torch.bool,device='cuda')]+[t['intervened'] for t in sets[1:]])
    eligible=torch.cat([torch.zeros(len(members[0]),dtype=torch.bool,device='cuda')]+[t['exploration_eligible'] for t in sets[1:]])
    all_members=[m for group in members for m in group];assert len(all_members)==len(y)
    def metric(probability,mask):
        if not bool(mask.any()):return None
        target=y[mask];predicted=probability[mask].clamp(1e-6,1-1e-6)
        yes,no=predicted[target==1],predicted[target==0]
        auc=float(((yes[:,None]>no[None,:]).float()+.5*(yes[:,None]==no[None,:])).mean()) if len(yes) and len(no) else None
        return dict(rows=int(mask.sum()),positive=int(target.sum()),log_loss=float(nn.functional.binary_cross_entropy(predicted,target)),
            brier=float((predicted-target).square().mean()),auc=auc)
    results={};folds={};checkpoints={};start=time.perf_counter()
    for split,field in [('seed_mod4_eq3','seed_holdout'),('held_out_site02','site_holdout')]:
        test=torch.cat([t[field] for t in sets]);train=~test
        a={(m['episode'],m['seed']) for m in all_members if not m[field]};b={(m['episode'],m['seed']) for m in all_members if m[field]}
        assert a and b and a.isdisjoint(b);folds[split]=dict(train=sorted(a),held_out=sorted(b))
        assert bool((train&domain).any()) and bool((test&domain&perturbed).any())
        models={}
        for regime in p['regimes']:
            save(out/'progress.json',dict(stage='fitting',split=split,regime=regime))
            use=train&(~domain if regime=='legacy_only' else domain if regime=='intervention_only' else torch.ones_like(domain))
            values=x[use];targets=y[use]
            if regime=='balanced':
                d=domain[use];weights=torch.where(d,.5/d.sum(),.5/(~d).sum())
            else:weights=torch.full((len(values),),1/len(values),device='cuda')
            mean=(values*weights[:,None]).sum(0)
            std=((values-mean).square()*weights[:,None]).sum(0).sqrt();std=torch.where(std<1e-3,torch.ones_like(std),std)
            normalized=(values-mean)/std;torch.manual_seed(p['seed']);net=nn.Linear(214,1,device='cuda')
            optimizer=torch.optim.Adam(net.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
            for _ in range(p['steps']):
                loss=(nn.functional.binary_cross_entropy_with_logits(net(normalized)[:,0],targets,reduction='none')*weights).sum()
                assert bool(torch.isfinite(loss));optimizer.zero_grad();loss.backward();optimizer.step()
            with torch.no_grad():probability=net((x-mean)/std)[:,0].sigmoid()
            metrics={name:metric(probability,mask) for name,mask in {
                'old_held_out':test&~domain,'old_selected_held_out':test&~domain&selected,
                'new_held_out':test&domain,'new_intervened_held_out':test&domain&perturbed,
                'new_baseline_held_out':test&domain&eligible&~perturbed,'train_consumed':use}.items()}
            file=out/(split+'-'+regime+'.pt')
            torch.save(dict(version='combat_interventional_predictor_checkpoint_v1',regime=regime,model=net.state_dict(),
                optimizer=optimizer.state_dict(),mean=mean,std=std,steps=p['steps'],geometry_version=SMOOTH_VERSION,
                protocol_sha256=pins[str(path)]),file);checkpoints[str(file)]=sha(file)
            models[regime]=dict(train_rows=int(use.sum()),parameters=215,metrics=metrics,
                raw_yaw_coefficient=float((net.weight[0,164]/std[164]).detach()),raw_pitch_coefficient=float((net.weight[0,165]/std[165]).detach()))
            print(split,regime,'complete',flush=True)
        results[split]=models
    torch.cuda.synchronize();save(out/'folds.json',folds)
    candidates=[]
    for regime in ('pooled','balanced','intervention_only'):
        checks=[]
        for models in results.values():
            reference=models['legacy_only']['metrics'];candidate=models[regime]['metrics']
            checks.append(candidate['new_intervened_held_out']['log_loss']<reference['new_intervened_held_out']['log_loss'] and
                candidate['old_held_out']['log_loss']<=reference['old_held_out']['log_loss']+.02)
        if all(checks):candidates.append(regime)
    for file,digest in pins.items():assert sha(Path(file))==digest,file
    report=dict(state='complete',version=p['version'],device='cuda',legacy_shots=len(members[0]),new_shots=len(y)-len(members[0]),
        source_cases=sum(m['source_cases'] for m in metadata),splits=results,eligible_regimes=candidates,
        selected_regime=candidates[0] if candidates else None,seconds=time.perf_counter()-start,inputs=pins,checkpoints=checkpoints,
        folds_sha256=sha(out/'folds.json'),script_sha256=sha(__file__),policy_training=False,runtime_integrated=False,promotion=False,
        scope='Fixed linear214,200steps on grouped TRAIN folds; old-only/new-only/pooled/equal-domain loss. New labels come from actual randomized actions. Development selection does not establish counterfactual optimality or battle improvement.')
    save(out/'report.json',report);save(out/'complete.json',dict(state='complete',report_sha256=sha(out/'report.json'),policy_training=False,promotion=False))
    save(out/'progress.json',dict(stage='complete'))
    print(json.dumps(dict(eligible_regimes=candidates,selected_regime=report['selected_regime'],seconds=report['seconds'])))


if __name__=='__main__':main()

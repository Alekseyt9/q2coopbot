"""Matched TRAIN predictive ablation: base181, recoil185, geometry214 on CUDA."""
import argparse
import gzip
import json
import math
import time
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True)
    args=ap.parse_args();path=args.protocol.resolve();out=path.parent;p=read(path)
    assert p['version']=='combat_action_geometry_probe_v1'
    assert (p['steps'],p['lr'],p['weight_decay'],p['seed'])==(200,.001,.0001,20261022)
    models=['train_constant','selected_target_strata','linear181','linear185','linear214','mlp181x32x16x1','mlp185x32x16x1','mlp214x32x16x1']
    assert p['models']==models and p['splits']==['seed_mod4_eq3','held_out_site02']
    assert not any(p[k] for k in ('policy_training','checkpoint_selection','runtime_integrated','promotion'))
    assert not (out/'report.json').exists() and not any(out.glob('*.pt'))
    pins={}
    def pin(file,expected=None):
        file=Path(file).resolve();digest=sha(file);assert expected is None or digest==expected,str(file)
        pins[str(file)]=digest;return file
    pin(path)
    def dataset(folder):
        folder=Path(folder);seal=read(pin(folder/'complete.json'));report=read(pin(folder/'report.json',seal['report_sha256']))
        assert seal['state']==report['state']=='complete' and not seal['ppo_ready']
        pin(folder/'dataset.pt',seal['dataset_sha256']);assert report['dataset_sha256']==seal['dataset_sha256']
        with gzip.open(pin(folder/'members.jsonl.gz',report['members_sha256']),'rt',encoding='utf-8') as f:members=[json.loads(line) for line in f]
        return report,members
    report,members=dataset(p['data']);legacy_report,legacy_members=dataset(p['legacy_data'])
    assert members==legacy_members and report['cases']==legacy_report['cases']
    from ppo_combat import torch,nn,training_devices
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from combat_action_geometry import features_for_version,names_for_version
    pin(Path(__file__).with_name('combat_action_geometry.py'))
    tensors=torch.load(Path(p['data'])/'dataset.pt',map_location='cuda',weights_only=True)
    VERSION=tensors['geometry_version'];NAMES=names_for_version(VERSION)
    geometry=lambda x:features_for_version(x,VERSION)
    legacy=torch.load(Path(p['legacy_data'])/'dataset.pt',map_location='cuda',weights_only=True)
    assert tensors['version']=='combat_shot_prediction_data_v3'
    assert tensors['geometry_names']==list(NAMES)
    for key in ('features','labels','selected_target','seed_holdout','site_holdout'):
        assert torch.equal(tensors[key],legacy[key]),key
    causal=tensors['features_with_recoil'];extra=tensors['action_geometry'];y=tensors['labels'];selected=tensors['selected_target']
    assert causal.shape==(len(members),185) and extra.shape==(len(members),29)
    assert torch.equal(causal[:,:181],legacy['features']) and torch.equal(geometry(causal),extra)
    save(out/'legacy-parity.json',dict(state='complete',shots=len(members),members_and_case_metadata_exact=True,
        original_features_labels_and_folds_exact=True,geometry_recomputed_exact=True,
        legacy_dataset_sha256=legacy_report['dataset_sha256'],new_dataset_sha256=report['dataset_sha256']))
    features={181:tensors['features'],185:causal,214:torch.cat((causal,extra),-1)}
    assert torch.equal(selected,causal[:,151]==0)
    def metric(probability,mask):
        if not bool(mask.any()):return None
        predicted=probability[mask].clamp(1e-6,1-1e-6);target=y[mask]
        yes,no=predicted[target==1],predicted[target==0]
        auc=float(((yes[:,None]>no[None,:]).float()+.5*(yes[:,None]==no[None,:])).mean()) if len(yes) and len(no) else None
        return dict(rows=int(mask.sum()),positive=int(target.sum()),log_loss=float(nn.functional.binary_cross_entropy(predicted,target)),
            brier=float((predicted-target).square().mean()),auc=auc,mean_probability=float(predicted.mean()))
    eye=NAMES.index('eye_yaw_error_pi');muzzle=NAMES.index('muzzle_yaw_error_pi')
    limit=torch.where(causal[:,161]>0,15/180,1.)
    delta=extra[:,eye]
    changed={};sensitivity_masks={}
    for name,sign in [('away',-1.),('toward',1.)]:
        candidate=causal.clone()
        move=torch.minimum(delta.abs(),torch.full_like(delta,5/180)) if name=='toward' else torch.full_like(delta,5/180)
        candidate[:,164]=(candidate[:,164]+sign*delta.sign()*move).clamp(-limit,limit)
        geo=geometry(candidate)
        differences=geo[:,[eye,muzzle]].abs()-extra[:,[eye,muzzle]].abs()
        mask=(extra[:,0]>0)&((differences>1/180).all(-1) if name=='away' else (differences<(-1/180)).all(-1))
        changed[name]={181:candidate[:,:181],185:candidate,214:torch.cat((candidate,geo),-1)}
        sensitivity_masks[name]=mask
    folds={};results={};checkpoints={};start=time.perf_counter()
    for split,field in [('seed_mod4_eq3','seed_holdout'),('held_out_site02','site_holdout')]:
        test=tensors[field];train=~test
        assert torch.equal(test,torch.tensor([m[field] for m in members],dtype=torch.bool,device='cuda'))
        a={(m['episode'],m['seed']) for m in members if not m[field]};b={(m['episode'],m['seed']) for m in members if m[field]}
        assert a and b and a.isdisjoint(b);folds[split]=dict(train=sorted(a),held_out=sorted(b))
        constant=torch.full_like(y,float((y[train].sum()+1)/(train.sum()+2)))
        strata=constant.clone()
        for category in (False,True):
            mask=train&(selected==category);strata[selected==category]=(y[mask].sum()+1)/(mask.sum()+2)
        outputs={'train_constant':constant,'selected_target_strata':strata};sensitivity={};parameters={}
        for name in models[2:]:
            width=int(name[6:] if name.startswith('linear') else name[3:].split('x')[0])
            x=features[width];mean=x[train].mean(0);std=x[train].std(0,unbiased=False)
            std=torch.where(std<1e-3,torch.ones_like(std),std);values=(x-mean)/std
            torch.manual_seed(p['seed'])
            net=nn.Linear(width,1,device='cuda') if name.startswith('linear') else nn.Sequential(
                nn.Linear(width,32,device='cuda'),nn.ReLU(),nn.Linear(32,16,device='cuda'),nn.ReLU(),nn.Linear(16,1,device='cuda'))
            optimizer=torch.optim.Adam(net.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
            save(out/'progress.json',dict(stage='fitting',split=split,model=name))
            for _ in range(p['steps']):
                loss=nn.functional.binary_cross_entropy_with_logits(net(values[train])[:,0],y[train])
                assert bool(torch.isfinite(loss));optimizer.zero_grad();loss.backward();optimizer.step()
            with torch.no_grad():
                probability=net(values)[:,0].sigmoid();outputs[name]=probability
                sensitivity[name]={}
                for direction in changed:
                    candidate=net((changed[direction][width]-mean)/std)[:,0].sigmoid()
                    mask=test&sensitivity_masks[direction]
                    difference=candidate[mask]-probability[mask]
                    sensitivity[name][direction]=dict(rows=int(mask.sum()),mean_probability_delta=float(difference.mean()) if bool(mask.any()) else None,
                        fraction_expected_sign=float(((difference<0) if direction=='away' else (difference>0)).float().mean()) if bool(mask.any()) else None)
            parameters[name]=sum(v.numel() for v in net.parameters())
            file=out/(split+'-'+name+'.pt')
            torch.save(dict(version='combat_action_geometry_predictor_v1',kind=name,input_width=width,model=net.state_dict(),
                optimizer=optimizer.state_dict(),mean=mean,std=std,steps=p['steps'],parameters=parameters[name],
                geometry_version=VERSION,dataset_sha256=report['dataset_sha256'],protocol_sha256=pins[str(path)]),file)
            checkpoints[str(file)]=sha(file)
            print(split,name,'complete',flush=True)
        results[split]=dict(train_battles=len(a),held_out_battles=len(b),parameters=parameters,
            models={name:dict(train=metric(v,train),held_out=metric(v,test),
                held_out_selected_target=metric(v,test&selected),held_out_no_target=metric(v,test&~selected)) for name,v in outputs.items()},
            action_sensitivity=sensitivity)
    torch.cuda.synchronize()
    for file,digest in pins.items():assert sha(Path(file))==digest,file
    save(out/'folds.json',folds)
    result=dict(state='complete',version=p['version'],device='cuda',shots=len(members),source_cases=report['source_cases'],splits=results,
        seconds=time.perf_counter()-start,inputs=pins,checkpoints=checkpoints,script_sha256=sha(__file__),
        folds_sha256=sha(out/'folds.json'),legacy_parity_sha256=sha(out/'legacy-parity.json'),
        policy_training=False,runtime_integrated=False,promotion=False,
        scope='Same pooled TRAIN rows/labels/folds; fixed200steps, hidden sizes32/16. Geometric action sensitivity is unexecuted and has no counterfactual outcome labels. No raycast, future state, runtime mask, policy fitting or battle-quality claim.')
    save(out/'report.json',result)
    save(out/'complete.json',dict(state='complete',report_sha256=sha(out/'report.json'),policy_training=False,promotion=False))
    save(out/'progress.json',dict(stage='complete'))
    print(json.dumps(dict(seconds=result['seconds'],held_out={split:{name:info['held_out'] for name,info in item['models'].items()} for split,item in results.items()})))


if __name__=='__main__':main()

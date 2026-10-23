"""Cross-fitted motor forecast versus observed geometry for actual TRAIN MG labels."""
import argparse
import gzip
import json
import time
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True)
    args=ap.parse_args();path=args.protocol.resolve();out=path.parent;p=read(path);pins={}
    assert p['version'] in ('combat_action_effect_shot_probe_v1','combat_action_effect_shot_probe_v2')
    reuse=p['version']=='combat_action_effect_shot_probe_v2'
    assert p['forecaster']=='factorized_mlp76x32_21x16'
    assert (p['forecaster_steps'],p['forecaster_lr'],p['forecaster_weight_decay'],p['forecaster_seed'])==(300,.001,.0001,20261023)
    assert (p['predictor_steps'],p['predictor_lr'],p['predictor_weight_decay'],p['predictor_seed'])==(200,.001,.0001,20261022)
    arms=(['centered_current_geometry','centered_own_kinematics100ms_geometry','centered_target_kinematics100ms_geometry',
           'centered_both_kinematics100ms_geometry','learned_motor_current_target_geometry','learned_motor_target100ms_geometry'] if reuse else
          ['historical_right_current_geometry','centered_current_geometry','centered_kinematics100ms_geometry','learned_motor_kinematics100ms_geometry'])
    assert p['feature_arms']==arms and not any(p[k] for k in ('policy_training','runtime_integrated','promotion'))
    assert not (out/'report.json').exists() and not any(out.glob('*.pt'))
    def pin(file,expected=None):
        file=Path(file).resolve();digest=sha(file);assert expected is None or digest==expected,str(file)
        pins[str(file)]=digest;return file
    pin(path);parent=Path(p['forecaster_probe']);labels=Path(p['labels'])
    reused_report=None;reused_groups=None
    if reuse:
        previous=Path(p['reuse_crossfit'])
        rs=read(pin(previous/'complete.json'));reused_report=read(pin(previous/'report.json',rs['report_sha256']))
        assert reused_report['state']=='complete' and reused_report['device']=='cuda' and reused_report['version']=='combat_action_effect_shot_probe_v1'
        reused_groups=read(pin(previous/'crossfit-groups.json',reused_report['crossfit_groups_sha256']))
        previous_protocol=pin(previous/'protocol.json')
    ps=read(pin(parent/'complete.json'));pr=read(pin(parent/'report.json',ps['report_sha256']))
    assert pr['state']=='complete' and pr['device']=='cuda' and p['forecaster'] in pr['eligible_models']
    pp=read(pin(parent/'protocol.json'));assert pp['version']=='combat_action_effect_probe_v3'
    data_path=pin(parent/'dataset.pt',ps['dataset_sha256'])
    parent_folds=read(pin(parent/'folds.json',pr['folds_sha256']))
    ls=read(pin(labels/'complete.json'));lr=read(pin(labels/'report.json',ls['report_sha256']))
    assert lr['state']=='complete' and lr['hand']==2 and not lr['neural_execution']
    with gzip.open(pin(labels/'members.jsonl.gz',lr['members_sha256']),'rt',encoding='utf-8') as f:
        members=[json.loads(line) for line in f]
    from ppo_combat import torch,nn,training_devices
    from combat_action_effect import inputs,FactorizedMotor,projected_geometry,POSITION_COLUMNS,DIRECTION_COLUMNS
    from combat_action_geometry import features_for_version,SMOOTH_VERSION,CENTERED_VERSION
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pin(Path(__file__).with_name('combat_action_effect.py'));pin(Path(__file__).with_name('combat_action_geometry.py'))
    data=torch.load(data_path,map_location='cuda',weights_only=True)
    assert data['version']=='combat_action_effect_data_v1' and not data['ppo_ready']
    x,target=data['features'],data['targets'];causal=x[:,:185];masks=data['masks']
    assert x.shape==(len(members),192) and target.shape==(len(members),5)
    assert float((inputs(causal,x[:,190:192])-x).abs().max())<1e-6
    original=[];ordered=[]
    for name in lr['datasets']:
        folder=Path(name);s=read(pin(folder/'complete.json'));r=read(pin(folder/'report.json',s['report_sha256']))
        assert not s['ppo_ready'];original.append(torch.load(pin(folder/'dataset.pt',s['dataset_sha256']),map_location='cuda',weights_only=True))
        with gzip.open(pin(folder/'members.jsonl.gz',r['members_sha256']),'rt',encoding='utf-8') as f:ordered.extend(map(json.loads,f))
    key=lambda m:(m['source'],m['episode'],m['seed'],m['shot'],m['frame'],m['index'])
    assert [key(m) for m in members]==[key(m) for m in ordered]
    assert torch.equal(causal,torch.cat([d['features_with_recoil'] for d in original]))
    y=torch.cat([d['labels'] for d in original])
    assert torch.equal(y,torch.tensor([m['positive'] for m in members],dtype=torch.float32,device='cuda'))
    domain=torch.tensor([m['source_dataset']>0 for m in members],dtype=torch.bool,device='cuda')
    group=lambda mask:sorted({(m['episode'],m['seed']) for m,yes in zip(members,mask.tolist()) if yes})
    inner_id=torch.tensor([m['seed']%3 for m in members],dtype=torch.long,device='cuda')
    def normalization(mask):
        mean=x[mask].mean(0);std=x[mask].std(0,unbiased=False);std=torch.where(std<.001,torch.ones_like(std),std)
        return mean,std
    def fit_motor(mask):
        mean,std=normalization(mask);torch.manual_seed(p['forecaster_seed']);net=FactorizedMotor()
        opt=torch.optim.Adam(net.parameters(),lr=p['forecaster_lr'],weight_decay=p['forecaster_weight_decay'])
        values=(x[mask]-mean)/std
        for _ in range(p['forecaster_steps']):
            loss=nn.functional.mse_loss(net(values),target[mask]);assert bool(torch.isfinite(loss))
            opt.zero_grad();loss.backward();opt.step()
        return net,mean,std,opt
    def metric(prob,mask):
        if not bool(mask.any()):return None
        q,t=prob[mask].clamp(1e-6,1-1e-6),y[mask]
        yes,no=q[t==1],q[t==0].sort().values
        auc=float(((torch.searchsorted(no,yes,right=False)+torch.searchsorted(no,yes,right=True)).float()*.5/len(no)).mean()) if len(yes) and len(no) else None
        return dict(shots=int(mask.sum()),positive=int(t.sum()),log_loss=float(nn.functional.binary_cross_entropy(q,t)),brier=float((q-t).square().mean()),auc=auc)
    results={};crossfit={};checkpoints={};start=time.perf_counter()
    for split,field in [('seed_mod4_eq3','seed_holdout'),('held_out_site02','site_holdout')]:
        test=masks[field];train=~test
        assert group(train)==[tuple(g) for g in parent_folds[split]['train']]
        assert group(test)==[tuple(g) for g in parent_folds[split]['held_out']]
        assert set(group(train)).isdisjoint(group(test))
        forecast=torch.empty_like(target);assignment=torch.full((len(x),),-1,dtype=torch.long,device='cuda');inner_records=[]
        for i in range(3):
            consume=train&(inner_id!=i);predict=train&(inner_id==i)
            assert bool(consume.any()) and bool(predict.any()) and not bool((consume&predict).any()) and not bool((consume&test).any())
            assert set(group(consume)).isdisjoint(group(predict))
            if reuse:
                old=reused_groups[split]['inner'][i]
                assert old['train']==[list(g) for g in group(consume)] and old['predict']==[list(g) for g in group(predict)]
                cp=pin(old['checkpoint'],reused_report['checkpoints'][old['checkpoint']])
                state=torch.load(cp,map_location='cuda',weights_only=True)
                assert state['steps']==300 and state['protocol_sha256']==sha(previous_protocol)
                mean,std=normalization(consume)
                assert torch.allclose(mean,state['mean'],atol=1e-6,rtol=1e-6) and torch.allclose(std,state['std'],atol=1e-6,rtol=1e-6)
                net=FactorizedMotor();net.load_state_dict(state['model'],strict=True)
            else:
                net,mean,std,opt=fit_motor(consume)
            with torch.no_grad():forecast[predict]=net((x[predict]-mean)/std)
            assignment[predict]=i
            if not reuse:
                cp=out/f'{split}-inner-{i}-motor.pt'
                torch.save(dict(model=net.state_dict(),mean=mean,std=std,optimizer=opt.state_dict(),steps=p['forecaster_steps'],protocol_sha256=pins[str(path)]),cp)
            checkpoints[str(cp)]=sha(cp)
            inner_records.append(dict(inner=i,train=group(consume),predict=group(predict),train_rows=int(consume.sum()),predict_rows=int(predict.sum()),checkpoint=str(cp)))
            save(out/'progress.json',dict(split=split,stage='crossfit',inner_done=i+1))
            print(split,'inner',i,'complete',flush=True)
        # Outer model was fitted only on this outer TRAIN set in the sealed probe.
        cp=parent/(split+'-'+p['forecaster']+'.pt');pin(cp,pr['checkpoints'][str(cp)])
        checkpoint=torch.load(cp,map_location='cuda',weights_only=True)
        assert checkpoint['hand']==2 and checkpoint['steps']==300 and checkpoint['protocol_sha256']==sha(parent/'protocol.json')
        assert checkpoint['model']['position_columns'].tolist()==list(POSITION_COLUMNS)
        assert checkpoint['model']['direction_columns'].tolist()==list(DIRECTION_COLUMNS)
        mean,std=normalization(train)
        assert torch.allclose(mean,checkpoint['mean'],atol=1e-6,rtol=1e-6) and torch.allclose(std,checkpoint['std'],atol=1e-6,rtol=1e-6)
        net=FactorizedMotor();net.load_state_dict(checkpoint['model'],strict=True)
        with torch.no_grad():forecast[test]=net((x[test]-checkpoint['mean'])/checkpoint['std'])
        assignment[test]=3
        error=float(((forecast[test,:3]-target[test,:3])*32).norm(dim=-1).mean())
        assert abs(error-pr['splits'][split][p['forecaster']]['held_out']['muzzle_error'])<.001
        assert bool((assignment>=0).all()) and bool(torch.isfinite(forecast).all())
        crossfit[split]=dict(inner=inner_records,outer_train=group(train),outer_held_out=group(test),
            outer_checkpoint=str(cp),outer_muzzle_error=error,all_rows_predicted_once=True,standard_ppo_replay=False)
        torch.save(dict(forecast=forecast,assignment=assignment,train=train,held_out=test,protocol_sha256=pins[str(path)]),out/(split+'-crossfit.pt'))
        fixed=torch.zeros_like(target);fixed[:,:3]=x[:,185:188]*(400*.1/32)
        if reuse:
            geometry={arms[0]:features_for_version(causal,CENTERED_VERSION),arms[1]:projected_geometry(causal,fixed,False),
                arms[2]:projected_geometry(causal,torch.zeros_like(fixed),True),arms[3]:projected_geometry(causal,fixed,True),
                arms[4]:projected_geometry(causal,forecast,False),arms[5]:projected_geometry(causal,forecast,True)}
        else:
            geometry={arms[0]:features_for_version(causal,SMOOTH_VERSION),arms[1]:features_for_version(causal,CENTERED_VERSION),
                arms[2]:projected_geometry(causal,fixed),arms[3]:projected_geometry(causal,forecast)}
        strata=dict(train=train,all_held_out=test,legacy_held_out=test&~domain,
            new_intervened_held_out=test&domain&masks['intervened'],
            new_baseline_held_out=test&domain&masks['exploration_eligible']&~masks['intervened'])
        rows={}
        for name in arms:
            features=torch.cat((causal,geometry[name]),-1);values=features[train]
            mean=values.mean(0);std=values.std(0,unbiased=False);std=torch.where(std<.001,torch.ones_like(std),std)
            torch.manual_seed(p['predictor_seed']);predictor=nn.Linear(214,1,device='cuda')
            opt=torch.optim.Adam(predictor.parameters(),lr=p['predictor_lr'],weight_decay=p['predictor_weight_decay'])
            normalized=(features-mean)/std
            for _ in range(p['predictor_steps']):
                loss=nn.functional.binary_cross_entropy_with_logits(predictor(normalized[train])[:,0],y[train])
                assert bool(torch.isfinite(loss));opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():prob=predictor(normalized)[:,0].sigmoid()
            rows[name]={k:metric(prob,m) for k,m in strata.items()}
            cp=out/f'{split}-{name}-shot.pt'
            torch.save(dict(model=predictor.state_dict(),optimizer=opt.state_dict(),mean=mean,std=std,arm=name,protocol_sha256=pins[str(path)]),cp)
            checkpoints[str(cp)]=sha(cp)
            print(split,name,'complete',flush=True)
        results[split]=rows
    torch.cuda.synchronize();checks={}
    for split,rows in results.items():
        learned=rows[arms[4 if reuse else 3]];centered=rows[arms[0 if reuse else 1]];fixed=rows[arms[1 if reuse else 2]]
        checks[split]=dict(new_intervened_better_than_centered=learned['new_intervened_held_out']['log_loss']<centered['new_intervened_held_out']['log_loss'],
            new_intervened_better_than_fixed=learned['new_intervened_held_out']['log_loss']<fixed['new_intervened_held_out']['log_loss'],
            legacy_regression_within_budget=learned['legacy_held_out']['log_loss']<=centered['legacy_held_out']['log_loss']+.02)
    accepted=all(all(c.values()) for c in checks.values())
    save(out/'crossfit-groups.json',crossfit)
    for file,digest in pins.items():assert sha(Path(file))==digest,file
    save(out/'report.json',dict(state='complete',version=p['version'],device='cuda',shots=len(members),source_cases=lr['source_cases'],
        splits=results,gate_checks=checks,development_gate_passed=accepted,inputs=pins,checkpoints=checkpoints,
        crossfit_groups_sha256=sha(out/'crossfit-groups.json'),script_sha256=sha(__file__),seconds=time.perf_counter()-start,
        training_performed=True,new_forecaster_fits=0 if reuse else 6,policy_training=False,runtime_integrated=False,promotion=False,
        scope='Cross-fitted TRAIN-only motor-derived geometry. Labels consumed only by permitted training folds; held-out shots are development, not native policy quality. No actor fitting, neural CPU/Go replay or model promotion.'))
    save(out/'complete.json',dict(state='complete',report_sha256=sha(out/'report.json'),development_gate_passed=accepted,ppo_ready=False))
    print(json.dumps(dict(development_gate_passed=accepted,gate_checks=checks,seconds=time.perf_counter()-start)))


if __name__=='__main__':main()

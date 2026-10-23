"""Refit the accepted motor/shot predictor on TRAIN with full-data cross-fitting."""
import argparse
import gzip
import json
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('probe','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();assert not args.out.exists();pins={}
    def pin(file,expected=None):
        file=Path(file).resolve();digest=sha(file);assert expected is None or digest==expected,str(file)
        pins[str(file)]=digest;return file
    seal=read(pin(args.probe/'complete.json'));probe=read(pin(args.probe/'report.json',seal['report_sha256']))
    assert probe['version']=='combat_action_effect_shot_probe_v2' and probe['development_gate_passed']
    assert all(all(v.values()) for v in probe['gate_checks'].values())
    p=read(pin(args.probe/'protocol.json'))
    assert (p['forecaster_steps'],p['forecaster_lr'],p['forecaster_weight_decay'],p['forecaster_seed'])==(300,.001,.0001,20261023)
    assert (p['predictor_steps'],p['predictor_lr'],p['predictor_weight_decay'],p['predictor_seed'])==(200,.001,.0001,20261022)
    parent=Path(p['forecaster_probe']);ps=read(pin(parent/'complete.json'))
    pr=read(pin(parent/'report.json',ps['report_sha256']));assert p['forecaster'] in pr['eligible_models'] and pr['device']=='cuda'
    data_path=pin(parent/'dataset.pt',ps['dataset_sha256'])
    labels=Path(p['labels']);ls=read(pin(labels/'complete.json'));lr=read(pin(labels/'report.json',ls['report_sha256']))
    assert lr['state']=='complete' and lr['hand']==2
    with gzip.open(pin(labels/'members.jsonl.gz',lr['members_sha256']),'rt',encoding='utf-8') as f:members=[json.loads(line) for line in f]
    from ppo_combat import torch,nn,training_devices
    from combat_action_effect import FactorizedMotor,projected_geometry,POSITION_COLUMNS,DIRECTION_COLUMNS
    from combat_action_geometry import names_for_version,CENTERED_VERSION
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for name in ('combat_action_effect.py','combat_action_geometry.py'):pin(Path(__file__).with_name(name))
    d=torch.load(data_path,map_location='cuda',weights_only=True);x,target=d['features'],d['targets'];causal=x[:,:185]
    assert x.shape==(len(members),192) and target.shape==(len(members),5)
    sets=[];sources=[]
    for name in lr['datasets']:
        folder=Path(name);s=read(pin(folder/'complete.json'));r=read(pin(folder/'report.json',s['report_sha256']))
        assert not s['ppo_ready'];file=pin(folder/'dataset.pt',s['dataset_sha256'])
        sets.append(torch.load(file,map_location='cuda',weights_only=True))
        sources.append(dict(path=str(file),sha256=s['dataset_sha256'],shots=r['shots']))
    assert torch.equal(causal,torch.cat([s['features_with_recoil'] for s in sets]))
    y=torch.cat([s['labels'] for s in sets])
    assert torch.equal(y,torch.tensor([m['positive'] for m in members],dtype=torch.float32,device='cuda'))
    args.out.mkdir(parents=True)
    save(args.out/'training-datasets.json',dict(version='combat_motor_shot_training_manifest_v1',purpose='TRAIN_only',
        sources=sources,action_effect_labels=str(data_path),action_effect_labels_sha256=sha(data_path),
        development_gate_report=str((args.probe/'report.json').resolve()),development_gate_sha256=sha(args.probe/'report.json'),
        hand=2,forecaster='factorized_mlp76x32_21x16',classifier_training='3foldwhole_episode_seed_mod3_crossfitted_motor_geometry_current_observed_target',
        train_shots=len(members),source_cases=lr['source_cases']))
    groups=lambda mask:sorted({(m['episode'],m['seed']) for m,yes in zip(members,mask.tolist()) if yes})
    def fit_motor(mask):
        mean=x[mask].mean(0);std=x[mask].std(0,unbiased=False);std=torch.where(std<.001,torch.ones_like(std),std)
        torch.manual_seed(p['forecaster_seed']);net=FactorizedMotor()
        opt=torch.optim.Adam(net.parameters(),lr=p['forecaster_lr'],weight_decay=p['forecaster_weight_decay'])
        for _ in range(p['forecaster_steps']):
            loss=nn.functional.mse_loss(net((x[mask]-mean)/std),target[mask]);assert bool(torch.isfinite(loss))
            opt.zero_grad();loss.backward();opt.step()
        return net,mean,std,opt
    ids=torch.tensor([m['seed']%3 for m in members],dtype=torch.long,device='cuda')
    forecast=torch.empty_like(target);assigned=torch.zeros(len(x),dtype=torch.bool,device='cuda');folds=[]
    for i in range(3):
        train,predict=ids!=i,ids==i
        assert bool(train.any()) and bool(predict.any()) and set(groups(train)).isdisjoint(groups(predict))
        net,mean,std,opt=fit_motor(train)
        with torch.no_grad():forecast[predict]=net((x[predict]-mean)/std)
        assert not bool(assigned[predict].any());assigned[predict]=True
        cp=args.out/f'inner-{i}-motor.pt'
        torch.save(dict(model=net.state_dict(),mean=mean,std=std,optimizer=opt.state_dict(),steps=300),cp)
        folds.append(dict(inner=i,train=groups(train),predict=groups(predict),checkpoint=str(cp),sha256=sha(cp)))
        print('Full TRAIN crossfit',i,'complete',flush=True)
    assert bool(assigned.all()) and bool(torch.isfinite(forecast).all())
    geometry=projected_geometry(causal,forecast,False)
    values=torch.cat((causal,geometry),-1);mean=values.mean(0);std=values.std(0,unbiased=False);std=torch.where(std<.001,torch.ones_like(std),std)
    torch.manual_seed(p['predictor_seed']);net=nn.Linear(214,1,device='cuda')
    opt=torch.optim.Adam(net.parameters(),lr=p['predictor_lr'],weight_decay=p['predictor_weight_decay'])
    for _ in range(p['predictor_steps']):
        loss=nn.functional.binary_cross_entropy_with_logits(net((values-mean)/std)[:,0],y);assert bool(torch.isfinite(loss))
        opt.zero_grad();loss.backward();opt.step()
    # Deployment forecaster consumes all TRAIN only after cross-fitted classifier
    # features are frozen. Its own fitted targets never generate classifier TRAIN.
    full=torch.ones(len(x),dtype=torch.bool,device='cuda')
    motor,motor_mean,motor_std,motor_opt=fit_motor(full)
    layers=lambda branch:[dict(weight=branch[i].weight.detach().tolist(),bias=branch[i].bias.detach().tolist()) for i in (0,2)]
    motor_file=dict(version='combat_action_effect_motor_v1',hand=2,input_width=192,
        position_columns=list(POSITION_COLUMNS),direction_columns=list(DIRECTION_COLUMNS),mean=motor_mean.tolist(),std=motor_std.tolist(),
        position_layers=layers(motor.position),direction_layers=layers(motor.direction),
        target_extrapolation_seconds=0.,displacement_scale=32.,angle_scale_degrees=5.)
    predictor=dict(kind='combat_geometry_shot_predictor_v1',geometry_version='combat_predicted_action_geometry_v1',
        input_width=214,causal_width=185,geometry_names=list(names_for_version(CENTERED_VERSION)),
        mean=mean.tolist(),std=std.tolist(),weight=net.weight[0].detach().tolist(),bias=float(net.bias[0].detach()),
        motor=motor_file,trained_weapon='machinegun',dataset_sha256=sha(args.out/'training-datasets.json'),
        scope='P(actual MG living monster damage | observation/action). Centered predicted own pose/recoil; current observed target. TRAIN classifier uses motor crossfit. No native pose inputs, fixed target lead, firing mask or policy improvement claim.')
    save(args.out/'predictor.json',predictor);loaded=read(args.out/'predictor.json')
    decoded=FactorizedMotor()
    with torch.no_grad():
        for branch,name in ((decoded.position,'position_layers'),(decoded.direction,'direction_layers')):
            for i,layer in zip((0,2),loaded['motor'][name]):
                branch[i].weight.copy_(torch.tensor(layer['weight'],device='cuda'));branch[i].bias.copy_(torch.tensor(layer['bias'],device='cuda'))
        actual=motor((x-motor_mean)/motor_std)
        restored=decoded((x-torch.tensor(loaded['motor']['mean'],device='cuda'))/torch.tensor(loaded['motor']['std'],device='cuda'))
        motor_error=float((actual-restored).abs().max());assert motor_error<1e-5
        forward_values=torch.cat((causal,projected_geometry(causal,restored,False)),-1)
        restored_score=nn.functional.linear((forward_values-torch.tensor(loaded['mean'],device='cuda'))/torch.tensor(loaded['std'],device='cuda'),
            torch.tensor(loaded['weight'],device='cuda')[None],torch.tensor([loaded['bias']],device='cuda'))[:,0]
        source_score=net((torch.cat((causal,projected_geometry(causal,actual,False)),-1)-mean)/std)[:,0]
        score_error=float((restored_score-source_score).abs().max());assert score_error<1e-5
        loss=float(nn.functional.binary_cross_entropy_with_logits(net((values-mean)/std)[:,0],y))
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),mean=mean,std=std,steps=200,
                    training_manifest_sha256=sha(args.out/'training-datasets.json')),args.out/'checkpoint.pt')
    torch.save(dict(model=motor.state_dict(),optimizer=motor_opt.state_dict(),mean=motor_mean,std=motor_std,steps=300,
                    training_manifest_sha256=sha(args.out/'training-datasets.json')),args.out/'motor-checkpoint.pt')
    save(args.out/'folds.json',folds)
    for path,digest in pins.items():assert sha(Path(path))==digest,path
    report=dict(state='complete',version='combat_action_effect_predictor_export_v1',device='cuda',
        train_shots=len(members),source_cases=lr['source_cases'],motor_parameters=2949,shot_parameters=215,
        crossfitted_train_logloss=loss,export_max_motor_error=motor_error,export_max_logit_error=score_error,
        predictor_sha256=sha(args.out/'predictor.json'),checkpoint_sha256=sha(args.out/'checkpoint.pt'),motor_checkpoint_sha256=sha(args.out/'motor-checkpoint.pt'),
        inputs=pins,folds_sha256=sha(args.out/'folds.json'),script_sha256=sha(__file__),training_performed=True,
        policy_training=False,runtime_integrated=False,promotion=False,
        scope='All-TRAIN refit after separate positive development gate. Serialization checked on CUDA, no Go neural scoring or native evaluation. Classifier trained on out-of-fold motor forecasts; deployment motor fit uses all TRAIN. Actor/reward/default unchanged.')
    save(args.out/'report.json',report)
    save(args.out/'complete.json',dict(state='complete',report_sha256=sha(args.out/'report.json'),
        predictor_sha256=report['predictor_sha256'],checkpoint_sha256=report['checkpoint_sha256'],motor_checkpoint_sha256=report['motor_checkpoint_sha256'],
        runtime_integrated=False,promotion=False))
    print(json.dumps({k:report[k] for k in ('train_shots','motor_parameters','shot_parameters','export_max_motor_error','export_max_logit_error')}))


if __name__=='__main__':main()

"""Grouped TRAIN motor-effect regression; all tensor/NN computation on CUDA."""
import argparse
import gzip
import json
import math
import time
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',type=Path,required=True)
    args=ap.parse_args();protocol=args.protocol.resolve();out=protocol.parent;p=read(protocol)
    assert p['version'] in ('combat_action_effect_probe_v1','combat_action_effect_probe_v2','combat_action_effect_probe_v3')
    assert (p['steps'],p['lr'],p['weight_decay'],p['seed'])==(300,.001,.0001,20261023)
    structured=p['version']=='combat_action_effect_probe_v3'
    expected_models=['factorized_linear76_21','factorized_mlp76x32_21x16'] if structured else ['linear192','mlp192x32x5']
    assert p['models']==expected_models and p['hand']==2
    assert not any(p[k] for k in ('policy_training','runtime_integrated','promotion'))
    assert not (out/'report.json').exists() and not any(out.glob('*.pt'))
    pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path)
        assert expected is None or digest==expected,str(path)
        pins[str(path)]=digest;return path
    pin(protocol)
    diagnostic=Path(p['diagnostic'])
    seal=read(pin(diagnostic/'complete.json'));dr=read(pin(diagnostic/'report.json',seal['report_sha256']))
    assert dr['state']=='complete' and dr['hand']==2 and not dr['neural_execution']
    with gzip.open(pin(diagnostic/'members.jsonl.gz',dr['members_sha256']),'rt',encoding='utf-8') as f:
        records=[json.loads(line) for line in f]
    with pin(diagnostic/'requests.jsonl',dr['request_sha256']).open(encoding='utf-8') as f:
        requests=[json.loads(line) for line in f]
    assert len(records)==len(requests)>0
    members=[];folders=[];source_cases=0
    for name in p['datasets']:
        folder=Path(name);folders.append(folder)
        seal=read(pin(folder/'complete.json'));report=read(pin(folder/'report.json',seal['report_sha256']))
        assert seal['state']==report['state']=='complete' and not seal['ppo_ready']
        pin(folder/'dataset.pt',seal['dataset_sha256']);source_cases+=report['source_cases']
        with gzip.open(pin(folder/'members.jsonl.gz',report['members_sha256']),'rt',encoding='utf-8') as f:
            ms=[json.loads(line) for line in f]
        assert len(ms)==report['shots'];members.extend(ms)
    keys=lambda rows:[(r['source'],r['episode'],r['seed'],r['shot'],r['frame'],r['index']) for r in rows]
    assert keys(records)==keys(members) and len(set(keys(records)))==len(records)
    from ppo_combat import torch,nn,training_devices
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sets=[torch.load(folder/'dataset.pt',map_location='cuda',weights_only=True) for folder in folders]
    causal=torch.cat([s['features_with_recoil'] for s in sets])
    assert causal.shape==(len(records),185) and causal.is_cuda
    if 'pure_go_causal_sha256' in dr:
        with pin(diagnostic/'pure-go-causal.jsonl',dr['pure_go_causal_sha256']).open(encoding='utf-8') as f:
            packed=[json.loads(line) for line in f]
        original=torch.tensor([q['causal'] for q in packed],dtype=torch.float64,device='cuda')
        causal_error=float((original.float()-causal).abs().max())
        assert causal_error<.00002,causal_error
        from combat_action_geometry import features_for_version,CENTERED_VERSION
        expected=features_for_version(original,CENTERED_VERSION)
        geometry=torch.tensor([q['geometry'] for q in packed],dtype=torch.float64,device='cuda')
        geometry_error=float((expected-geometry).abs().max())
        assert geometry_error<1e-7,geometry_error
        save(out/'causal-parity.json',dict(state='complete',shots=len(packed),device='cuda',
            original_causal_max_error=causal_error,centered_geometry_fp64_max_error=geometry_error,
            go_neural_execution=False,scope='Pure Go observed feature packing versus original CUDA causal185; pure centered geometry versus CUDA FP64. No Go model evaluation.'))
    tensor=lambda values:torch.tensor(values,dtype=torch.float32,device='cuda')
    observations=[q['observation'] for q in requests]
    pos=tensor([o['position'] for o in observations]);velocity=tensor([o['velocity'] for o in observations])
    assert torch.allclose(causal[:,10:13]*400,velocity,atol=.0001,rtol=1e-6)
    assert torch.equal(causal[:,3]>0,torch.tensor([o['ducked'] for o in observations],device='cuda'))
    yaw=torch.atan2(causal[:,6],causal[:,7]);cy,sy=yaw.cos(),yaw.sin()
    local=lambda v:torch.stack((cy*v[:,0]+sy*v[:,1],-sy*v[:,0]+cy*v[:,1],v[:,2]),-1)
    base=pos.clone();base[:,2]+=torch.where(causal[:,3]>0,-10.,14.)
    native_muzzle=tensor([r['native_muzzle_label'] for r in records])
    native_view=tensor([r['native_view_label'] for r in records])
    native_recoil=tensor([r['native_recoil_label'] for r in records])
    native_aim=tensor([r['native_aim_label'] for r in records])
    kick=torch.where(causal[:,181:182]>0,causal[:,182:185]*32,0.)
    base_angles=torch.stack((torch.atan2(causal[:,8],causal[:,9])*180/math.pi+causal[:,165]*180+kick[:,0],
                             yaw*180/math.pi+causal[:,164]*180+kick[:,1]),-1)
    true_angles=(native_view+native_recoil)[:,:2]
    wrap=lambda a:torch.atan2((a*math.pi/180).sin(),(a*math.pi/180).cos())*180/math.pi
    direction=lambda a:torch.stack(((a[:,0]*math.pi/180).cos()*(a[:,1]*math.pi/180).cos(),
        (a[:,0]*math.pi/180).cos()*(a[:,1]*math.pi/180).sin(),-(a[:,0]*math.pi/180).sin()),-1)
    assert float((direction(true_angles)-native_aim).abs().max())<6e-5
    # Native targets are separated from the causal matrix by construction.
    target=torch.cat((local(native_muzzle-base)/32,wrap(true_angles-base_angles)/5),-1)
    own_local=local(velocity)/400
    delta=causal[:,164]*math.pi;f,s=causal[:,162],causal[:,163]
    requested=torch.stack((f*delta.cos()+s*delta.sin(),f*delta.sin()-s*delta.cos()),-1)
    timing=tensor([[o['observation_age_ms']/300,o['previous_applied_command']['Msec']/100] for o in observations])
    x=torch.cat((causal,own_local,requested,timing),-1)
    assert x.shape==(len(records),192) and bool(torch.isfinite(x).all()) and bool(torch.isfinite(target).all())
    position_columns=list(range(6))+list(range(13,73))+list(range(185,190))+list(range(166,169))+list(range(190,192))
    direction_columns=list(range(6))+list(range(8,10))+list(range(15,17))+list(range(181,188))+[165,161]+list(range(190,192))
    assert len(position_columns)==76 and len(direction_columns)==21
    # Separate heads cannot use enemy positions, target identity or site cues to
    # predict recoil. The action policy itself remains fully unconstrained.
    class Factorized(nn.Module):
        def __init__(self,hidden):
            super().__init__()
            self.register_buffer('position_columns',torch.tensor(position_columns,dtype=torch.long,device='cuda'))
            self.register_buffer('direction_columns',torch.tensor(direction_columns,dtype=torch.long,device='cuda'))
            self.position=nn.Sequential(nn.Linear(76,32,device='cuda'),nn.ReLU(),nn.Linear(32,3,device='cuda')) if hidden else nn.Linear(76,3,device='cuda')
            self.direction=nn.Sequential(nn.Linear(21,16,device='cuda'),nn.ReLU(),nn.Linear(16,2,device='cuda')) if hidden else nn.Linear(21,2,device='cuda')
            with torch.no_grad():
                for module in (self.position,self.direction):
                    final=module[-1] if hidden else module
                    final.weight.zero_();final.bias.zero_()
        def forward(self,values):
            return torch.cat((self.position(values[:,self.position_columns]),self.direction(values[:,self.direction_columns])),-1)
    masks={k:torch.cat([s.get(k,torch.zeros_like(s['seed_holdout'])) for s in sets]) for k in ('seed_holdout','site_holdout','intervened','exploration_eligible')}
    selected=causal[:,151]==0
    # Fixed causal baselines: no future velocity/command time is substituted.
    zero=torch.zeros_like(target)
    constant_velocity=zero.clone();constant_velocity[:,:3]=own_local*400*.1/32
    ay,ap,roll=base_angles[:,1]*math.pi/180,base_angles[:,0]*math.pi/180,kick[:,2]*math.pi/180
    right=torch.stack((-roll.sin()*ap.sin()*ay.cos()+roll.cos()*ay.sin(),
        -roll.sin()*ap.sin()*ay.sin()-roll.cos()*ay.cos(),-roll.sin()*ap.cos()),-1)
    historical_right=zero.clone();historical_right[:,:3]=local(8*right)/32
    def metrics(pred,mask):
        if not bool(mask.any()):return None
        diff=(pred-target)[mask]
        angles=base_angles+pred[:,3:]*5
        # atan2(cross,dot) avoids acos numerical noise near a perfect direction.
        a,b=direction(angles)[mask],native_aim[mask]
        angular=torch.atan2(torch.linalg.vector_norm(torch.linalg.cross(a,b),dim=-1),(a*b).sum(-1))*180/math.pi
        return dict(shots=int(mask.sum()),muzzle_error=float((diff[:,:3]*32).norm(dim=-1).mean()),
            horizontal_muzzle_error=float((diff[:,:2]*32).norm(dim=-1).mean()),
            vertical_muzzle_mae=float((diff[:,2]*32).abs().mean()),direction_error_degrees=float(angular.mean()),
            pitch_mae_degrees=float((diff[:,3]*5).abs().mean()),yaw_mae_degrees=float((diff[:,4]*5).abs().mean()),
            normalized_mse=float(diff.square().mean()))
    all_mask=torch.ones(len(x),dtype=torch.bool,device='cuda')
    baselines={'historical_right_pose':historical_right,'centered_current_pose':zero,'constant_velocity_100ms':constant_velocity}
    save(out/'decomposition.json',dict(all={name:metrics(pred,all_mask) for name,pred in baselines.items()},
        known_bbox={name:metrics(pred,selected) for name,pred in baselines.items()},
        observed_age_ms_counts=dict(__import__('collections').Counter(o['observation_age_ms'] for o in observations)),
        note='All label geometry here is computed on CUDA. game frame and server-prefix frame clocks are not converted to latency.'))
    torch.save(dict(version='combat_action_effect_data_v1',purpose='predictive_train_only',features=x,targets=target,
                    masks=masks,selected_target=selected,base_angles=base_angles,native_aim=native_aim,
                    source_cases=source_cases,hand=2,ppo_ready=False),out/'dataset.pt')
    folds={};results={};checkpoints={};start=time.perf_counter()
    for label,field in [('seed_mod4_eq3','seed_holdout'),('held_out_site02','site_holdout')]:
        test=masks[field];train=~test
        train_groups={(r['episode'],r['seed']) for r in records if not r[field]}
        test_groups={(r['episode'],r['seed']) for r in records if r[field]}
        assert train_groups and test_groups and train_groups.isdisjoint(test_groups)
        folds[label]=dict(train=sorted(train_groups),held_out=sorted(test_groups))
        strata=dict(train=train,held_out=test,intervened_held_out=test&masks['intervened'],
                    baseline_held_out=test&masks['exploration_eligible']&~masks['intervened'])
        comparison={name:{k:metrics(pred,m) for k,m in strata.items()} for name,pred in baselines.items()}
        mean=x[train].mean(0);std=x[train].std(0,unbiased=False);std=torch.where(std<.001,torch.ones_like(std),std)
        normalized=(x-mean)/std
        for name in p['models']:
            torch.manual_seed(p['seed'])
            if structured:
                net=Factorized(name=='factorized_mlp76x32_21x16')
            else:
                net=nn.Linear(192,5,device='cuda') if name=='linear192' else nn.Sequential(nn.Linear(192,32,device='cuda'),nn.ReLU(),nn.Linear(32,5,device='cuda'))
                final=net if name=='linear192' else net[-1]
                with torch.no_grad():final.weight.zero_();final.bias.zero_()
            opt=torch.optim.Adam(net.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
            for _ in range(p['steps']):
                loss=nn.functional.mse_loss(net(normalized[train]),target[train])
                assert bool(torch.isfinite(loss));opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():pred=net(normalized)
            comparison[name]={k:metrics(pred,m) for k,m in strata.items()}
            cp=out/(label+'-'+name+'.pt')
            torch.save(dict(version='combat_action_effect_checkpoint_v1',model=net.state_dict(),optimizer=opt.state_dict(),
                mean=mean,std=std,protocol_sha256=sha(protocol),steps=p['steps'],hand=2),cp)
            checkpoints[str(cp)]=sha(cp)
            comparison[name]['parameters']=sum(v.numel() for v in net.parameters())
            print(label,name,'complete',flush=True)
        results[label]=comparison
    torch.cuda.synchronize()
    eligible=[]
    for name in p['models']:
        if all(result[name]['held_out']['muzzle_error']<result['constant_velocity_100ms']['held_out']['muzzle_error'] and
               result[name]['held_out']['direction_error_degrees']<result['centered_current_pose']['held_out']['direction_error_degrees']
               for result in results.values()):eligible.append(name)
    save(out/'folds.json',folds)
    for path,digest in pins.items():assert sha(Path(path))==digest,path
    save(out/'report.json',dict(state='complete',version=p['version'],device='cuda',shots=len(records),
        source_cases=source_cases,battles_with_shots=len({(r['episode'],r['seed']) for r in records}),
        splits=results,eligible_models=eligible,inputs=pins,checkpoints=checkpoints,seconds=time.perf_counter()-start,
        dataset_sha256=sha(out/'dataset.pt'),script_sha256=sha(__file__),folds_sha256=sha(out/'folds.json'),
        training_performed=True,policy_training=False,runtime_integrated=False,promotion=False,
        scope='Fixed-budget supervised action-effect development on grouped randomized TRAIN. Native muzzle/recoil/view only targets, no future inputs or attack policy. No battle improvement or counterfactual optimality established.'))
    save(out/'complete.json',dict(state='complete',report_sha256=sha(out/'report.json'),dataset_sha256=sha(out/'dataset.pt'),ppo_ready=False))
    print(json.dumps(dict(eligible_models=eligible,seconds=time.perf_counter()-start)))


if __name__=='__main__':main()

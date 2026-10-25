"""CUDA-only route BC on executed recontact, with frozen-yaw coordinate admission."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from ppo_combat import torch,nn,training_devices
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model
from combat_memory_route_movement import inputs,export
from evaluate_registered_combat_models import read,save,sha

VERSION='combat_executed_recontact_bc_v1'
CHECKPOINT='combat_recontact_bc_checkpoint_v1'


def compatible(source,model):
    # All actor/action behavior, including std and target/aim samplers, must
    # match; only movement branches and critic are permitted to differ.
    ignored={'motor_residual','memory_route_movement','value','attention'}
    assert source.keys()==model.keys()
    assert all(source[k]==model[k] for k in model if k not in ignored), 'Teacher frozen behavior differs'
    assert source['attention']['actor']==model['attention']['actor']


def load(report,split,model,actor):
    episodes=[e for e in report['episodes'] if e['split']==split];assert episodes
    arrays=[];positions=[];targets=[]
    for i,e in enumerate(episodes):
        for name in ('pack','labels','windows'):assert sha(e[name])==e[name+'_sha256']
        assert sha(e['source_model'])==e['source_model_sha256']
        source=read(e['source_model']);compatible(source,model)
        data=np.load(e['pack'],allow_pickle=False);x=data['features'];idx=data['indices']
        labels=[json.loads(s) for s in Path(e['labels']).open()]
        assert x.ndim==2 and x.shape[1]==1554 and len(idx)==len(labels)==e['rows']
        assert np.isfinite(x).all() and np.isfinite(data['local_movement']).all()
        assert np.abs(data['local_movement']).max()<=1 and np.array_equal(data['frames'][idx],[r['frame'] for r in labels])
        assert all(r['source_sample'].get('target',0)==0 and not r['source_sample'].get('aim_candidate') for r in labels)
        # For target=0, both spatial and sampled-motion aim corrections are
        # exactly zero. Compare full causal raw yaw means on CUDA as well.
        teacher=branch(base_model(source,allow_shared=True),'actor').requires_grad_(False)
        xx=torch.tensor(x,device='cuda')[None,:]
        with torch.no_grad():
            before=actor(xx)[0][0];raw=teacher(xx)[0][0]
            assert torch.equal(before[:,2:],raw[:,2:]), 'Frozen behavior differs on teacher histories'
            _,active=inputs(xx[0],before)
            assert bool(active[torch.tensor(idx,device='cuda')].all())
        for j,label in enumerate(labels):
            yaw=float(label['source_sample']['latent'][2])
            delta=180*np.tanh(yaw)
            assert abs(delta-label['applied']['yaw_delta_degrees'])<.01, 'Teacher yaw was intervened; local axes invalid'
            assert label['command']['Forward']/400==float(data['local_movement'][j,0]) or abs(label['command']['Forward']/400-float(data['local_movement'][j,0]))<1e-7
        arrays.append(x);positions.extend((i,int(j)) for j in idx);targets.extend(data['local_movement'].tolist())
        del teacher,xx
    features=torch.zeros(len(arrays),max(map(len,arrays)),1554,device='cuda')
    for i,x in enumerate(arrays):features[i,:len(x)]=torch.tensor(x,device='cuda')
    return features,torch.tensor(positions,device='cuda',dtype=torch.long),torch.tensor(targets,device='cuda')


def chosen(actor,data):
    x,pos,_=data;return actor(x)[0][pos[:,0],pos[:,1]]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True,type=Path);ap.add_argument('--data',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path);ap.add_argument('--steps',type=int,default=150)
    ap.add_argument('--resume',type=Path)
    a=ap.parse_args();assert not a.out.exists() and 1<=a.steps<=1000;training_devices()
    torch.set_default_device('cuda');torch.set_num_threads(2);torch.manual_seed(20261025)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    report=read(a.data);assert report['version']=='combat_executed_recontact_demonstrations_v1' and report['state']=='passed' and not report['ppo_eligible']
    for p,h in report['inputs'].items():assert sha(p)==h
    assert report['counts']['train']['worlds']>=4 and report['counts']['train']['rows']>=32
    assert report['counts']['diagnostic']['worlds']>=1
    model=read(a.model);model_hash=sha(a.model)
    assert model['memory_route_movement']['version']=='combat_memory_route_movement_v1', 'Yaw v2 requires its own trainer'
    actor=branch(base_model(model,allow_shared=True),'actor')
    actor.requires_grad_(False);actor.memory_route_branch.requires_grad_(True)
    parameters=list(actor.memory_route_branch.parameters());assert all(p.is_cuda for p in actor.parameters())
    frozen={k:v.detach().clone() for k,v in actor.state_dict().items() if not k.startswith('memory_route_branch.')}
    train,diagnostic=(load(report,s,model,actor) for s in ('train','diagnostic'))
    optimizer=torch.optim.Adam(parameters,lr=.0003);assert not optimizer.state
    config=dict(version=VERSION,seed=20261025,learning_rate=.0003,source_retention=.02,latent_clip=.95,steps=a.steps)
    completed=0;parent_hash=model_hash
    with torch.no_grad():anchor=chosen(actor,train).detach().clone();before_diag=chosen(actor,diagnostic).detach().clone()
    if a.resume:
        cp=torch.load(a.resume,map_location='cuda',weights_only=True)
        assert cp['version']==CHECKPOINT and cp['checkpoint_training_scope']==VERSION
        assert cp['weights_sha256']==model_hash and cp['data_sha256']==sha(a.data)
        assert all(cp['config'][k]==v for k,v in config.items() if k!='steps')
        completed=cp['completed_steps'];assert completed<a.steps
        actor.memory_route_branch.load_state_dict(cp['route']);optimizer.load_state_dict(copy.deepcopy(cp['optimizer']))
        anchor=cp['anchor'];parent_hash=cp['parent_weights_sha256']
        torch.set_rng_state(cp['cpu_rng'].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in cp['cuda_rng']])
    target=torch.atanh(train[2].clamp(-.95,.95))
    def metrics(data):
        raw=chosen(actor,data)
        return dict(command_mae=float((raw[:,:2].tanh()-data[2]).abs().mean()),
                    latent_mse=float((raw[:,:2]-torch.atanh(data[2].clamp(-.95,.95))).square().mean()))
    with torch.no_grad():before=dict(train=metrics(train),diagnostic=metrics(diagnostic))
    losses=[]
    for _ in range(completed,a.steps):
        raw=chosen(actor,train)
        loss=(raw[:,:2]-target).square().mean()+.02*(raw[:,:2]-anchor[:,:2]).square().mean()
        assert torch.isfinite(loss);optimizer.zero_grad();loss.backward()
        norm=nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(norm);optimizer.step();losses.append(float(loss.detach()))
    assert all(torch.equal(actor.state_dict()[k],v) for k,v in frozen.items())
    with torch.no_grad():
        assert torch.equal(chosen(actor,train)[:,2:],anchor[:,2:])
        assert torch.equal(chosen(actor,diagnostic)[:,2:],before_diag[:,2:])
        after=dict(train=metrics(train),diagnostic=metrics(diagnostic))
    result=copy.deepcopy(model);export(result,actor.memory_route_branch)
    assert all(result[k]==v for k,v in model.items() if k!='memory_route_movement')
    assert result['memory_route_movement']!=model['memory_route_movement']
    assert all(sha(p)==h for p,h in report['inputs'].items())
    a.out.mkdir(parents=True,exist_ok=False);save(a.out/'weights.json',result);save(a.out/'config.json',config)
    cp=dict(version=CHECKPOINT,checkpoint_training_scope=VERSION,weights_sha256=sha(a.out/'weights.json'),
        parent_weights_sha256=parent_hash,data_sha256=sha(a.data),config=config,completed_steps=a.steps,
        route=actor.memory_route_branch.state_dict(),optimizer=optimizer.state_dict(),anchor=anchor,
        cuda_rng=torch.cuda.get_rng_state_all(),cpu_rng=torch.get_rng_state())
    torch.save(cp,a.out/'checkpoint.pt')
    save(a.out/'report.json',dict(version=VERSION,state='passed',device=torch.cuda.get_device_name(),training_performed=True,
        before=before,after=after,train_rows=len(train[2]),diagnostic_rows=len(diagnostic[2]),
        trainable_parameters=sum(p.numel() for p in parameters),frozen_actor_verified=True,critic_std_motor_pose_unchanged=True,
        coordinate_admission='Identical frozen behavior, target0, no aim candidates, rawyaw exact on causal GPU histories and unchanged actual teacher yaw',
        weights_sha256=sha(a.out/'weights.json'),checkpoint_sha256=sha(a.out/'checkpoint.pt'),
        inputs={str(a.model.resolve()):model_hash,str(a.data.resolve()):sha(a.data)},completed_steps=a.steps,resumed_steps=completed,
        optimizer_initialization='resumed_recontact_bc' if a.resume else 'fresh_recontact_bc_not_ppo',
        ppo_data_mixed=False,promotion=False,scope='Narrow executed recontact route-only pilot; warehouse/pose support limited, no optimality or quality claim.'))
    print(json.dumps(dict(before=before,after=after)),flush=True)


if __name__=='__main__':main()

"""CUDA-only movement factor BC on explicit native rule demonstrations.

Fresh typed optimizer, no fabricated PPO resume. Conditional aim/fire weights,
base actor/std and critic remain frozen; changed trajectories may change aim.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from ppo_combat import torch,nn,training_devices
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model
from combat_motor_residual import export as export_motor
from combat_memory_route_movement import export as export_route
from evaluate_registered_combat_models import read,save,sha

VERSION='combat_movement_demonstration_bc_v1'
PREFIXES=('motor_branch.','memory_route_branch.')


def load_data(report,split):
    episodes=[e for e in report['episodes'] if e['split']==split]
    arrays=[];locations=[];targets=[]
    for b,e in enumerate(episodes):
        assert sha(e['pack'])==e['pack_sha256']
        data=np.load(e['pack'],allow_pickle=False)
        x=data['features'];idx=data['indices'];y=data['movement']
        assert x.ndim==2 and x.shape[1]==1554 and idx.shape==(len(y),) and y.shape[1]==2
        assert np.isfinite(x).all() and np.isfinite(y).all() and (np.abs(y)<=1).all()
        arrays.append(x);locations.extend((b,int(i)) for i in idx);targets.extend(y.tolist())
    assert episodes
    x=torch.zeros(len(arrays),max(map(len,arrays)),1554,device='cuda')
    for i,values in enumerate(arrays):x[i,:len(values)]=torch.tensor(values,device='cuda')
    positions=torch.tensor(locations,device='cuda',dtype=torch.long)
    y=torch.tensor(targets,device='cuda',dtype=torch.float32)
    return x,positions,y


def chosen(actor,data):
    x,positions,_=data;raw=actor(x)[0]
    return raw[positions[:,0],positions[:,1]]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True,type=Path);ap.add_argument('--data',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path);ap.add_argument('--steps',type=int,default=150)
    ap.add_argument('--resume',type=Path,help='Typed BC checkpoint; --steps is the new total step count')
    a=ap.parse_args();assert not a.out.exists() and 1<=a.steps<=1000
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.manual_seed(20261025);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    report=read(a.data);assert report['version']=='combat_movement_demonstrations_v1' and report['state']=='passed' and report['ppo_eligible'] is False
    for p,h in report['inputs'].items():assert sha(p)==h,'Demonstration evidence changed'
    train,diagnostic=(load_data(report,s) for s in ('train','diagnostic'))
    model=read(a.model);model_hash=sha(a.model)
    assert model['memory_route_movement']['version']=='combat_memory_route_movement_v1', 'Yaw v2 requires its own trainer'
    actor=branch(base_model(model,allow_shared=True),'actor');actor.requires_grad_(False)
    assert actor.motor_branch is not None and actor.memory_route_branch is not None
    actor.motor_branch.requires_grad_(True);actor.memory_route_branch.requires_grad_(True)
    parameters=[p for p in actor.parameters() if p.requires_grad]
    assert parameters and all(p.is_cuda for p in actor.parameters())
    frozen={k:v.detach().clone() for k,v in actor.state_dict().items() if not k.startswith(PREFIXES)}
    optimizer=torch.optim.Adam([dict(params=list(actor.motor_branch.parameters()),name='motor'),
                                dict(params=list(actor.memory_route_branch.parameters()),name='route')],lr=.0003)
    assert not optimizer.state
    completed=0;parent_hash=model_hash
    if a.resume:
        previous=torch.load(a.resume,map_location='cuda',weights_only=True)
        assert previous['version']=='combat_movement_bc_checkpoint_v1' and previous['checkpoint_training_scope']==VERSION
        assert previous['weights_sha256']==model_hash and previous['data_sha256']==sha(a.data)
        for key,value in dict(version=VERSION,seed=20261025,learning_rate=.0003,source_retention=.02,latent_clip=.95,actor_scope='motor_and_route').items():
            assert previous['config'][key]==value,key
        completed=previous.get('completed_steps',previous['config']['steps']);assert completed<a.steps
        actor.load_state_dict(previous['actor']);optimizer.load_state_dict(copy.deepcopy(previous['optimizer']))
        baseline_train=previous['baseline_train'];baseline_diagnostic=previous['baseline_diagnostic']
        parent_hash=previous['parent_weights_sha256']
        torch.set_rng_state(previous['cpu_rng'].cpu());torch.cuda.set_rng_state_all([r.cpu() for r in previous['cuda_rng']])
    else:
        with torch.no_grad():
            baseline_train=chosen(actor,train).detach().clone();baseline_diagnostic=chosen(actor,diagnostic).detach().clone()
    assert baseline_train.shape==(len(train[2]),81) and baseline_diagnostic.shape==(len(diagnostic[2]),81)
    target=torch.atanh(train[2].clamp(-.95,.95))
    def objective():
        raw=chosen(actor,train)
        imitation=(raw[:,:2]-target).square().mean()
        retention=(raw[:,:2]-baseline_train[:,:2]).square().mean()
        return imitation+.02*retention
    def metrics(data):
        raw=chosen(actor,data)
        return dict(command_mean_absolute_error=float((raw[:,:2].tanh()-data[2]).abs().mean()),
                    latent_mse=float((raw[:,:2]-torch.atanh(data[2].clamp(-.95,.95))).square().mean()))
    with torch.no_grad():before=dict(train=metrics(train),diagnostic=metrics(diagnostic))
    losses=[]
    for step in range(completed,a.steps):
        loss=objective();assert torch.isfinite(loss)
        optimizer.zero_grad();loss.backward();norm=nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(norm)
        optimizer.step();losses.append(float(loss.detach()))
    with torch.no_grad():
        after=dict(train=metrics(train),diagnostic=metrics(diagnostic))
        train_raw=chosen(actor,train);diagnostic_raw=chosen(actor,diagnostic)
        assert torch.equal(train_raw[:,2:],baseline_train[:,2:]) and torch.equal(diagnostic_raw[:,2:],baseline_diagnostic[:,2:])
    assert all(torch.equal(actor.state_dict()[k],v) for k,v in frozen.items())
    result=copy.deepcopy(model);export_motor(result,actor.motor_branch);export_route(result,actor.memory_route_branch)
    for key in model:
        if key not in ('motor_residual','memory_route_movement'):assert result[key]==model[key],key
    assert sha(a.model)==model_hash
    a.out.mkdir(parents=True)
    save(a.out/'weights.json',result)
    config=dict(version=VERSION,steps=a.steps,seed=20261025,learning_rate=.0003,source_retention=.02,
        latent_clip=.95,checkpoint_selection='fixed_final_step',actor_scope='motor_and_route',optimizer_initialization='fresh_bc_not_ppo_resume')
    save(a.out/'config.json',config)
    cp=dict(version='combat_movement_bc_checkpoint_v1',checkpoint_training_scope=VERSION,
        weights_sha256=sha(a.out/'weights.json'),parent_weights_sha256=parent_hash,data_sha256=sha(a.data),completed_steps=a.steps,
        config=config,actor=actor.state_dict(),optimizer=optimizer.state_dict(),
        baseline_train=baseline_train,baseline_diagnostic=baseline_diagnostic,
        cuda_rng=torch.cuda.get_rng_state_all(),cpu_rng=torch.get_rng_state())
    torch.save(cp,a.out/'checkpoint.pt')
    save(a.out/'report.json',dict(version=VERSION,state='passed',device=torch.cuda.get_device_name(),training_performed=True,
        train_rows=len(train[2]),diagnostic_rows=len(diagnostic[2]),trainable_parameters=sum(p.numel() for p in parameters),
        before=before,after=after,loss_first=losses[0],loss_last=losses[-1],
        inputs={str(a.model.resolve()):model_hash,str(a.data.resolve()):sha(a.data)},
        weights_sha256=sha(a.out/'weights.json'),checkpoint_sha256=sha(a.out/'checkpoint.pt'),
        frozen_actor_verified=True,critic_unchanged=True,std_unchanged=True,nonmovement_raw_outputs_exact=True,
        optimizer_initialization='resumed_typed_bc' if a.resume else 'fresh_bc_not_ppo_resume',
        resumed_steps=completed,performed_steps=a.steps-completed,resume_sha256=sha(a.resume) if a.resume else None,
        ppo_data_mixed=False,promotion=False,
        scope='Movement action factors of successful rules only, with actual observed histories. Teacher forcing and frozen conditional aim need new native validation; no full joint imitation or optimal-action claim.'))
    print(json.dumps(dict(before=before,after=after),allow_nan=False),flush=True)


if __name__=='__main__':main()

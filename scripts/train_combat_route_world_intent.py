"""CUDA joint movement/yaw auxiliary in observation/world coordinates; 51 params."""
import argparse
import copy
import json
from pathlib import Path
from ppo_combat import torch,nn,training_devices
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model
from combat_memory_route_movement import YAW_VERSION,inputs
from train_combat_route_yaw_ridge import load_data
from combat_checkpoint_scope import require_checkpoint_scope
from train_combat_memory_route_ppo import admit
from ppo_recurrent import prepare_context,sequence
from evaluate_registered_combat_models import read,save,sha

SCOPE='combat_route_joint_world_intent_bc_v1'
CHECKPOINT='combat_route_world_intent_bc_checkpoint_v1'


def blank(model):
    result=copy.deepcopy(model);head=result['memory_route_movement']
    assert head['version']==YAW_VERSION
    head['layers'][-1]['weight']=[[0.]*16 for _ in range(3)];head['layers'][-1]['bias']=[0.]*3
    return result


def proposal(data,theta):
    latent=data['base3']+data['x']@theta.T
    f,s=latent[:,:2].tanh().unbind(-1);yaw=torch.pi*latent[:,2].tanh()
    world=torch.stack((f*yaw.cos()+s*yaw.sin(),f*yaw.sin()-s*yaw.cos()),-1)
    desired=torch.stack((data['turn'].cos(),data['turn'].sin()),-1)
    return world,desired,yaw


def metrics(data,theta):
    theta=theta.detach()
    world,desired,yaw=proposal(data,theta)
    error=yaw-data['turn'];alignment=(world*desired).sum(-1)/(world.norm(dim=-1)+1e-8)
    return dict(world_mse=float((world-.5*desired).square().mean()),mean_command_alignment=float(alignment.mean()),
        heading_mae_degrees=float(torch.atan2(error.sin(),error.cos()).abs().mean()*180/torch.pi),
        mean_absolute_turn_degrees=float(yaw.abs().mean()*180/torch.pi))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True,type=Path);ap.add_argument('--data',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path);ap.add_argument('--steps',type=int,default=400);ap.add_argument('--resume',type=Path)
    a=ap.parse_args();assert not a.out.exists() and 1<=a.steps<=2000;training_devices()
    torch.set_default_device('cuda');torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    model=read(a.model);fixed=blank(model);report=read(a.data)
    assert report['version']=='combat_observed_static_route_heading_v1' and report['state']=='passed' and report['ppo_eligible'] is False
    for p,h in report['inputs'].items():assert sha(p)==h
    # Fixed source reconstruction protects against silently training another encoder.
    sources=[p for p in report['inputs'] if Path(p).name=='weights.json'];assert len(sources)==1
    original=read(sources[0]);source_zero=copy.deepcopy(original);source_zero['memory_route_movement']['version']=YAW_VERSION
    source_zero['memory_route_movement']['layers'][-1]['weight'].append([0.]*16);source_zero['memory_route_movement']['layers'][-1]['bias'].append(0.)
    assert blank(source_zero)==fixed
    actor=branch(base_model(fixed,allow_shared=True),'actor').requires_grad_(False)
    train,diagnostic=(load_data(report,s,actor) for s in ('train','diagnostic'))
    assert not({tuple(w) for w in train['worlds']}&{tuple(w) for w in diagnostic['worlds']})
    head=model['memory_route_movement']['layers'][-1]
    initial=torch.tensor([r+[b] for r,b in zip(head['weight'],head['bias'])],device='cuda',dtype=torch.float64)
    theta=nn.Parameter(initial.clone());optimizer=torch.optim.Adam([theta],lr=.01)
    config=dict(version=SCOPE,learning_rate=.01,target_speed=.5,heading_weight=.1,anchor=.0001,max_grad_norm=1.)
    completed=0;anchor=initial.clone()
    if a.resume:
        cp=torch.load(a.resume,map_location='cuda',weights_only=True);require_checkpoint_scope(cp,SCOPE)
        assert cp['version']==CHECKPOINT and cp['weights_sha256']==sha(a.model) and cp['data_sha256']==sha(a.data) and cp['config']==config
        completed=cp['completed_steps'];assert completed<a.steps
        theta.data.copy_(cp['theta']);optimizer.load_state_dict(copy.deepcopy(cp['optimizer']));anchor=cp['anchor']
    before={s:metrics(d,theta) for s,d in [('train',train),('diagnostic',diagnostic)]}
    for _ in range(completed,a.steps):
        world,desired,yaw=proposal(train,theta)
        loss=(world-.5*desired).square().mean()+.1*(1-(yaw-train['turn']).cos()).mean()+.0001*(theta-anchor).square().mean()
        assert torch.isfinite(loss);optimizer.zero_grad();loss.backward()
        norm=nn.utils.clip_grad_norm_([theta],1.);assert torch.isfinite(norm);optimizer.step()
    assert bool(torch.isfinite(theta).all()) and float(theta.detach().abs().max())<1e4
    after={s:metrics(d,theta) for s,d in [('train',train),('diagnostic',diagnostic)]}
    result=copy.deepcopy(model);last=result['memory_route_movement']['layers'][-1]
    last['weight']=theta[:,:16].detach().float().cpu().tolist();last['bias']=theta[:,16].detach().float().cpu().tolist()
    assert blank(result)==fixed
    changed=branch(base_model(result,allow_shared=True),'actor').requires_grad_(False)
    for e in report['episodes']:
        rr,hh,_,_=admit(Path(e['source_cuda']),e['source_model_sha256'],[]);prepared=prepare_context(hh,rr,'cuda')
        with torch.no_grad():
            old,_=sequence(actor,prepared);new,_=sequence(changed,prepared)
            x=torch.tensor([r['features'] for r in rr],device='cuda');_,active=inputs(x,old)
            assert torch.equal(old[~active],new[~active]) and torch.equal(old[:,3:],new[:,3:])
            captured=[];handle=changed.memory_route_branch[-1].register_forward_pre_hook(lambda m,args:captured.append(args[0].detach()))
            raw,_=sequence(changed,prepared);handle.remove()
            _,loc,inv,_=prepared;basis=captured[0][loc[:,0],loc[:,1]][inv].double()
            expected=old[:,:3].double()+torch.cat((basis,torch.ones(len(basis),1,device='cuda',dtype=torch.float64)),-1)@theta.detach().float().double().T
            torch.testing.assert_close(raw[active,:3].double(),expected[active],atol=2e-5,rtol=1e-5)
    assert all(sha(p)==h for p,h in report['inputs'].items())
    a.out.mkdir(parents=True,exist_ok=False);save(a.out/'weights.json',result)
    cp=dict(version=CHECKPOINT,checkpoint_training_scope=SCOPE,weights_sha256=sha(a.out/'weights.json'),data_sha256=sha(a.data),config=config,
        theta=theta.detach(),optimizer=optimizer.state_dict(),anchor=anchor,completed_steps=a.steps)
    torch.save(cp,a.out/'checkpoint.pt')
    save(a.out/'report.json',dict(version=SCOPE,state='passed',device=torch.cuda.get_device_name(),training_performed=True,
        trainable_parameters=51,steps=a.steps,resumed_steps=completed,before=before,after=after,config=config,
        train_rows=len(train['x']),diagnostic_rows=len(diagnostic['x']),frozen_other_fields_exact=True,gate_and_other_raw_outputs_exact=True,
        inputs={str(a.model.resolve()):sha(a.model),str(a.data.resolve()):sha(a.data)},
        weights_sha256=sha(a.out/'weights.json'),checkpoint_sha256=sha(a.out/'checkpoint.pt'),promotion=False,
        scope='Joint world movement/yaw geometric auxiliary; mean-command kinematics, not native traversability, optimal teacher or PPO. New native quality required.'))
    print(json.dumps(dict(before=before,after=after)),flush=True)


if __name__=='__main__':main()

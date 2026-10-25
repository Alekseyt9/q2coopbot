"""Zero-initialize route yaw output; verify real-history identity on CUDA."""
import argparse
import copy
import json
from pathlib import Path
from ppo_combat import torch,training_devices
from ppo_recurrent import prepare_context,sequence
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model
from combat_memory_route_movement import VERSION,YAW_VERSION,validate,inputs,apply
from combat_memory_route_ppo import RouteScope
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True,type=Path);ap.add_argument('--screen',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path);a=ap.parse_args();assert not a.out.exists()
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    model=read(a.model);model_hash=sha(a.model);validate(model)
    assert model['memory_route_movement']['version']==VERSION
    migrated=copy.deepcopy(model);head=migrated['memory_route_movement'];head['version']=YAW_VERSION
    head['layers'][-1]['weight'].append([0.]*16);head['layers'][-1]['bias'].append(0.);validate(migrated)
    assert all(migrated[k]==v for k,v in model.items() if k!='memory_route_movement')
    actor=branch(base_model(model,allow_shared=True),'actor').requires_grad_(False)
    other=branch(base_model(migrated,allow_shared=True),'actor').requires_grad_(False)
    value=branch(model,'value').requires_grad_(False);new_value=branch(migrated,'value').requires_grad_(False)
    spec=read(a.screen/'spec.json');assert read(a.screen/'acceptance.json')['state']=='passed'
    cases=[];active_rows=0;pins={str(a.model.resolve()):model_hash};gating_checked=False
    for v in spec['variants']:
        if v['model_sha256']!=model_hash:continue
        directory=a.screen/f'case-{v["index"]:02d}/cuda';meta=read(directory/'report.json')
        assert meta['model_sha256']==model_hash and read(directory/'cuda-verification.json')['state']=='passed'
        rows,context=[[json.loads(s) for s in (directory/name).open()] for name in ('rollout.jsonl','sequence.jsonl')]
        for name,key in [('rollout.jsonl','rollout_sha256'),('sequence.jsonl','sequence_sha256')]:
            assert sha(directory/name)==meta[key];pins[str((directory/name).resolve())]=meta[key]
        prepared=prepare_context(context,rows,'cuda')
        with torch.no_grad():
            old,mem=sequence(actor,prepared,32,True,'actor');new,new_mem=sequence(other,prepared,32,True,'actor')
            old_v,vm=sequence(value,prepared,32,True,'value');new_v,nvm=sequence(new_value,prepared,32,True,'value')
            torch.testing.assert_close(old,new,atol=1e-5,rtol=1e-5)
            assert torch.equal(old[:,2:],new[:,2:]) and torch.equal(old_v,new_v)
            assert max(mem,new_mem,vm,nvm)<2e-5
            x=torch.tensor([r['features'] for r in rows],device='cuda');_,active=inputs(x,old);active_rows+=int(active.sum())
            if bool(active.any()) and not gating_checked:
                other.memory_route_branch[-1].bias[2]=.25
                shifted,_=sequence(other,prepared)
                assert torch.equal(shifted[~active],new[~active]) and torch.equal(shifted[:,:2],new[:,:2])
                assert torch.equal(shifted[:,3:],new[:,3:])
                torch.testing.assert_close(shifted[active,2]-new[active,2],torch.full((int(active.sum()),),.25,device='cuda'),atol=1e-6,rtol=1e-6)
                for name in ('visible','dead','missing','no-memory','unknown-route'):
                    changed=x.clone()
                    if name=='visible':changed[:,73]=1
                    elif name=='dead':changed[:,0]=0
                    elif name=='missing':changed[:,1489]=0
                    elif name=='no-memory':changed[:,881:1121:30]=0
                    else:changed[:,1490::8]=0
                    assert torch.equal(apply(other.memory_route_branch,changed,old),old),name
                other.memory_route_branch[-1].bias[2]=0.
                gradient_data=(x,old,active)
                gating_checked=True
        cases.append(dict(index=v['index'],rows=len(rows),actor_max_error=float((old-new).abs().max()),critic_exact=True,
                          context_max_error=max(mem,new_mem,vm,nvm)))
    assert cases and active_rows>0 and gating_checked
    # A synthetic gradient proves yaw is trainable without exporting fitted weights.
    other.memory_route_branch.requires_grad_(True)
    gradient_x,gradient_raw,gradient_active=gradient_data
    corrected=apply(other.memory_route_branch,gradient_x,gradient_raw.detach())
    corrected[gradient_active,2].sum().backward()
    assert other.memory_route_branch[-1].bias.grad[2]!=0
    try:RouteScope(migrated,actor_lr=.0003,value_lr=.0003)
    except AssertionError:legacy_rejected=True
    else:raise AssertionError('Legacy PPO scope accepted yaw v2')
    assert all(sha(p)==h for p,h in pins.items())
    a.out.mkdir(parents=True,exist_ok=False);save(a.out/'weights.json',migrated)
    save(a.out/'report.json',dict(version='combat_route_yaw_zero_migration_v1',state='passed',device=torch.cuda.get_device_name(),
        cases=cases,rows=sum(c['rows'] for c in cases),route_active_rows=active_rows,
        source_model_sha256=model_hash,weights_sha256=sha(a.out/'weights.json'),inputs=pins,
        nonzero_yaw_gating_verified=True,yaw_gradient_nonzero=True,legacy_training_scope_rejected=legacy_rejected,
        frozen_other_model_fields_exact=True,training_performed=False,promotion=False,
        scope='Zero migration and temporary CUDA yaw perturbation/gradient only; no fitted yaw policy or gameplay-quality claim.'))
    print(dict(cases=len(cases),rows=sum(c['rows'] for c in cases),route_active_rows=active_rows,zero_migration=True),flush=True)


if __name__=='__main__':main()

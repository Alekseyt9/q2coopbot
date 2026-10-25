"""Real-history CUDA checkpoint/export/next-step audit for motor+route PPO."""
import argparse
import copy
from pathlib import Path
from ppo_combat import torch, nn, training_devices
from ppo_recurrent import prepare_context, sequence
from combat_movement_ppo import MovementScope, SCOPE, PREFIXES
from combat_memory_route_ppo import RouteScope
from combat_checkpoint_scope import require_checkpoint_scope
from combat_route_task_objective import AUTONOMOUS, collect_evidence
from train_combat_memory_route_ppo import admit
from evaluate_registered_combat_models import read, save, sha


def equal(a,b):
    if torch.is_tensor(a): return torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trained',required=True,type=Path)
    ap.add_argument('--request',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args();assert not a.out.exists();training_devices()
    torch.set_default_device('cuda');torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    report=read(a.trained/'report.json');config=read(a.trained/'config.json');model=read(a.trained/'weights.json')
    model_hash=sha(a.trained/'weights.json');assert model_hash==report['weights_sha256']
    assert sha(a.trained/'checkpoint.pt')==report['checkpoint_sha256']
    checkpoint=torch.load(a.trained/'checkpoint.pt',map_location='cuda',weights_only=True)
    make=lambda:MovementScope(model,actor_lr=config['actor_lr'],value_lr=config['value_lr'])
    left,right=make(),make()
    for scope in (left,right):
        assert not scope.actor_optimizer.state and not scope.value_optimizer.state
        assert scope.restore(checkpoint,model_hash,config,AUTONOMOUS)==report['consumed_rollouts']
        assert scope.export()==model, 'JSON export differs after checkpoint restore'
    request=read(a.request);rows,history=[],[]
    for directory in request['data']:
        rr,hh,_,_=admit(Path(directory),request['behavior_sha256'],[]);rows.extend(rr);history.extend(hh)
    # Binding by hash must select the same actual candidate as the legacy label.
    legacy,_=collect_evidence(Path(request['capture_root']),rows)
    bound,_=collect_evidence(Path(request['capture_root']),rows,model_hash=request['behavior_sha256'])
    assert bound==legacy
    prepared=prepare_context(history,rows,'cuda')
    with torch.no_grad():
        x,_=sequence(left.actor,prepared);y,_=sequence(right.actor,prepared)
        assert torch.equal(x,y)
    # Verify Adam continuation on the same real histories for both optimizers.
    # This is a serialization probe, not an additional PPO-trained candidate.
    for scope in (left,right):
        raw,_=sequence(scope.actor,prepared)
        loss=raw[:,:2].square().mean()
        scope.actor_optimizer.zero_grad();loss.backward()
        nn.utils.clip_grad_norm_(scope.actor_parameters,config['max_grad_norm']);scope.actor_optimizer.step()
        value,_=sequence(scope.value,prepared)
        scope.value_optimizer.zero_grad();value.square().mean().backward()
        nn.utils.clip_grad_norm_(scope.value.parameters(),config['max_grad_norm']);scope.value_optimizer.step()
        scope.verify_frozen()
    assert equal(left.actor.state_dict(),right.actor.state_dict())
    assert equal(left.value.state_dict(),right.value.state_dict())
    assert equal(left.actor_optimizer.state_dict(),right.actor_optimizer.state_dict())
    assert equal(left.value_optimizer.state_dict(),right.value_optimizer.state_dict())
    rejected=0
    for wrong in (None,'combat_movement_demonstration_bc_v1','combat_memory_route_movement_ppo_v1'):
        bad=copy.deepcopy(checkpoint);bad['checkpoint_training_scope']=wrong
        before=copy.deepcopy(left.actor.state_dict())
        try:left.restore(bad,model_hash,config,AUTONOMOUS)
        except ValueError:rejected+=1
        else:raise AssertionError('Wrong scope admitted')
        assert equal(before,left.actor.state_dict())
    for wrong_hash,wrong_config,objective in [('0'*64,config,AUTONOMOUS),(model_hash,{**config,'actor_steps':99},AUTONOMOUS),
                                              (model_hash,config,'combat_continuity_v1')]:
        before=copy.deepcopy(left.actor.state_dict())
        try:left.restore(checkpoint,wrong_hash,wrong_config,objective)
        except AssertionError:rejected+=1
        else:raise AssertionError('Wrong identity admitted')
        assert equal(before,left.actor.state_dict())
    for wrong in (None,'combat_memory_route_movement_ppo_v1'):
        try:require_checkpoint_scope(checkpoint,wrong)
        except ValueError:rejected+=1
        else:raise AssertionError('Dedicated checkpoint admitted to another scope')
    parent_path=next(Path(p) for p in report['inputs'] if Path(p).name=='weights.json')
    parent=read(parent_path)
    assert parent['motor_residual']!=model['motor_residual'] and parent['memory_route_movement']!=model['memory_route_movement']
    assert all(parent[k]==model[k] for k in parent if k not in ('motor_residual','memory_route_movement','value','attention'))
    assert parent['attention']['actor']==model['attention']['actor']
    save(a.out,dict(state='passed',device='cuda',rows=len(rows),exact_restore=True,exact_next_actor_and_critic_step=True,
        actor_and_value_adam_exact=True,frozen_actor_verified=True,both_movement_branches_changed=True,
        json_export_roundtrip_exact=True,negative_identity_cases=rejected,evidence_hash_binding_matches=True,
        quality_claim=False,weights_sha256=model_hash,checkpoint_sha256=sha(a.trained/'checkpoint.pt')))
    print('CUDA motor+route scope audit passed',flush=True)


if __name__=='__main__':main()

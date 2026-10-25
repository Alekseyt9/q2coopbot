"""CUDA branch gating, gradients, frozen scope and exact optimizer resume test.

Synthetic numerical objectives only; no combat training or usable fitted weights.
"""
import argparse
import copy
import json
from pathlib import Path
from ppo_combat import torch, training_devices
from combat_memory_route_ppo import RouteScope, SCOPE
from combat_memory_route_movement import inputs, apply
from evaluate_registered_combat_models import read, save, sha


def equal(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepared',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.manual_seed(20261025);torch.use_deterministic_algorithms(True)
    a.out.mkdir(parents=True,exist_ok=False)
    model_path=a.prepared/'weights.json';model=read(model_path)
    enriched=[json.loads(s) for s in (a.prepared/'enriched-1.jsonl').open(encoding='utf-8')]
    x=torch.tensor([r['features'] for r in enriched],device='cuda')
    raw=torch.randn(len(x),81,device='cuda')
    config=dict(actor_lr=.0003,value_lr=.0003)
    original=RouteScope(model,**config)
    head=original.actor.memory_route_branch
    features,active=inputs(x,raw)
    assert int(active.sum())>0
    with torch.no_grad():
        assert torch.equal(apply(head,x,raw),raw)
        head[-1].bias.copy_(torch.tensor([.25,-.5],device='cuda'))
        corrected=apply(head,x,raw)
        assert torch.equal(corrected[...,2:],raw[...,2:])
        assert torch.equal(corrected[~active],raw[~active])
        torch.testing.assert_close(corrected[active,:2]-raw[active,:2],torch.tensor([.25,-.5],device='cuda').expand(int(active.sum()),2))
        # Every gate must suppress even a nonzero learned correction.
        for kind in ('visible','dead','missing','unknown','no-memory'):
            changed=x.clone()
            if kind=='visible':changed[:,73]=1
            elif kind=='dead':changed[:,0]=0
            elif kind=='missing':changed[:,1489]=0
            elif kind=='unknown':changed[:,1490::8]=0
            else:changed[:,881:1121:30]=0
            assert torch.equal(apply(head,changed,raw),raw),kind
        head[-1].bias.zero_()
    before={k:v.detach().clone() for k,v in head.state_dict().items()}
    def step(scope):
        scope.actor_optimizer.zero_grad();scope.value_optimizer.zero_grad()
        output=apply(scope.actor.memory_route_branch,x,raw)
        # Exercises nonzero actor gradients and critic optimizer moments without
        # treating historical outcomes as fresh on-policy training labels.
        loss=(output[active,:2]-raw[active,:2]-torch.tensor([.2,-.1],device='cuda')).square().mean()
        loss.backward();scope.actor_optimizer.step()
        value=scope.value(x[:4,None,:])[0]
        critic_loss=(value-.3).square().mean();critic_loss.backward();scope.value_optimizer.step()
        scope.verify_frozen()
    step(original)
    assert any(not torch.equal(v,head.state_dict()[k]) for k,v in before.items())
    checkpoint=original.snapshot(sha(model_path),config,['synthetic-numerical-step-1'])
    cp=a.out/'scope-test.pt';torch.save(checkpoint,cp)
    restored=RouteScope(model,**config)
    consumed=restored.restore(torch.load(cp,map_location='cuda',weights_only=True),sha(model_path),config)
    assert consumed==['synthetic-numerical-step-1']
    assert equal(original.actor.state_dict(),restored.actor.state_dict()) and equal(original.value.state_dict(),restored.value.state_dict())
    step(original);step(restored)
    assert equal(original.actor.state_dict(),restored.actor.state_dict()) and equal(original.value.state_dict(),restored.value.state_dict())
    assert equal(original.actor_optimizer.state_dict(),restored.actor_optimizer.state_dict())
    assert equal(original.value_optimizer.state_dict(),restored.value_optimizer.state_dict())
    # Generic checkpoint scope must reject this dedicated optimizer layout.
    from combat_checkpoint_scope import require_checkpoint_scope
    try:require_checkpoint_scope(checkpoint)
    except ValueError:pass
    else:raise AssertionError('Dedicated optimizer admitted by generic scope')
    invalid=copy.deepcopy(checkpoint);invalid['behavior_sha256']='0'*64
    try:restored.restore(invalid,sha(model_path),config)
    except AssertionError:pass
    else:raise AssertionError('Wrong behavior identity accepted')
    save(a.out/'report.json',dict(state='passed',device=torch.cuda.get_device_name(),scope=SCOPE,
        route_active_rows=int(active.sum()),gating_verified=True,frozen_actor_verified=True,
        exact_resume_verified=True,generic_checkpoint_rejected=True,wrong_behavior_rejected=True,
        model_sha256=sha(model_path),checkpoint_sha256=sha(cp),
        combat_training_performed=False,usable_fitted_weights_exported=False,promotion=False))
    print('CUDA route scope and exact resume passed',flush=True)


if __name__=='__main__':main()

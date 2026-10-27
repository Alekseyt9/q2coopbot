"""Synthetic CUDA optimizer resume proof; does not fit gameplay data."""
import argparse
import json
from pathlib import Path

from ppo_combat import torch,training_devices
from process_combat_architecture_pool import read,save,sha
from combat_movement_option_ppo import MovementOptionScope,SCOPE
from combat_full_joint_ppo import equal_tree


def run(model_path,out):
    assert not out.exists()
    training_devices();torch.set_num_threads(2);torch.manual_seed(20261027)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    model=read(model_path);config=dict(version=SCOPE,actor_lr=.001,value_lr=.001)
    features=torch.randn((32,1554),device='cuda');raw=torch.randn((32,81),device='cuda')
    target=torch.arange(32,device='cuda')%5;value_target=torch.linspace(-1,1,32,device='cuda')
    rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
    def fresh():
        scope=MovementOptionScope(model,config);torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng)
        return scope
    def steps(scope,n):
        for _ in range(n):
            logits,value=scope.options(features,raw)
            scope.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss=torch.nn.functional.cross_entropy(logits,target)
            actor_loss.backward();scope.actor_optimizer.step()
            scope.value_optimizer.zero_grad(set_to_none=True)
            value_loss=(value-value_target).square().mean()
            value_loss.backward();scope.value_optimizer.step()
        scope.verify()
    full=fresh();steps(full,8)
    split=fresh();steps(split,4)
    out.mkdir(parents=True)
    weights=out/'synthetic-partial-weights.json'
    weights.write_text(json.dumps(split.export(),separators=(',',':')),encoding='utf-8')
    checkpoint=out/'synthetic-partial-checkpoint.pt'
    torch.save(split.snapshot(sha(weights),[],dict(actor_steps=4,value_steps=4,current_rollout='synthetic-no-gameplay')),checkpoint)
    resumed=MovementOptionScope(read(weights),config)
    consumed,progress=resumed.restore(torch.load(checkpoint,map_location='cuda',weights_only=True),sha(weights))
    assert not consumed and progress['actor_steps']==progress['value_steps']==4
    steps(resumed,4)
    assert equal_tree(full.options.state_dict(),resumed.options.state_dict())
    assert equal_tree(full.actor_optimizer.state_dict(),resumed.actor_optimizer.state_dict())
    assert equal_tree(full.value_optimizer.state_dict(),resumed.value_optimizer.state_dict())
    assert full.export()==resumed.export()
    save(out/'report.json',dict(state='complete',device='cuda',steps=8,split_steps=[4,4],
        actor_and_value_tensors_exact=True,adam_moments_exact=True,json_export_exact=True,
        frozen_source_sha256=full.source_sha256,source_weights_sha256=sha(model_path),
        checkpoint_sha256=sha(checkpoint),training_scope=SCOPE,
        scope='Synthetic algebra-only data. No gameplay TRAIN/validation fitting, PPO claim, quality evaluation or deployment. Proves typed optimizer serialization/resume, not native-data training admission.'))
    print(dict(state='complete',resume='4+4 equals8 exactly',source_frozen=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();run(a.model.resolve(),a.out.resolve())

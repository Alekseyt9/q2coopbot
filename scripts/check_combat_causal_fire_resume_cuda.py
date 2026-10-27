"""Exact continuation, frozen state, and ordinary-PPO rejection on real fits."""
import argparse
from pathlib import Path
from ppo_combat import torch, training_devices
from combat_full_joint_ppo import equal_tree,SCOPE
from combat_checkpoint_scope import require_checkpoint_scope
from train_combat_causal_fire_cuda import VERSION,CHECKPOINT_VERSION
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('continuous','resumed','source','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();training_devices()
    cp=lambda p:torch.load(p/'checkpoint.pt',map_location='cuda',weights_only=True)
    left,right,source=cp(a.continuous),cp(a.resumed),cp(a.source)
    assert left['version']==right['version']==CHECKPOINT_VERSION
    assert left['checkpoint_training_scope']==right['checkpoint_training_scope']==VERSION
    checks={}
    for key in ('actor','value','joint','log_std','actor_optimizer','value_optimizer','rng','cuda_rng',
                'config','optimizer_layout','preference_config','preference_steps_completed','origin','consumed_seeds',
                'consumed_rollouts','retention_teacher_logits','retention_ids','dataset_sha256','retention_request_sha256'):
        assert equal_tree(left[key],right[key]),'Resume differs: '+key
        checks['exact_'+key]=True
    for key in ('value','joint','log_std','value_optimizer'):
        assert equal_tree(left[key],source[key]),'Frozen state changed: '+key
        checks['frozen_'+key]=True
    for name,value in source['actor'].items():
        trainable='weapon_fire_residual_branch.' if left['preference_config'].get('cycle_residual_only') else 'conditional_fire_branch.'
        if not name.startswith(trainable):
            assert equal_tree(left['actor'][name],value),'Non-fire actor changed: '+name
    checks['frozen_actor']=True
    assert sha(a.continuous/'weights.json')==sha(a.resumed/'weights.json')
    checks['identical_export']=True
    try:require_checkpoint_scope(left,SCOPE)
    except ValueError:checks['ordinary_ppo_rejected']=True
    else:raise AssertionError('Ordinary PPO accepted preference scope')
    for root in (a.continuous,a.resumed):
        report=read(root/'report.json')
        assert report['state']=='passed' and report['technical_only']
        assert report['non_fire_parameters_and_adam_unchanged'] and report['critic_and_adam_unchanged']
    checks['frozen_adam_checks_passed']=True
    save(a.out,dict(state='passed',device='cuda',checks=checks,weights_sha256=sha(a.continuous/'weights.json'),
        scope='Technical continuation/freeze validation only; no native combat quality claim.'))
    print('Passed',len(checks),'CUDA continuation/freeze checks')


if __name__=='__main__':main()

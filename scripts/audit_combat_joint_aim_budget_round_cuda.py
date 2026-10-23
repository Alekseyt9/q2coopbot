"""Seal matched B/G/S update states; neural tensors only on CUDA."""
import argparse
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha
from train_combat_bc import torch, training_devices
from fork_conditional_weapon_exploration_cuda import equal_tree


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--round-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();training_devices();torch.set_default_device('cuda')
    reports={};checkpoints={};pins={}
    for arm in ('B','G','S'):
        root=a.round_root/arm;closed=read(root/'complete.json')
        for name,key in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):
            p=(root/name).resolve();assert sha(p)==closed[key];pins[str(p)]=closed[key]
        report=read(root/'report.json');assert report['state']=='complete' and report['device']=='cuda' and report['arm']==arm
        assert report['predictor_exact'] and report['source_checkpoint_memory_exact'] and report['inactive_phase_parameters_and_adam_exact']
        assert report['checkpoint_optimizer_roundtrip_exact'] and report['rng_preserved']
        reports[arm]=report;checkpoints[arm]=torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=True)
    rounds={r['round'] for r in reports.values()};assert len(rounds)==1
    first=next(iter(rounds))==1
    if first:
        assert len({r['behavior_sha256'] for r in reports.values()})==1 and len({r['rollout_sha256'] for r in reports.values()})==1
        for arm in ('G','S'):
            for key in ('value','value_optimizer','rng','cuda_rng'):
                assert equal_tree(checkpoints['B'][key],checkpoints[arm][key]),key
    else:
        assert len({r['behavior_sha256'] for r in reports.values()})==3 and len({r['rollout_sha256'] for r in reports.values()})==3
    assert reports['B']['head_frozen_exact'] and reports['B']['head_steps']==0
    for arm in ('B','S'):
        assert reports[arm]['full_approx_kl']<=.005+1e-6
        assert max(reports[arm]['base_approx_kl'],reports[arm]['category_exact_kl_active'])<=.0025+1e-6
    assert reports['G']['full_approx_kl']<=.005+1e-6
    result=dict(state='complete',version='combat_joint_aim_budget_round_audit_v1',device='cuda',round=next(iter(rounds)),
        first_shared_on_policy_corpus=first,critic_and_adam_exact_across_arms=first,
        inputs=pins,arms={k:{n:r[n] for n in ('base_steps','head_steps','critic_steps','rows','nonzero_correction_rows',
            'full_approx_kl','base_approx_kl','category_exact_kl_active','baseline_probability_after','weights_sha256')} for k,r in reports.items()},
        quality_verified=False,promotion=False,script_sha256=sha(__file__))
    a.out.mkdir(parents=True);save(a.out/'report.json',result);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(result)


if __name__=='__main__':main()

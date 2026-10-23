"""Seal B/G experimental updates; compare neural state exclusively on CUDA."""
import argparse
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha
from train_combat_bc import torch, training_devices
from combat_geometry_ppo import TRAINING_VERSION
from fork_conditional_weapon_exploration_cuda import equal_tree


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--round-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); assert not a.out.exists(); training_devices()
    reports = {}; checkpoints = {}; pins = {}
    for arm in ('B', 'G'):
        root = a.round_root/arm; seal = read(root/'complete.json')
        for name, key in (('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'), ('report.json', 'report_sha256')):
            p = (root/name).resolve(); assert sha(p) == seal[key]; pins[str(p)] = seal[key]
        r = read(root/'report.json')
        assert r['state'] == 'complete' and r['version'] == TRAINING_VERSION and r['device'] == 'cuda' and r['arm'] == arm
        assert r['predictor_exact'] and r['legacy_head_exact'] and r['checkpoint_optimizer_roundtrip_exact'] and r['rng_preserved']
        assert r['verified_native_cases'] == 32 and r['current_cached_full_lp_max_error'] < 1e-4
        assert r['full_approx_kl'] <= .005+1e-6 and max(r['base_approx_kl'], r['category_exact_kl_active']) <= .0025+1e-6
        reports[arm] = r; checkpoints[arm] = torch.load(root/'checkpoint.pt', map_location='cuda', weights_only=True)
        assert checkpoints[arm]['geometry_aim_training']['arm'] == arm
        assert checkpoints[arm]['geometry_aim_training']['updates_completed'] == r['round']
        assert checkpoints[arm]['consumed_rollouts'][-1] == r['rollout_sha256']
    assert reports['B']['round'] == reports['G']['round']
    assert reports['B']['experiment_protocol_sha256'] == reports['G']['experiment_protocol_sha256']
    first = reports['B']['round'] == 1
    if first:
        assert reports['B']['behavior_sha256'] == reports['G']['behavior_sha256']
        assert reports['B']['rollout_sha256'] == reports['G']['rollout_sha256']
        for key in ('value', 'value_optimizer', 'rng', 'cuda_rng'):
            assert equal_tree(checkpoints['B'][key], checkpoints['G'][key]), key
    else:
        assert reports['B']['rollout_sha256'] != reports['G']['rollout_sha256']
    assert reports['B']['head_frozen_exact'] and reports['B']['head_steps'] == 0
    for key in ('aim_candidate_head', 'aim_candidate_optimizer'):
        assert equal_tree(checkpoints['B'][key], checkpoints['G'][key]), 'Inactive legacy state: '+key
    result = dict(state='complete', version='combat_geometry_ppo_round_audit_v1', device='cuda', round=reports['B']['round'],
        first_shared_on_policy_corpus=first, first_critic_adam_rng_equal=first, inputs=pins, promotion=False, quality_verified=False,
        arms={arm: {k: r[k] for k in ('rows', 'base_steps', 'head_steps', 'critic_steps', 'full_approx_kl', 'base_approx_kl',
            'category_exact_kl_active', 'baseline_probability_after', 'weights_sha256')} for arm, r in reports.items()})
    a.out.mkdir(parents=True); save(a.out/'report.json', result)
    save(a.out/'complete.json', dict(state='complete', report_sha256=sha(a.out/'report.json')))
    print(result)


if __name__ == '__main__': main()

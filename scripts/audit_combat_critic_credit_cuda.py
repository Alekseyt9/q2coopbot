"""Seal the matched critic-target/timing experiment using CUDA comparisons."""
import argparse
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    import torch
    assert torch.cuda.is_available()
    from fork_conditional_weapon_exploration_cuda import equal_tree
    root = args.root.resolve(); protocol = read(root / 'protocol.json')
    assert read(root / 'training-complete.json')['state'] == 'complete'
    source = Path(protocol['source'])
    initial = torch.load(source / 'checkpoint.pt', map_location='cuda', weights_only=True)
    checkpoints = {}; reports = {}; inputs = []
    for mode in protocol['modes']:
        update = root / mode; seal = read(update / 'complete.json')
        for name, key in [('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'), ('report.json', 'report_sha256')]:
            assert sha(update / name) == seal[key]
            inputs.append(dict(path=str(update / name), sha256=seal[key]))
        report = read(update / 'report.json')
        assert report['device'] == 'cuda' and report['actor_steps'] <= 10
        boundary_path = root / (mode + '-initial.pt')
        assert sha(boundary_path) == report['initial_boundary_sha256']
        boundary = torch.load(boundary_path, map_location='cuda', weights_only=True)
        for key in ['actor', 'value', 'log_std', 'actor_optimizer', 'value_optimizer', 'rng', 'cuda_rng']:
            assert equal_tree(boundary[key], initial[key]), (mode, key)
        assert boundary['rollout_sha256'] not in initial['consumed_rollouts']
        checkpoints[mode] = torch.load(update / 'checkpoint.pt', map_location='cuda', weights_only=True)
        reports[mode] = report
    a, b, legacy = checkpoints['mc-after'], checkpoints['mc-before'], checkpoints['legacy']
    for key in ['value', 'value_optimizer']:
        assert equal_tree(a[key], b[key]), key
    for key in ['actor', 'log_std', 'actor_optimizer']:
        assert equal_tree(a[key], legacy[key]), ('actor unaffected by after-target change', key)
    old_control = root.parent / 'combat-future-aux-calibrated-v1-20261022/control'
    assert sha(root / 'legacy/weights.json') == sha(old_control / 'weights.json'), 'Legacy default regressed'
    assert not equal_tree(a['actor'], b['actor']), 'Warm baseline had no actor effect'
    assert reports['mc-before']['critic_credit']['stats']['actor_unchanged']
    for mode in ['mc-after', 'mc-before']:
        assert reports[mode]['critic_credit']['stats']['steps'] == 40
        assert len(checkpoints[mode]['critic_credit_forks']) == 1
    for name, digest in protocol['inputs'].items():
        path = Path(name)
        if path.name == 'ppo_recurrent.py': path = root / 'legacy/trainer.py'
        if path.name == 'combat_critic_credit.py': path = root / 'critic-helper.py'
        assert sha(path) == digest, str(path)
    save(args.out, dict(state='complete', device='cuda', source_boundary_exact=True,
         legacy_runtime_exact=True, after_target_actor_exact_to_legacy=True,
         mc_before_after_final_critic_and_adam_exact=True, candidate_actor_different=True,
         initial_rollout_unconsumed=True, variants={mode: dict(actor_steps=r['actor_steps'],
         kl=r['final_approx_kl'], critic=r['critic_credit']) for mode,r in reports.items()},
         inputs=inputs, script_sha256=sha(__file__), promotion=False, training_performed=False))
    print(args.out)


if __name__ == '__main__':
    main()

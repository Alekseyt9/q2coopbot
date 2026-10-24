"""Fork only objective metadata, with identical CUDA neural/optimizer/history state."""
import argparse
import copy
from pathlib import Path
import shutil

from evaluate_registered_combat_models import read, save, sha
from train_combat_bc import training_devices, torch
from fork_conditional_weapon_exploration_cuda import equal_tree
from ppo_recurrent import durable_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'registry', 'reward', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    pin(__file__)
    parent_seal = read(pin(args.source / 'complete.json'))
    for name, key in (('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'), ('report.json', 'report_sha256')):
        pin(args.source / name, parent_seal[key])
    parent_report = read(args.source / 'report.json')
    assert parent_report['version'] == 'combat_diverse_registry_ppo_v1' and parent_report['round'] == 8
    assert parent_report['device'] == 'cuda' and parent_report['checkpoint_optimizer_roundtrip_exact']
    candidate = read(pin(args.reward))
    assert candidate['version'] == 'combat_reward_v14' and candidate['blocked_ground_request'] == -.01
    index = read(pin(args.registry))
    descriptors = [(name, read(pin(args.registry.parent / name))) for name in index['files']]
    assert len(descriptors) == 24
    repo = Path(__file__).resolve().parents[1]
    reward_paths = {str((repo / e['recipe']['reward_config']).resolve()) for _, e in descriptors}
    assert len(reward_paths) == 1
    old_path = pin(next(iter(reward_paths)))
    old_reward = read(old_path)
    expected = copy.deepcopy(old_reward)
    expected.update(version='combat_reward_v14', blocked_ground_request=-.01)
    assert expected == candidate, 'Only the explicit proposal cost may change'
    training_devices()
    cp = torch.load(args.source / 'checkpoint.pt', map_location='cuda', weights_only=True)
    assert cp['weights_sha256'] == parent_seal['weights_sha256']
    assert cp['config']['objective_reward_sha256'] == sha(old_path)
    assert cp['geometry_aim_training']['version'] == 'combat_diverse_registry_ppo_v1'
    assert cp['geometry_aim_training']['updates_completed'] == 8
    # CPU load/save transports state only; all tensor equality checks run on CUDA.
    serial = torch.load(args.source / 'checkpoint.pt', map_location='cpu', weights_only=True)
    args.out.mkdir(parents=True)
    branches = {}
    for label, coefficient in (('zero', 0), ('cost', -.01)):
        root = args.out / label
        root.mkdir()
        reward = copy.deepcopy(candidate)
        reward['blocked_ground_request'] = coefficient
        reward_path = root / 'reward.json'
        save(reward_path, reward)
        child = copy.deepcopy(serial)
        child['config']['objective_reward_sha256'] = sha(reward_path)
        migration = dict(version='combat_blocked_ground_objective_fork_v1', branch=label,
            source_checkpoint_sha256=parent_seal['checkpoint_sha256'],
            source_reward_sha256=sha(old_path), reward_sha256=sha(reward_path),
            coefficient=coefficient, fresh_own_policy_data_required=True)
        child['objective_forks'] = child.get('objective_forks', []) + [migration]
        initial = root / 'initializer'
        initial.mkdir()
        shutil.copy2(args.source / 'weights.json', initial / 'weights.json')
        durable_write(initial / 'checkpoint.pt', lambda stream: torch.save(child, stream))
        loaded = torch.load(initial / 'checkpoint.pt', map_location='cuda', weights_only=True)
        for key in cp:
            if key not in ('config', 'objective_forks'):
                assert equal_tree(cp[key], loaded[key]), key
        config = copy.deepcopy(loaded['config'])
        config['objective_reward_sha256'] = cp['config']['objective_reward_sha256']
        assert config == cp['config']
        assert loaded['objective_forks'][:-1] == cp.get('objective_forks', [])
        assert loaded['objective_forks'][-1] == migration
        assert sha(initial / 'weights.json') == parent_seal['weights_sha256']
        registry = root / 'registry'
        registry.mkdir()
        relative_reward = reward_path.resolve().relative_to(repo).as_posix()
        for name, original in descriptors:
            assert Path(name).name == name
            updated = copy.deepcopy(original)
            updated['recipe']['reward_config'] = relative_reward
            # Retain both revision and seed_revision: physical instance generation
            # is identical; the separate registry SHA records the new objective.
            save(registry / name, updated)
            restored = copy.deepcopy(updated)
            restored['recipe']['reward_config'] = original['recipe']['reward_config']
            assert restored == original
        save(registry / 'index.json', index)
        report = dict(state='complete', version='combat_blocked_ground_objective_fork_v1', device='cuda',
            branch=label, coefficient=coefficient, source_weights_sha256=parent_seal['weights_sha256'],
            source_checkpoint_sha256=parent_seal['checkpoint_sha256'], reward_sha256=sha(reward_path),
            actor_value_std_adam_rng_history_exact=True, checkpoint_roundtrip_exact=True,
            weights_bytes_exact=True, training_performed=False, promotion=False, inputs=pins)
        save(initial / 'report.json', report)
        save(initial / 'complete.json', dict(state='complete', weights_sha256=sha(initial / 'weights.json'),
            checkpoint_sha256=sha(initial / 'checkpoint.pt'), report_sha256=sha(initial / 'report.json')))
        branches[label] = dict(initializer=str(initial.resolve()), registry=str((registry / 'index.json').resolve()),
            reward=str(reward_path.resolve()), reward_sha256=sha(reward_path),
            seal_sha256=sha(initial / 'complete.json'))
    assert all(sha(path) == digest for path, digest in pins.items())
    save(args.out / 'report.json', dict(state='complete', device='cuda', branches=branches, inputs=pins,
        training_performed=False, promotion=False, fresh_own_policy_data_required=True))
    save(args.out / 'complete.json', dict(state='complete', report_sha256=sha(args.out / 'report.json')))
    print(branches)


if __name__ == '__main__':
    main()

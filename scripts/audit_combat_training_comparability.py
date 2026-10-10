"""Compare A/B conditions and fork receipts without executing a neural network."""
import argparse
import copy
import pathlib

from process_combat_architecture_pool import read, save, sha


def audit(root):
    protocol = read(root / 'protocol.json')
    names = ('control', 'quality')
    plans = {n: read(root / n / 'capture/train/plan.json') for n in names}
    configs = {n: read(root / n / 'config.json') for n in names}
    forks = {n: read(root / n / 'fork/report.json') for n in names}
    for n in names:
        assert forks[n]['device'] == 'cuda' and forks[n]['actor_std_exact']
        assert forks[n]['checkpoint_readback_exact']
        assert forks[n]['weights_sha256'] == sha(root / n / 'fork/weights.json')
        assert forks[n]['checkpoint_sha256'] == sha(root / n / 'fork/checkpoint.pt')
        assert configs[n]['objective_reward_sha256'] == protocol[n + '_reward_sha256']
        configs[n].pop('objective_reward_sha256')
        assert forks[n]['critic_optimizer_preserved'] == protocol['critic_optimizer_preserved']
    assert configs['control'] == configs['quality']
    assert forks['control']['weights_sha256'] == forks['quality']['weights_sha256']
    records = []
    assert len(plans['control']['tasks']) == len(plans['quality']['tasks'])
    for left, right in zip(plans['control']['tasks'], plans['quality']['tasks']):
        assert left['seeds'] == right['seeds'] and left['instances'] == right['instances']
        for field in ('split', 'modes', 'runner_path', 'runner_sha256'):
            assert left[field] == right[field], field
        episodes = [copy.deepcopy(t['episode']) for t in (left, right)]
        for episode in episodes:
            episode['recipe'].pop('reward_config')
        assert episodes[0] == episodes[1]
        records.append(dict(family=left['episode']['id'], seeds=left['seeds'], same_instances=True))
    count = sum(len(record['seeds']) for record in records)
    assert count == protocol['training_episodes_per_branch']
    return dict(state='complete', episodes_per_branch=count,
                fork_weights_sha256=forks['control']['weights_sha256'],
                config_same_except_reward=True,
                source_plans_sha256={n: sha(root / n / 'capture/train/plan.json') for n in names},
                families=records, shared_training_pool=protocol['shared_training_pool'],
                critic_optimizer_preserved=protocol['critic_optimizer_preserved'],
                scope='Conditions and pinned CUDA fork receipts only; trajectories may differ. No numerical NN check or quality claim.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    save(args.root / 'training-comparability.json', result)
    print('Comparable conditions:', result['episodes_per_branch'], 'episodes per branch')

"""Audit v10 rewards on the closed two-actor Soldier smoke; no NN work."""
import argparse
import json
import math
from pathlib import Path

from audit_coop_paired_export import read, sha, lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    capture = read(root / 'report.json')
    assert capture['accepted'] and capture['source_unchanged'] and capture['native_source_unchanged']
    results, hashes = [], {}
    for role in range(2):
        directory = root / f'dataset-role-{role}-reward-v10'
        report = read(directory / 'report.json')
        assert report['paired_confirmed'] and report['paired_experimental_reward']
        assert not report['paired_training_ready']
        assert report['synchronous_confirmed'] and report['observed_reset_confirmed']
        for path, digest in report['paired_source_sha256'].items():
            assert sha(Path(path)) == digest
        rewards = list(lines(directory / 'rewards.jsonl'))
        steps = list(lines(directory / 'steps.jsonl'))
        assert len(rewards) == len(steps) == report['steps']
        available = [r for r in rewards if r['available']]
        assert len(available) == report['reward_steps']
        assert math.isclose(sum(r['score'] for r in available), report['reward_sum'], abs_tol=1e-10)
        provider_nav = []
        for step, reward in zip(steps, rewards):
            assert step['index'] == reward['step']
            if not reward['available']:
                continue
            components = reward['components']
            assert math.isfinite(reward['score'])
            assert math.isclose(sum(components.values()), reward['score'], abs_tol=1e-10)
            # Actual smoke: only role1 dealt damage/killed the Soldier.
            if role == 0:
                assert components['monster_damage'] == components['monster_kill'] == 0
            if step['owner'] == 'provider':
                provider_nav.append(components['navigation_potential'])
        results.append(dict(role=role, transitions=len(steps), available_rewards=len(available),
                            score=report['reward_sum'], provider_navigation_steps=len(provider_nav),
                            provider_navigation_sum=sum(provider_nav),
                            monster_damage_reward=sum(r['components']['monster_damage'] for r in available),
                            monster_kill_reward=sum(r['components']['monster_kill'] for r in available)))
        for path in directory.iterdir():
            if path.is_file():
                hashes[str(path)] = sha(path)
    assert math.isclose(results[1]['monster_damage_reward'], .3, abs_tol=1e-10)
    assert results[1]['monster_kill_reward'] == 5
    closure = dict(version='coop_navigation_reward_audit_v1', results=results,
                   source_sha256=hashes, ppo_trainable=False, training_performed=False,
                   scope='Offline experimental reward arithmetic and actor attribution on one closed capture; no model improvement claim.')
    (root / 'navigation-reward-verification.json').write_text(json.dumps(closure, indent=2), encoding='utf-8')
    print(json.dumps(results))


if __name__ == '__main__':
    main()

"""Audit v11 own-effect credit and joint death cost on the closed live smoke."""
import argparse
import json
import math
from pathlib import Path
from audit_coop_paired_export import read, sha, lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    closure = read(root/'joint-terminal-verification.json')
    assert closure['live_stop_verified'] and not closure['ppo_trainable']
    for path, digest in closure['source_sha256'].items():
        assert sha(Path(path)) == digest
    results, hashes = [], {}
    for role in range(2):
        directory = root/f'dataset-coop-reward-role-{role}'
        report = read(directory/'report.json')
        assert report['paired_confirmed'] and report['paired_experimental_reward']
        assert not report['paired_training_ready']
        assert report['reward_config']['version'] == 'combat_reward_v11'
        for path, digest in report['paired_source_sha256'].items():
            assert sha(Path(path)) == digest
        steps = list(lines(directory/'steps.jsonl'))
        rewards = list(lines(directory/'rewards.jsonl'))
        outcomes = list(lines(directory/'server_outcomes.jsonl'))
        assert len(steps) == len(rewards) == len(outcomes) == report['steps'] == report['reward_steps'] == 80
        assert all(r['available'] and math.isfinite(r['score']) for r in rewards)
        assert math.isclose(sum(r['score'] for r in rewards), report['reward_sum'], abs_tol=1e-10)
        for s, r, o in zip(steps, rewards, outcomes):
            assert s['index'] == r['step'] == o['step']
            assert math.isclose(sum(r['components'].values()), r['score'], abs_tol=1e-10)
            assert r['components']['monster_damage'] == r['components']['monster_kill'] == 0
            if s.get('joint_terminal'):
                assert s['next_observation']['identity']['frame'] == report['joint_death_boundary']['end_frame']
                assert len(o['joint_death_events']) == 1 and o['joint_death_events'][0]['target'] == 2
            else:
                assert not o.get('joint_death_events') and r['components']['peer_death'] == 0
        peer = sum(r['components']['peer_death'] for r in rewards)
        own = sum(r['components']['death'] for r in rewards)
        assert (peer, own) == ((-5, 0) if role == 0 else (0, -5))
        results.append(dict(role=role, reward_steps=len(rewards), score=report['reward_sum'],
                            peer_death_cost=peer, own_death_cost=own, monster_damage_reward=0, monster_kill_reward=0))
        for path in directory.iterdir():
            if path.is_file(): hashes[str(path)] = sha(path)
    result = dict(version='coop_joint_reward_audit_v1', results=results, source_sha256=hashes,
                  training_performed=False, ppo_trainable=False)
    (root/'joint-reward-verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(results))


if __name__ == '__main__': main()

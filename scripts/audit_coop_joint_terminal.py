"""Audit paired death terminals against native proof, excluding the late tail."""
import argparse
import json
from pathlib import Path
from audit_coop_paired_export import read, sha, lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    proof = read(root/'joint-boundary-proof.json')
    capture = read(root/'report.json')
    assert not proof['ppo_trainable']
    assert sha(Path(proof['source'])) == proof['source_sha256']
    for path, digest in proof['trace_sha256'].items():
        assert sha(Path(path)) == digest
    boundary = proof['joint_death_boundary']
    live_stop = capture.get('live_joint_stop_verified', False)
    if live_stop:
        assert capture['accepted'] and capture['stop_frame'] == boundary['end_frame']
        assert proof['native']['pairs'][-1]['steps'][0]['end_frame'] == boundary['end_frame']
        for name in ('PairLearner', 'PairLeader'):
            trace = list(lines(root/f'{name}.jsonl'))
            assert trace[-1]['terminal_observation_only'] and trace[-1]['frame'] == boundary['end_frame']
            assert len(trace) == capture['completed_pairs']+1
    outputs = []
    hashes = {}
    for role, suffix in enumerate(('dataset-joint-role-0', 'dataset-joint-role-1-v3')):
        directory = root/suffix
        report = read(directory/'report.json')
        assert report['joint_death_boundary'] == boundary
        assert report['paired_confirmed'] and not report['paired_training_ready']
        assert report['command_proof']['accepted'] and report['command_proof']['matched'] == len(proof['native']['pairs'])
        for path, digest in report['paired_source_sha256'].items():
            assert sha(Path(path)) == digest
        steps = list(lines(directory/'steps.jsonl'))
        assert len(steps) == report['steps'] == boundary['end_frame']-proof['native']['release']['frame']
        assert sum(s['terminal'] for s in steps) == report['terminals'] == 1
        assert steps[-1]['joint_terminal'] == boundary
        assert steps[-1]['reason'] == 'coop_participant_death' and not steps[-1]['truncated']
        assert steps[-1]['next_observation']['identity']['frame'] == boundary['end_frame']
        assert steps[-1]['next_observation']['health'] == boundary['health_after'][role]
        assert all(s['observation']['identity']['frame'] < boundary['end_frame'] for s in steps)
        effects = list(lines(directory/'server_outcomes.jsonl'))
        assert len(effects) == len(steps)
        # This learner killed Soldier after peer death. That late kill must vanish.
        assert sum(o['monster_kills'] for o in effects) == 0
        if role == 1:
            assert sum(o['deaths'] for o in effects) == 1
        outputs.append(dict(role=role, steps=len(steps), terminal_frame=boundary['end_frame'],
                            terminal_health=boundary['health_after'][role], late_monster_kills=0))
        for path in directory.iterdir():
            if path.is_file(): hashes[str(path)] = sha(path)
    result = dict(version='coop_joint_terminal_audit_v1', results=outputs, source_sha256=hashes,
                  ppo_trainable=False, training_performed=False, live_stop_verified=live_stop)
    (root/'joint-terminal-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(outputs))


if __name__ == '__main__': main()

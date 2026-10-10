"""Observed command/movement metrics for completed paired pilot captures.

Attack means sent button, not a discharged round; clear-target absence is not
proof of a miss. Position deltas measure actual motion, not requested movement.
No neural inference or training.
"""
import argparse
import json
import math
import pathlib
import statistics
from process_combat_architecture_pool import read, save, sha


def episode(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    selected = []
    for r in rows:
        c = r.get('combat_policy', {})
        o = c.get('observation', {})
        if o.get('identity', {}).get('frame', 0) < 100 or 'sent_command' not in r:
            continue
        if c.get('selection', {}).get('owner') == 'provider':
            selected.append((r, o))
    by_frame = {r.get('combat_policy', {}).get('observation', {}).get('identity', {}).get('frame'):r
                for r in rows}
    attack_frames, blocked_target_attack, stationary, longest = 0, 0, 0, 0
    run = 0
    distance, turn = 0., 0.
    first_attack = None
    previous_yaw = None
    previous_frame = None
    measured = 0
    for r, o in selected:
        cmd = r['sent_command']
        f = o['identity']['frame']
        attack = bool(cmd['Buttons'] & 1)
        if attack:
            attack_frames += 1
            if first_attack is None: first_attack = f-100
            if not any(e.get('clear_shot') is True for e in o['enemies']):
                blocked_target_attack += 1
        if previous_yaw is not None and f == previous_frame+1:
            turn += abs((cmd['Yaw']-previous_yaw+32768) % 65536-32768)*360/65536
        else:
            run = 0
        previous_yaw = cmd['Yaw']
        previous_frame = f
        nxt = by_frame.get(f+1, {}).get('combat_policy', {}).get('observation')
        if nxt is None:
            run = 0
            continue
        step = math.dist(o['position'], nxt['position'])
        if step > 128:  # A teleport is not locomotion.
            run = 0
            continue
        distance += step
        measured += 1
        if step <= 1:
            stationary += 1
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return dict(provider_frames=len(selected), attack_command_frames=attack_frames,
        first_attack_delay_game_seconds=None if first_attack is None else first_attack/10,
        attack_without_observed_clear_target_frames=blocked_target_attack,
        measured_motion_frames=measured, stationary_frames=stationary,
        longest_stationary_game_seconds=longest/10,
        travelled_units=distance, absolute_yaw_turn_degrees=turn, trace_sha256=sha(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evaluation', type=pathlib.Path, required=True)
    a = ap.parse_args()
    summaries = {}
    for branch in ('before', 'after'):
        capture = read(a.evaluation/branch/'report.json')
        assert capture['accepted'] == capture['completed']
        cases = [dict(seed=r['seed'], **episode(pathlib.Path(r['root'])/'PairLearner.jsonl'))
                 for r in capture['results']]
        delays = [c['first_attack_delay_game_seconds'] for c in cases if c['first_attack_delay_game_seconds'] is not None]
        summaries[branch] = dict(cases=cases, median_first_attack_delay_game_seconds=statistics.median(delays) if delays else None,
            no_attack_episodes=sum(c['attack_command_frames']==0 for c in cases),
            stationary_frames=sum(c['stationary_frames'] for c in cases),
            measured_motion_frames=sum(c['measured_motion_frames'] for c in cases),
            attack_without_observed_clear_target_frames=sum(c['attack_without_observed_clear_target_frames'] for c in cases))
    save(a.evaluation/'behavior-comparison.json', dict(version='coop_observed_behavior_comparison_v1',
        branches=summaries, scope=__doc__))


if __name__ == '__main__':
    main()

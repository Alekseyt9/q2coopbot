"""Offline first-life motion and horizontal aim diagnostics; no reward changes."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def summarize(rows):
    result = dict(steps=0, motion_pairs=0, moving_requested=0,
                  requested_move_stationary=0, longest_stationary_run=0,
                  move_zeroed=0, path_length=0., visible_target_steps=0,
                  attack_steps=0, visible_attack_steps=0,
                  applied_visible_attack_yaw_error_gt15=0)
    run = 0
    previous_frame = None
    for s in rows:
        o = s['observation']
        if s['owner'] != 'provider' or o['identity']['life'] != 1:
            run = 0
            continue
        frame = o['identity']['frame']
        if previous_frame is None or frame != previous_frame + 1:
            run = 0
        previous_frame = frame
        result['steps'] += 1
        a, ap = s['action'], s['applied_action']
        requested = math.hypot(a['forward'], a['side']) > .05
        result['moving_requested'] += requested
        result['move_zeroed'] += requested and math.hypot(ap['forward'], ap['side']) < .02
        n = s.get('next_observation')
        if n and n['identity']['life'] == 1 and n['identity']['frame'] == frame + 1:
            distance = math.dist(o['position'], n['position'])
            result['motion_pairs'] += 1
            result['path_length'] += distance
            stalled = requested and distance < .5
            result['requested_move_stationary'] += stalled
            run = run + 1 if stalled else 0
            result['longest_stationary_run'] = max(result['longest_stationary_run'], run)
        else:
            run = 0
        targets = [e for e in o['enemies'] if e.get('clear_shot') is True]
        result['visible_target_steps'] += bool(targets)
        result['attack_steps'] += bool(ap['attack'])
        if targets and ap['attack']:
            result['visible_attack_steps'] += 1
            target = min(targets, key=lambda e: e['distance'])
            r = target['relative']
            target_yaw = math.degrees(math.atan2(r[1], r[0]))
            applied_yaw = o['view_angles'][1] * 360 / 65536 + ap['yaw_delta_degrees']
            error = abs((target_yaw - applied_yaw + 180) % 360 - 180)
            result['applied_visible_attack_yaw_error_gt15'] += error > 15
    result['path_length'] = round(result['path_length'], 3)
    return result


def analyze(batch):
    report_bytes = (batch / 'report.json').read_bytes()
    report = json.loads(report_bytes)
    if not report['capture_complete'] or not report['provenance_valid']:
        raise ValueError('Invalid batch capture/provenance')
    results = []
    for episode in report['results']:
        if not all(episode[k] for k in ['capture_valid', 'dispatch_valid', 'seed_confirmed']):
            raise ValueError('Invalid episode')
        path = Path(episode['root']) / 'dataset' / 'steps.jsonl'
        data = path.read_bytes()
        metrics = summarize(json.loads(line) for line in data.splitlines() if line)
        results.append(dict(seed=episode['seed'], steps_path=str(path),
                            steps_sha256=hashlib.sha256(data).hexdigest(), metrics=metrics))
    return dict(batch=str(batch), report_sha256=hashlib.sha256(report_bytes).hexdigest(), episodes=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', action='append', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Fresh output required')
    result = dict(version='combat_diagnostics_v1',
                  scope='Provider first life only. Stationary: requested planar magnitude >0.05, next-frame displacement <0.5 units. Zeroed: applied planar magnitude <0.02. Yaw: nearest clear target at current observation, after applied yaw delta; excludes pitch, projectile travel and hit attribution. Longest run requires consecutive frames.',
                  batches=[analyze(p.resolve()) for p in args.batch])
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(args.out)

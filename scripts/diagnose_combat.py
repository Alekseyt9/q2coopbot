"""Offline first-life motion and observed target-origin aim diagnostics."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def origin_aim_error(observation, applied, target):
    # Observation v3 omits enemy bbox/aim point. Use the observed origin,
    # explicitly not a reconstructed upper-body point or shot hit test.
    x, y, z = target['relative']
    z -= -2 if observation['ducked'] else 22  # quake.Snapshot.EyePoint
    length = math.sqrt(x*x + y*y + z*z)
    if length <= 1e-9:
        return None
    yaw = observation['view_angles'][1] * 360 / 65536 + applied['yaw_delta_degrees']
    pitch = observation['view_angles'][0] * 360 / 65536 + applied['pitch_delta_degrees']
    target_pitch = -math.degrees(math.atan2(z, math.hypot(x, y)))
    pitch_error = abs((target_pitch - pitch + 180) % 360 - 180)
    yr, pr = math.radians(yaw), math.radians(pitch)
    dot = (math.cos(pr)*math.cos(yr)*x + math.cos(pr)*math.sin(yr)*y - math.sin(pr)*z) / length
    angle = math.degrees(math.acos(max(-1., min(1., dot))))
    return pitch_error, angle, pitch


def bbox_aim_error(observation, applied, target):
    solid = target.get('observed_solid')
    if solid is None or solid in (0,31) or solid & 31 == 0:
        return None
    bottom = -((solid >> 5) & 31) * 8
    top = ((solid >> 10) & 63) * 8 - 32
    if top <= bottom:
        return None
    z = (top+bottom)/2 if top-bottom < 16 else min(max(22,bottom+8),top-8)
    relative = list(target['relative'])
    relative[2] += z
    return origin_aim_error(observation,applied,dict(relative=relative))


def summarize(rows):
    result = dict(steps=0, motion_pairs=0, moving_requested=0,
                  requested_move_stationary=0, longest_stationary_run=0,
                  move_zeroed=0, path_length=0., visible_target_steps=0,
                  attack_steps=0, visible_attack_steps=0,
                  applied_visible_attack_yaw_error_gt15=0,
                  origin_aim_samples=0, origin_pitch_error_sum=0.,
                  origin_angle_error_sum=0., origin_pitch_error_gt15=0,
                  origin_angle_error_gt15=0, applied_attack_pitch_near_limit=0,
                  pitch_limit_interventions=0, bbox_aim_samples=0,
                  bbox_pitch_error_gt15=0, bbox_angle_error_gt15=0,
                  parasite_range_samples=0, parasite_within288=0,
                  parasite_range_sum=0., parasite_motion_pairs=0,
                  parasite_range_increasing=0, parasite_range_decreasing=0,
                  parasite_actual_retreat=0, parasite_near_stationary=0)
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
        result['pitch_limit_interventions'] += any(
            c.get('component') == 'pitch' and c.get('reason') == 'protocol_pitch_limit'
            for c in s.get('interventions', []))
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
        parasites=[e for e in targets if e['class']=='monster_parasite']
        if parasites:
            enemy=min(parasites,key=lambda e:e['distance'])
            x,y=enemy['relative'][:2]
            distance=math.hypot(x,y)
            result['parasite_range_samples']+=1
            result['parasite_range_sum']+=distance
            result['parasite_within288']+=distance<288
            if n and n['identity']['life']==1 and n['identity']['frame']==frame+1:
                matching=[e for e in n.get('enemies',[]) if e['id']==enemy['id'] and e['class']==enemy['class'] and e.get('clear_shot') is True]
                if len(matching)==1:
                    following=math.hypot(*matching[0]['relative'][:2])
                    result['parasite_motion_pairs']+=1
                    result['parasite_range_increasing']+=following-distance>.5
                    result['parasite_range_decreasing']+=following-distance<-.5
                dx,dy=n['position'][0]-o['position'][0],n['position'][1]-o['position'][1]
                if distance>1e-9:
                    result['parasite_actual_retreat']+=-(dx*x+dy*y)/distance>.5
                result['parasite_near_stationary']+=distance<288 and math.hypot(dx,dy)<.5
        if targets and ap['attack']:
            result['visible_attack_steps'] += 1
            target = min(targets, key=lambda e: e['distance'])
            r = target['relative']
            target_yaw = math.degrees(math.atan2(r[1], r[0]))
            applied_yaw = o['view_angles'][1] * 360 / 65536 + ap['yaw_delta_degrees']
            error = abs((target_yaw - applied_yaw + 180) % 360 - 180)
            result['applied_visible_attack_yaw_error_gt15'] += error > 15
            aim = origin_aim_error(o, ap, target)
            bbox = bbox_aim_error(o,ap,target)
            if bbox is not None:
                result['bbox_aim_samples'] += 1
                result['bbox_pitch_error_gt15'] += bbox[0] > 15
                result['bbox_angle_error_gt15'] += bbox[1] > 15
            if aim is not None:
                pitch_error, angle, pitch = aim
                result['origin_aim_samples'] += 1
                result['origin_pitch_error_sum'] += pitch_error
                result['origin_angle_error_sum'] += angle
                result['origin_pitch_error_gt15'] += pitch_error > 15
                result['origin_angle_error_gt15'] += angle > 15
                result['applied_attack_pitch_near_limit'] += abs(pitch) >= 88
    result['path_length'] = round(result['path_length'], 3)
    result['parasite_range_sum']=round(result['parasite_range_sum'],3)
    for key in ['origin_pitch_error_sum', 'origin_angle_error_sum']:
        result[key] = round(result[key], 3)
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
    result = dict(version='combat_diagnostics_v4',
                  scope='Provider first life only. Stationary: requested planar magnitude >0.05, next-frame displacement <0.5 units. Zeroed: applied planar magnitude <0.02. Yaw: nearest clear target at current observation, after applied yaw delta. Origin pitch/3D angle: same target observed origin from current eye (standing +22, ducked -2), after applied angle deltas. Origin metrics ignore bounds. Bbox metrics use optional observed_solid and the Go AimPoint packed-bbox clamp; unavailable/non-bbox solids are masked. These angles exclude projectile travel, firing muzzle offset, target motion, and hit attribution; not shot accuracy. Near pitch limit: abs(applied pitch)>=88 degrees on nondegenerate visible attack samples. Parasite: horizontal range to nearest clear target, within288 is a maneuver prior not guaranteed safety; increasing/decreasing range threshold 0.5 units per matched-id next frame; actual retreat is self displacement projected away from current target >0.5, near stationary is planar displacement <0.5 inside288. Longest stationary run requires consecutive frames.',
                  batches=[analyze(p.resolve()) for p in args.batch])
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(args.out)

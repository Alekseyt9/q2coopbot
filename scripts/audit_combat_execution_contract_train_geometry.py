"""Reconstruct native MG geometry in already sealed accepted TRAIN windows; no NN execution."""
import argparse
import collections
import json
import math
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha
from report_combat_machinegun_hits import scan, vector
from report_combat_machinegun_aim import measure, angular
from report_registered_machinegun_outcomes import outcome
from export_combat_aim_feedback import poses
from audit_combat_hitbox_aim import intersects, ballistic_basis, spread_ray


def geometry(shot, step, observed, pose):
    fields = shot['fields']
    start, aim, impact = (vector(fields[name]) for name in ('start', 'aim', 'impact'))
    spread = vector(fields['spread'], 2)
    lower = [a+b for a, b in zip(pose['origin'], pose['mins'])]
    upper = [a+b for a, b in zip(pose['origin'], pose['maxs'])]
    actual = [a-b for a, b in zip(impact, start)]
    reconstructed = spread_ray(aim, spread)
    error = angular(actual, reconstructed) if math.dist(start, impact) > 2 else None
    assert error is None or error < .1, 'Native spread ray reconstruction mismatch'
    effective = ballistic_basis(aim)[0]
    nominal = intersects(start, effective, lower, upper)
    spread_hit = intersects(start, reconstructed, lower, upper, maximum=1.)
    segment = intersects(start, actual, lower, upper, maximum=1.)
    contact = bool(int(fields['target']) == pose['entity'] and int(fields['damageable']) and int(fields['health_before']) > 0)
    assert not contact or segment, 'Native contact outside selected shot-time AABB'
    center = [(a+b)/2 for a, b in zip(lower, upper)]
    bottom, top = pose['mins'][2], pose['maxs'][2]
    height = (top+bottom)/2 if top-bottom < 16 else min(max(22, bottom+8), top-8)
    point = list(pose['origin']); point[2] += height
    ideal_upper = intersects(start, spread_ray([a-b for a, b in zip(point, start)], spread), lower, upper, maximum=1.)
    ideal_center = intersects(start, spread_ray([a-b for a, b in zip(center, start)], spread), lower, upper, maximum=1.)
    above = below = False
    horizontal = sum(v*v for v in effective[:2])
    distance = sum((c-s)*d for c, s, d in zip(center[:2], start[:2], effective[:2]))/horizontal if horizontal > 1e-12 else -1
    if not nominal and distance > 0:
        height_at_range = start[2]+distance*effective[2]
        above, below = height_at_range > upper[2], height_at_range < lower[2]
    counters = collections.Counter(geometry_shots=1, nominal_intersects=int(nominal), spread_intersects=int(spread_hit),
        segment_intersects=int(segment), selected_native_contact=int(contact),
        nominal_hit_spread_miss=int(nominal and not spread_hit), nominal_miss_spread_hit=int(not nominal and spread_hit),
        nominal_miss_above_at_center_range=int(above), nominal_miss_below_at_center_range=int(below),
        ideal_upper_bbox_only=int(ideal_upper), ideal_center_bbox_only=int(ideal_center),
        ideal_center_only_bbox=int(ideal_center and not ideal_upper), ideal_upper_only_bbox=int(ideal_upper and not ideal_center))
    category = ('selected_native_contact' if contact else 'spread_ray_misses_selected_bbox' if not spread_hit else
                'spread_ray_hits_bbox_but_actual_segment_stops_before' if not segment else 'segment_intersects_without_selected_contact')
    counters['category:'+category] += 1
    counters['category:'+category+':'+outcome(shot)] += 1
    observation = step['observation']
    enemy = next(enemy for enemy in observation['enemies'] if enemy['id'] == pose['entity'] and
                 enemy.get('observed_track', 0) == step['action'].get('target_track', 0))
    origin = [a+b for a, b in zip(observation['position'], enemy['relative'])]
    counters['observed_origin_error_units_sum'] += math.dist(origin, pose['origin'])
    counters['native_reconstruction_verified'] += error is not None
    return counters, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workflow', 'shots', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), 'Fresh output required'
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = pins.get(str(path)) or sha(path)
        assert expected is None or expected == digest, str(path)
        pins[str(path)] = digest
        return path

    for name in ('audit_combat_execution_contract_train_geometry.py', 'audit_combat_hitbox_aim.py',
                 'export_combat_aim_feedback.py', 'report_combat_machinegun_hits.py', 'report_combat_machinegun_aim.py',
                 'report_registered_machinegun_outcomes.py', 'report_combat_selected_target_aim.py', 'native_projectile_window.py'):
        pin(Path(__file__).with_name(name))
    seal = read(pin(args.shots/'complete.json'))
    previous = read(pin(args.shots/'report.json', seal['report_sha256']))
    assert seal['state'] == previous['state'] == 'complete' and previous['train_cases'] == 192 and previous['rows'] == 21201
    for path, digest in previous['inputs'].items():
        pin(path, digest)
    root = args.workflow.resolve()
    assert str(root/'complete.json') in previous['inputs']
    references = {(case['arm'], case['round'], case['seed']): case for case in previous['cases']}
    cases, variants, rounds, maximum_error = [], {}, [], 0.
    for arm in ('legacy', 'checked'):
        arm_counts = collections.Counter()
        for number in range(1, 5):
            round_counts = collections.Counter()
            processing = root/f'round-{number:02d}'/arm/'corpus/processing-train'
            report = read(pin(processing/'report.json', previous['inputs'][str(processing/'report.json')]))
            for case in report['cases']:
                reference = references[arm, number, case['seed']]
                directory = processing/f"case-{case['job']:04d}"
                meta_path = directory/'cuda/report.json'
                meta = read(pin(meta_path, previous['inputs'][str(meta_path)]))
                replay = pin(directory/'cuda/rollout.jsonl', meta['rollout_sha256'])
                with replay.open(encoding='utf-8-sig') as stream:
                    accepted = {(row['index'], row['frame']) for row in map(json.loads, stream)}
                steps = pin(directory/'native/replay-0/steps.jsonl', meta['source_sha256'][str(directory/'native/replay-0/steps.jsonl')])
                windows = {}
                with steps.open(encoding='utf-8-sig') as stream:
                    for step in map(json.loads, stream):
                        if (step['index'], step['observation']['identity']['frame']) in accepted:
                            key = tuple(step['native_step'][name] for name in ('spawncount', 'begin_frame', 'actor', 'sequence'))
                            assert key not in windows
                            windows[key] = step
                assert len(windows) == len(accepted) == reference['rows']
                log = next(Path(path) for path in meta['source_sha256'] if path.endswith('server.log'))
                pin(log, meta['source_sha256'][str(log)])
                with log.open(encoding='utf-8-sig') as stream:
                    shots = scan(stream, set(windows))
                with log.open(encoding='utf-8-sig') as stream:
                    truth = poses(stream, shots)
                counters = collections.Counter(cases=1, accepted_rows=len(accepted))
                for shot in shots:
                    if not shot['eligible']:
                        continue
                    counters['eligible_shots'] += 1
                    step = windows[shot['window']]
                    observed = measure(shot, step)
                    if observed['status'] != 'measured':
                        counters['unmeasured:'+observed['status']] += 1
                        continue
                    pose = truth.get((shot['generation'], int(shot['fields']['shot'])), {}).get(observed['entity'])
                    if pose is None:
                        counters['selected_target_missing_living_pose'] += 1
                        if int(shot['fields']['target']) == observed['entity'] and int(shot['fields']['health_before']) <= 0:
                            counters['missing_pose_confirmed_selected_corpse_contact'] += 1
                        continue
                    if int(shot['fields']['muzzle_blocked']) or int(shot['fields']['water']):
                        counters['muzzle_blocked_or_water'] += 1
                        continue
                    values, error = geometry(shot, step, observed, pose)
                    counters.update(values)
                    if error is not None:
                        maximum_error = max(maximum_error, error)
                expected = reference['groups'].get('all', {}).get('shots', 0)
                assert counters['eligible_shots'] == expected
                cases.append(dict(arm=arm, round=number, seed=case['seed'], counts=dict(counters)))
                round_counts.update(counters)
            rounds.append(dict(arm=arm, round=number, counts=dict(round_counts)))
            arm_counts.update(round_counts)
            print(dict(arm=arm, round=number, geometry_shots=round_counts['geometry_shots']), flush=True)
        variants[arm] = dict(arm_counts)
    assert len(cases) == 192 and sum(v['eligible_shots'] for v in variants.values()) == 6122
    assert all(sha(path) == digest for path, digest in pins.items())
    args.out.mkdir(parents=True)
    save(args.out/'report.json', dict(state='complete', version='combat_execution_contract_train_geometry_v1',
        variants=variants, rounds=rounds, cases=cases, max_spread_reconstruction_error_degrees=maximum_error, inputs=pins,
        neural_execution=False, new_games=False, training_performed=False, promotion=False,
        scope='Only accepted TRAIN rows from the sealed 192-case experiment. Native selected living shot-time AABB, '
              'integer ballistic direction and logged spread; actual selected contact checks finite-segment intersection. '
              'Categories partition measured straight shots; auxiliary geometry counters overlap. Earlier contact may '
              'be another actor or geometry. Ideal-point comparisons ignore BSP/occluders and assume exact fresh recoil '
              'compensation; they are not a live teacher or a success probability. No hidden policy inputs or fitting.'))
    save(args.out/'complete.json', dict(state='complete', report_sha256=sha(args.out/'report.json')))


if __name__ == '__main__':
    main()

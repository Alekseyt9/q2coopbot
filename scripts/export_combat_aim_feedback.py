"""Offline shot-time labels. Native truth must never become policy features."""
import argparse
import collections
import gzip
import json
import math
import pathlib
import re

from native_projectile_window import records
from process_combat_architecture_pool import read, save, sha
from report_combat_machinegun_hits import FIRE, fields, scan, vector
from report_combat_machinegun_aim import angular, measure
from report_combat_selected_target_aim import wrap
from native_combat_release import release_boundary

POSE = re.compile(r'^sv_test_aim_pose spawncount=(-?\d+) server_frame=(\d+) g_test_aim_pose version=1 (.*)$')


def poses(stream, shots):
    """Match each pose to its fire event, command window and both clocks."""
    by_shot = {(s['generation'], int(s['fields']['shot'])): s for s in shots}
    result = {}; previous = {}; active_shot = None
    for window, line in records(stream):
        fire = FIRE.fullmatch(line)
        if fire:
            active_shot = (int(fire[1]), int(fields(fire[3])['shot']))
        if not line.startswith('sv_test_aim_pose '):
            continue
        match = POSE.fullmatch(line)
        assert match, 'Malformed shot-time pose'
        generation, frame, tail = match.groups()
        f = fields(tail)
        assert set(f) == set('map frame shot actor entity class health solid origin mins maxs velocity'.split())
        key = (int(generation), int(f['shot']))
        assert key in by_shot and key == active_shot, 'Pose without preceding fire'
        shot = by_shot[key]
        assert shot['window'] == window and shot['server_frame'] == int(frame)
        assert all(f[k] == shot['fields'][k] for k in ('map', 'frame', 'shot', 'actor'))
        assert int(f['entity']) > 0 and int(f['health']) > 0
        # SVF_MONSTER also includes turret_driver; classname is not a type prefix contract.
        assert int(f['solid']) >= 0 and f['class']
        pose = dict(entity=int(f['entity']), monster_class=f['class'], health=int(f['health']),
                    solid=int(f['solid']), **{k: vector(f[k]) for k in ('origin', 'mins', 'maxs', 'velocity')})
        assert all(a < b for a, b in zip(pose['mins'], pose['maxs']))
        # Quake monster AI can move origin directly without setting edict.velocity.
        # Keep the finite difference explicitly separate from instantaneous velocity.
        identity = (int(generation), pose['entity'], pose['monster_class'])
        old = previous.get(identity); game_frame = int(f['frame'])
        pose['shot_interval_horizontal_speed'] = None
        pose['shot_interval_frames'] = None
        if old:
            dt = game_frame-old[0]
            assert dt > 0
            pose['shot_interval_frames'] = dt
            pose['shot_interval_horizontal_speed'] = math.dist(pose['origin'][:2], old[1][:2])/(dt*.1)
        previous[identity] = (game_frame, pose['origin'])
        members = result.setdefault(key, {})
        assert pose['entity'] not in members, 'Duplicate entity pose for shot'
        members[pose['entity']] = pose
    return result


def label(shot, pose, step):
    o = step['observation']
    entity = step['action']['target_entity']
    assert entity == pose['entity']
    enemy = next(e for e in o['enemies'] if e['id'] == entity and
                 e.get('observed_track', 0) == step['action'].get('target_track', 0))
    f = shot['fields']
    muzzle, aim, view, recoil = (vector(f[k]) for k in ('start', 'aim', 'view', 'recoil'))
    # Same body-point convention as the existing observed-bbox diagnostic.
    bottom, top = pose['mins'][2], pose['maxs'][2]
    height = (top + bottom) / 2 if top - bottom < 16 else min(max(22, bottom + 8), top - 8)
    point = list(pose['origin']); point[2] += height
    direction = [p - m for p, m in zip(point, muzzle)]
    assert sum(x*x for x in direction) > 1e-12
    yaw = math.degrees(math.atan2(direction[1], direction[0]))
    pitch = -math.degrees(math.atan2(direction[2], math.hypot(*direction[:2])))
    observed_position = [a+b for a, b in zip(o['position'], enemy['relative'])]
    observed = measure(shot, step)
    assert observed['status'] == 'measured'
    source_view = [v*360/65536 for v in o['view_angles']]
    # Offline desired command before fresh native recoil, not a runtime servo.
    desired_view = [pitch-recoil[0], wrap(yaw-recoil[1])]
    return dict(entity=entity, track=step['action'].get('target_track', 0),
        native_pose=pose, native_muzzle=muzzle, native_aim=aim, native_recoil=recoil,
        desired_point=point, desired_view_degrees=desired_view,
        desired_command_delta_degrees=[wrap(desired_view[i]-source_view[i]) for i in range(2)],
        correction_from_executed_view_degrees=[wrap(desired_view[i]-view[i]) for i in range(2)],
        native_bodypoint_error_degrees=angular(aim, direction),
        observed_bodypoint_error_degrees=observed['post_recoil_degrees'],
        observed_origin_error_units=math.dist(observed_position, pose['origin']),
        distance=math.sqrt(sum(x*x for x in direction)),
        own_speed=math.hypot(*vector(f['velocity'])[:2]), target_speed=math.hypot(*pose['velocity'][:2]),
        target_interval_speed=pose['shot_interval_horizontal_speed'],
        selected_target_damage=observed['selected_target_damage'],
        muzzle_blocked=bool(int(f['muzzle_blocked'])), water=bool(int(f['water'])))


def summarize(rows):
    def stats(items):
        return dict(shots=len(items), **{k: sum(r[k] for r in items)/len(items) if items else None
            for k in ('native_bodypoint_error_degrees', 'observed_bodypoint_error_degrees',
                      'observed_origin_error_units', 'own_speed', 'target_speed', 'distance')})
    return dict(all=stats(rows), strata={name: stats([r for r in rows if predicate(r)])
        for name, predicate in [
            ('own_stationary', lambda r: r['own_speed'] < 10),
            ('own_moving', lambda r: r['own_speed'] >= 10),
            ('own_fast', lambda r: r['own_speed'] >= 100),
            ('target_velocity_at_least_10', lambda r: r['target_speed'] >= 10),
            ('target_interval_moving', lambda r: (r['target_interval_speed'] or 0) >= 10),
            ('both_interval_moving', lambda r: r['own_speed'] >= 10 and (r['target_interval_speed'] or 0) >= 10),
            ('distance_below_128', lambda r: r['distance'] < 128),
            ('distance_128_to_256', lambda r: 128 <= r['distance'] < 256),
            ('distance_at_least_256', lambda r: r['distance'] >= 256),
            ('recoil_above_6', lambda r: abs(r['native_recoil'][0]) > 6),
            ('muzzle_blocked', lambda r: r['muzzle_blocked'])]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    args = ap.parse_args()
    root, out = args.corpus.resolve(), args.out.resolve()
    assert not out.exists(), 'Fresh output required'
    pins = {}

    def pin(path, expected=None):
        path = pathlib.Path(path)
        digest = sha(path)
        assert expected is None or digest == expected, 'Input receipt mismatch: ' + str(path)
        if str(path) in pins:
            assert pins[str(path)] == digest
        pins[str(path)] = digest
        return path

    for name in ('native_combat_release.py','native_projectile_window.py','report_combat_machinegun_hits.py','report_combat_machinegun_aim.py'):
        pin(pathlib.Path(__file__).with_name(name))

    seal = read(pin(root/'complete.json'))
    assert seal['state'] == 'complete' and not seal['training_performed']
    protocol = read(pin(root/'protocol.json', seal['protocol_sha256']))
    assert not protocol['diagnostic_intervention']
    for binding in protocol['inputs']:
        pin(binding['path'], binding['sha256'])
    pool = read(pin(root/'pool/report.json', seal['pool_sha256']))
    assert pool['state'] == 'complete' and pool['source_unchanged']
    assert pool['usable_captures'] == len(pool['jobs']) == seal['native_cases'] == seal['cuda_cases']
    for binding in pool['plans']:
        plan = read(pin(binding['path'], binding['sha256']))
        assert not plan.get('diagnostic_nominal_aim')
    out.mkdir(parents=True)
    try:
        summaries = {}; seen_seeds = set()
        for processed in seal['processed']:
            split = processed['label']
            assert split in ('train', 'development')
            report_path = pin(processed['report'], processed['sha256'])
            report = read(report_path)
            assert report['state'] == 'complete' and report['device'] == 'cuda' and not report['training_performed']
            assert report['pool_sha256'] == seal['pool_sha256']
            counters = collections.Counter(); labels = []; seeds = set()
            destination = out/(split+'.jsonl.gz')
            with gzip.open(destination, 'wt', encoding='utf-8', compresslevel=6) as output:
                for case in report['cases']:
                    seed = case['seed']; assert seed not in seen_seeds
                    seen_seeds.add(seed); seeds.add(seed)
                    assert case['split'] == ('train' if split == 'train' else 'validation')
                    job = next(j for j in pool['jobs'] if j['job'] == case['job'])
                    assert job['seed'] == seed and job['mode'] == 'learned'
                    directory = report_path.parent/f"case-{case['job']:04d}"
                    capture = read(pin(pathlib.Path(job['root'])/'report.json'))
                    assert capture['capture_complete'] and capture['provenance_valid'] and len(capture['results']) == 1
                    log = pin(pathlib.Path(capture['results'][0]['root'])/'server.log')
                    manifest = read(pin(pathlib.Path(job['root'])/'manifest.json'))
                    with log.open(encoding='utf-8-sig') as stream:
                        generation, release, _ = release_boundary(stream, manifest, seed)
                    raw = pin(directory/'native/replay-0/steps.jsonl')
                    replay = pin(directory/'cuda/rollout.jsonl')
                    by_index = {r['index']: r for r in map(json.loads, replay.read_text(encoding='utf-8-sig').splitlines())}
                    windows = {}
                    for step in map(json.loads, raw.read_text(encoding='utf-8-sig').splitlines()):
                        n = step.get('native_step'); o = step['observation']; execution = step.get('server_execution', {})
                        if not (step['owner'] == 'provider' and n and o['identity']['life'] == 1 and o['identity']['frame'] >= release and o['health'] > 0
                                and execution.get('matched') and execution.get('window_exclusive') and not execution.get('recovery_commands')):
                            continue
                        row = by_index[step['index']]
                        assert row['seed'] == seed and row['frame'] == o['identity']['frame'] and row['sample'] == step['sample']
                        assert len(row['features']) == 1147 and all(math.isfinite(x) for x in row['features'])
                        assert n['spawncount'] == o['identity']['spawncount'] and n['actor'] == o['identity']['actor']
                        assert n['spawncount'] == generation
                        key = (n['spawncount'], n['begin_frame'], n['actor'], n['sequence'])
                        assert key not in windows; windows[key] = (step, row)
                    with log.open(encoding='utf-8-sig') as stream:
                        shots = scan(stream, set(windows))
                    with log.open(encoding='utf-8-sig') as stream:
                        shot_poses = poses(stream, shots)
                    for shot in shots:
                        if not shot['eligible']:
                            continue
                        counters['eligible_shots'] += 1
                        step, row = windows[shot['window']]
                        observed = measure(shot, step)
                        if observed['status'] != 'measured':
                            counters[observed['status']] += 1; continue
                        assert step['observation']['identity']['map'] == shot['fields']['map']
                        pose = shot_poses.get((shot['generation'], int(shot['fields']['shot'])), {}).get(observed['entity'])
                        if pose is None:
                            counters['selected_target_without_living_native_pose'] += 1; continue
                        item = label(shot, pose, step); labels.append(item)
                        counters['labelled_shots'] += 1
                        output.write(json.dumps(dict(version='combat_aim_feedback_labels_v2', split=split,
                            seed=seed, index=row['index'], frame=row['frame'], native_window=shot['window'],
                            native_shot=int(shot['fields']['shot']), features=row['features'], sample=row['sample'],
                            offline_label=item), separators=(',', ':'), allow_nan=False)+'\n')
            assert labels, 'No shot-time labels'
            summaries[split] = dict(cases=len(seeds), seeds=sorted(seeds), counts=dict(counters),
                coverage=summarize(labels), output=str(destination), sha256=sha(destination))
        for path, digest in pins.items():
            assert sha(path) == digest, 'Input changed during export'
        save(out/'report.json', dict(state='complete', version='combat_aim_feedback_labels_v2',
            corpus=str(root), inputs=[dict(path=p, sha256=h) for p, h in pins.items()], splits=summaries,
            script_sha256=sha(__file__), release_boundary='native_start_network_frame_v1', training_performed=False, promotion=False,
            scope='Exact native MG shot-time living target pose, joined only to explicitly selected observed entity/track and exclusive command. Original 1147 client features only; native labels separate. Nominal body-point geometry is not a visibility guarantee or an optimal-action teacher. No spread cancellation. Development is not final evaluation.'))
        print(json.dumps({k: dict(cases=v['cases'], counts=v['counts'], coverage=v['coverage']) for k, v in summaries.items()}))
    except BaseException as error:
        save(out/'failed.json', dict(state='failed', error=str(error)))
        raise


if __name__ == '__main__':
    main()

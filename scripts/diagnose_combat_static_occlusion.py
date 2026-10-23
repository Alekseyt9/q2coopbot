"""Observed-pose static BSP occlusion on sealed randomized TRAIN MG shots."""
import argparse
import collections
import gzip
import json
import math
import subprocess
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha
from process_combat_architecture_pool import compact_closed_exports
from report_combat_machinegun_hits import scan, vector
from report_combat_machinegun_aim import measure
from report_registered_machinegun_outcomes import outcome


def summarize(rows):
    outcomes = collections.Counter(r['outcome'] for r in rows)
    selected = [r for r in rows if r['world']['geometry']['known']]
    intersect = [r for r in selected if r['world']['geometry']['intersects']]
    blocked = [r for r in intersect if r['world']['static_before_bbox']]
    solid = [r for r in selected if r['outcome'] == 'solid_contact']
    return dict(shots=len(rows), outcomes=dict(outcomes), selected_known=len(selected),
        bbox_intersections=len(intersect), static_before_bbox=len(blocked),
        blocked_outcomes=dict(collections.Counter(r['outcome'] for r in blocked)),
        blocked_selected_target_damage=sum(r.get('selected_target_damage', False) for r in blocked),
        blocked_native_muzzle_error_mean=(sum(r['native_muzzle_error_world_units'] for r in blocked)/len(blocked) if blocked else None),
        intersect_selected_target_damage=sum(r.get('selected_target_damage', False) for r in intersect),
        intersect_outcomes=dict(collections.Counter(r['outcome'] for r in intersect)),
        solid_selected=len(solid), solid_selected_bbox_intersect=sum(r['world']['geometry']['intersects'] for r in solid),
        solid_selected_static_before_bbox=sum(r['world']['static_before_bbox'] for r in solid),
        observed_muzzle_startsolid=sum(r['world']['muzzle_trace']['StartSolid'] for r in selected),
        maps=dict(collections.Counter(r['map'] for r in rows)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--captures', type=Path, nargs='+', required=True)
    ap.add_argument('--exporter', type=Path, required=True)
    ap.add_argument('--assets', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--hand', type=int, choices=(0,1,2), default=2)
    args = ap.parse_args()
    assert not args.out.exists()
    args.out.mkdir(parents=True)
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    save(args.out/'protocol.json', dict(version='combat_static_observed_shot_occlusion_v3', hand=args.hand,
        purpose='train_diagnostic_only', sources=[str(p.resolve()) for p in args.captures],
        formula='Current client pose + sampled delta + observed recoil + native handedness MG muzzle; exact forward bbox slabs, static MASK_SHOT BSP trace to bbox entry.',
        tolerance_world_units=.125, training_performed=False, neural_execution=False,
        limitations=['No dynamic movers/entities', 'No next-pose/recoil prediction',
                     'No integer-angle ballistic quantization or spread', 'No native hidden-state inputs'],
        decision='Describe potential missing geometry; no controller, reward, model or default modification.'))
    requests, members, seen = [], [], set()
    for capture in args.captures:
        capture = capture.resolve()
        native = read(pin(capture/'native-complete.json'))
        assert native['state'] == 'complete' and not native['ppo_ready']
        recipe = read(pin(capture/'recipe.json', native['recipe_sha256']))
        assert recipe['split'] == 'train' and recipe['purpose'] == 'shot_predictive_train_only'
        pool = read(pin(capture/'pool/report.json', native['pool_sha256']))
        assert pool['state'] == 'complete' and pool['source_unchanged']
        seal = read(pin(capture/'dataset/complete.json'))
        report = read(pin(capture/'dataset/report.json', seal['report_sha256']))
        assert report['state'] == 'complete' and report['device'] == 'cuda'
        assert not report['ppo_ready'] and not report['policy_training']
        member_file = pin(capture/'dataset/members.jsonl.gz', report['members_sha256'])
        with gzip.open(member_file, 'rt', encoding='utf-8') as f:
            ms = [json.loads(line) for line in f]
        assert len(ms) == report['shots']
        by_job = collections.defaultdict(list)
        for m in ms:
            key = (m['episode'], m['seed'], m['shot'])
            assert key not in seen
            seen.add(key)
            by_job[m['job']].append(m)
        jobs = {j['job']: j for j in pool['jobs']}
        for job, members_for_job in sorted(by_job.items()):
            j = jobs[job]
            cp = pin(Path(j['root'])/'report.json')
            cr = read(cp)
            assert cr['capture_complete'] and cr['provenance_valid']
            path = Path(cr['results'][0]['root'])/'dataset/steps.jsonl'
            path = pin(path, report['inputs'][str(path.resolve())])
            wanted = {m['index'] for m in members_for_job}
            with path.open(encoding='utf-8-sig') as f:
                lookup = {s['index']: s for s in map(json.loads, f) if s['index'] in wanted}
            assert len(lookup) == len(wanted)
            windows = {}
            for s in lookup.values():
                n = s['native_step']
                key = (n['spawncount'],n['begin_frame'],n['actor'],n['sequence'])
                assert key not in windows
                windows[key] = s
            log = Path(cr['results'][0]['root'])/'server.log'
            log = pin(log, report['inputs'][str(log.resolve())])
            with log.open(encoding='utf-8-sig') as f:
                shots = {int(s['fields']['shot']): s for s in scan(f,set(windows)) if s['eligible']}
            for m in members_for_job:
                s = lookup[m['index']]
                assert s['observation']['identity']['frame'] == m['frame']
                assert s['owner'] == 'provider'
                # Explicitly omit native_step, native events and next_observation.
                sample = {k: s['sample'][k] for k in ('latent','vertical','weapon','target','aim_mode') if k in s['sample']}
                requests.append(dict(observation=s['observation'], sample=sample))
                shot = shots[m['shot']]
                n = s['native_step']
                assert shot['window'] == (n['spawncount'],n['begin_frame'],n['actor'],n['sequence'])
                assert outcome(shot) == m['outcome']
                measured = measure(shot,s)
                members.append(dict(m, capture=str(capture), map=s['observation']['identity']['map'],
                    target_status=measured['status'], selected_target_damage=measured.get('selected_target_damage',False),
                    native_muzzle_label=vector(shot['fields']['start']),
                    native_view_label=vector(shot['fields']['view']),
                    native_recoil_label=vector(shot['fields']['recoil']),
                    native_aim_label=vector(shot['fields']['aim']),
                    native_velocity_label=vector(shot['fields']['velocity']),
                    native_game_frame_label=int(shot['fields']['frame']),
                    native_server_frame_label=shot['server_frame']))
    assert requests
    ip, op = args.out/'requests.jsonl', args.out/'observed-world.jsonl'
    with ip.open('w', encoding='utf-8') as f:
        for row in requests:
            f.write(json.dumps(row, separators=(',', ':'))+'\n')
    pin(args.exporter)
    subprocess.run([str(args.exporter.resolve()), '--input', str(ip.resolve()), '--out', str(op.resolve()),
                    '--assets', str(args.assets.resolve()), '--hand',str(args.hand)], check=True,
                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    with op.open(encoding='utf-8') as f:
        world = [json.loads(line) for line in f]
    assert len(world) == len(members)
    rows = [dict(m, world=w) for m,w in zip(members,world)]
    for row in rows:
        w = row['world']
        assert w['hand']==args.hand
        row['native_muzzle_error_world_units'] = math.dist(w['geometry']['muzzle'],row['native_muzzle_label']) if w['geometry']['known'] else None
        if w['geometry']['known']:
            assert w['muzzle_trace']['Valid']
            if w['geometry']['intersects']:
                assert w['static_trace']['Valid']
    maps = {r['map']: r['world']['bsp_sha256'] for r in rows}
    assert all(r['world']['bsp_sha256'] == maps[r['map']] for r in rows)
    with gzip.open(args.out/'members.jsonl.gz', 'wt', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, separators=(',', ':'))+'\n')
    groups = {'all': summarize(rows)}
    for name in ['base3','ware2']:
        groups[name] = summarize([r for r in rows if r['map'] == name])
    for name in ['intervened','baseline']:
        groups[name] = summarize([r for r in rows if r['exploration_eligible'] and r['intervened'] == (name == 'intervened')])
    compact_closed_exports(args.out)
    for path, digest in pins.items():
        assert sha(Path(path)) == digest, path
    save(args.out/'report.json', dict(state='complete',version='combat_static_observed_shot_occlusion_v3',hand=args.hand,
        groups=groups, maps=maps, cases=len({(r['episode'],r['seed']) for r in rows}),
        inputs=pins, exporter_sha256=sha(args.exporter), script_sha256=sha(__file__),
        request_sha256=sha(ip), world_sha256=sha(op), members_sha256=sha(args.out/'members.jsonl.gz'),
        training_performed=False, neural_execution=False, promotion=False,
        scope='Observed static approximation only. Intersection/occlusion does not establish actual ballistic cause or action counterfactual value. Native outcomes are labels only; no fit or replay.'))
    save(args.out/'complete.json', dict(state='complete', report_sha256=sha(args.out/'report.json')))
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':
    main()

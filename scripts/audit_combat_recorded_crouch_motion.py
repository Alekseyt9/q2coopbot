"""Reconstruct existing physics diagnostics before assessing a stopped-crouch proposal."""
import argparse
import collections
import json
import math
import os
from pathlib import Path
import subprocess

from evaluate_registered_combat_models import read, save, sha


SNAPSHOT_FIELDS = ('ducked', 'inventory', 'inventory_known', 'inventory_age_frames', 'inventory_open',
    'map', 'frame', 'delta_frame', 'self', 'self_velocity', 'gravity', 'on_ground', 'pm_type', 'pm_time',
    'pm_flags', 'pm_state_known', 'collision_bodies', 'teammate', 'teammate_entity', 'last_teammate',
    'last_teammate_entity', 'teammate_age_frames', 'health', 'armor', 'ammo', 'weapon', 'gun_frame',
    'view_angles', 'kick_angles_degrees', 'kick_angles_known', 'delta_angles', 'enemies', 'projectiles',
    'barrels', 'obstacles', 'defeated', 'pickups', 'movers', 'beams', 'removed_entities', 'suppressed',
    'sounds', 'explosions')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    repo = Path(__file__).resolve().parents[1]
    pins, entries, labels = {}, [], {}

    def pin(path, expected=None):
        path = Path(path).resolve(); digest = pins.get(str(path)) or sha(path)
        assert expected is None or digest == expected, str(path)
        assert str(path) not in pins or pins[str(path)] == digest
        pins[str(path)] = digest
        return path

    for path in (Path(__file__), repo / 'internal/bot/recorded_motion_audit.go', repo / 'cmd/q2motion-audit/main.go',
                 repo / 'internal/bot/pmove_ground_prediction.go', repo / 'internal/bot/combat_motion_path.go'):
        pin(path)
    for corpus in args.corpus:
        corpus = corpus.resolve()
        closed = read(pin(corpus / 'complete.json'))
        protocol = read(pin(corpus / 'protocol.json', closed['protocol_sha256']))
        assert protocol['train_only'] and len(protocol['assignments']) == 1
        assignment = protocol['assignments'][0]
        assert assignment['split'] == 'train'
        plan = read(pin(assignment['plan'], assignment['plan_sha256']))
        recipes = {seed: task['episode']['id'] for task in plan['tasks'] for seed in task['seeds']}
        pool = read(pin(corpus / 'pool/report.json', closed['pool_sha256']))
        jobs = {j['job']: j for j in pool['jobs']}
        meta = read(pin(corpus / 'processing-train/report.json', closed['processed'][0]['sha256']))
        assert meta['device'] == 'cuda' and len(meta['cases']) == 24
        for case in meta['cases']:
            job = jobs[case['job']]
            folder = corpus / 'processing-train' / f"case-{job['job']:04d}"
            cm = read(pin(folder / 'cuda/report.json'))
            with pin(folder / 'cuda/rollout.jsonl', cm['rollout_sha256']).open(encoding='utf-8-sig') as stream:
                ids = {r['index'] for r in map(json.loads, stream)}
            steps = next((folder / 'native').glob('replay-*/steps.jsonl')).resolve()
            selected = {}
            for step in map(json.loads, pin(steps, cm['source_sha256'][str(steps)]).open(encoding='utf-8-sig')):
                if step['index'] in ids and step['action']['vertical'] == 'crouch' and step['applied_action']['vertical'] == 'release':
                    frame = step['observation']['identity']['frame']
                    assert frame not in selected
                    selected[frame] = step
            capture = read(pin(Path(job['root']) / 'report.json'))
            assert capture['capture_complete'] and capture['provenance_valid']
            result = capture['results'][0]
            worker = Path(result['root'])
            manifest = read(pin(Path(job['root']) / 'manifest.json'))
            # Completed runtimes deliberately shed duplicate PAK links. Read
            # the retained shared source, checking its original content hashes.
            cleanup = read(pin(worker / 'runtime/pak-cleanup.json'))
            asset_roots = set()
            assets = {}
            for asset in cleanup['removed']:
                path = pin(asset['source'], asset['sha256'].lower())
                asset_roots.add(path.parent)
                assets[path.name] = asset['sha256'].lower()
            assert len(asset_roots) == 1 and set(assets) == {'pak0.pak', 'pak1.pak', 'pak2.pak'}
            assert all(assets[a['path']] == a['sha256'].lower() for a in manifest['bsp_assets'])
            asset_root = asset_roots.pop()
            matched = set()
            trace = pin(worker / 'bot.jsonl')
            with trace.open(encoding='utf-8-sig') as stream:
                for raw in map(json.loads, stream):
                    frame = raw['observation_frame']
                    if frame not in selected:
                        continue
                    step = selected[frame]
                    assert frame not in matched
                    matched.add(frame)
                    capture_policy = raw['combat_policy']
                    assert capture_policy['observation'] == step['observation']
                    assert raw['sent_command'] == step['sent_command'] == capture_policy['applied_command']
                    assert capture_policy['proposed'] == step['action']
                    key = f"{corpus.parent.parent.name}/{corpus.parent.name}/{job['seed']}/{frame}"
                    assert key not in labels
                    entry = dict(id=key, runtime_root=str(asset_root),
                        snapshot={k: raw[k] for k in SNAPSHOT_FIELDS if k in raw},
                        proposed=capture_policy['proposed_command'], applied=raw['sent_command'],
                        logged_applied_prediction=raw.get('pmove_ground_prediction'), logged_ground_surface=raw['ground_surface'])
                    entries.append(entry)
                    obs, nxt = step['observation'], step.get('next_observation')
                    adjacent = bool(nxt and nxt['identity']['frame'] == frame + 1 and all(
                        obs['identity'][k] == nxt['identity'][k] for k in ('map', 'connection', 'spawncount', 'actor', 'life')))
                    labels[key] = dict(corpus=str(corpus), seed=job['seed'], episode=recipes[job['seed']], frame=frame,
                        on_ground=obs['on_ground'], vertical_velocity=obs['velocity'][2],
                        reasons=[v['reason'] for v in step.get('interventions') or [] if v['component'] == 'movement'],
                        next_position=nxt['position'] if adjacent else None,
                        native_comparable=bool(adjacent and nxt['on_ground'] and nxt['health'] == obs['health'] and nxt['armor'] == obs['armor']))
            assert matched == set(selected)
    assert len(entries) == 638
    args.out.mkdir(parents=True)
    save(args.out / 'request.json', dict(version='combat_recorded_motion_request_v1', entries=entries))
    save(args.out / 'labels.json', labels)
    env = dict(os.environ, GOCACHE=str(repo / 'workspace/build/gocache'), GOTOOLCHAIN='auto')
    with (args.out / 'build.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(['go', 'build', '-o', str(args.out.resolve() / 'q2motion-audit.exe'), './cmd/q2motion-audit'],
            cwd=repo, env=env, stdout=log, stderr=log)
    assert result.returncode == 0, str(args.out / 'build.log')
    with (args.out / 'geometry.log').open('w', encoding='utf-8') as log:
        result = subprocess.run([str(args.out.resolve() / 'q2motion-audit.exe'), '--request', str((args.out / 'request.json').resolve()),
            '--out', str((args.out / 'geometry.json').resolve())], cwd=repo, stdout=log, stderr=log)
    assert result.returncode == 0, str(args.out / 'geometry.log')
    geometry = read(args.out / 'geometry.json')
    assert not geometry['neural_execution'] and not geometry['guard_changed']
    totals, reasons, rows, errors = collections.Counter(), collections.defaultdict(collections.Counter), [], []
    for item in geometry['results']:
        label = labels[item['id']]
        audit = item['audit']
        counts = dict(cases=1, applied_prediction_mismatch=int(not item['applied_prediction_exact']),
            surface_mismatch=int(not item['ground_surface_exact']), applied_prediction_present=int(item['applied_prediction_present']),
            stopped_crouch_eligible=int(audit['stopped_crouch_eligible']),
            stopped_crouch_supported=int(audit['stopped_crouch_prediction'] is not None),
            stopped_crouch_path_checked=int(audit['stopped_crouch_path_checked']),
            proposed_supported=int(audit['proposed_prediction'] is not None), proposed_path_checked=int(audit['proposed_path_checked']))
        totals.update(counts)
        for reason in label['reasons']:
            reasons[reason].update(counts)
        position_error = None
        if label['native_comparable'] and audit['applied_prediction']:
            position_error = math.dist(audit['applied_prediction']['position'], label['next_position'])
            errors.append(position_error)
        rows.append(dict(**label, id=item['id'], **counts, applied_native_position_error=position_error))
    assert totals['cases'] == len(entries)
    assert all(sha(p) == h for p, h in pins.items())
    mechanical = totals['applied_prediction_mismatch'] == totals['surface_mismatch'] == 0 and totals['applied_prediction_present'] > 0
    errors.sort()
    save(args.out / 'report.json', dict(state='complete', version='combat_recorded_crouch_motion_audit_v1',
        mechanical_reconstruction_passed=mechanical, totals=dict(totals), reasons={k: dict(v) for k, v in reasons.items()},
        rows=rows, inputs=pins, request_sha256=sha(args.out / 'request.json'), geometry_sha256=sha(args.out / 'geometry.json'),
        native_position_error=dict(count=len(errors), maximum=max(errors, default=None),
            mean=sum(errors) / len(errors) if errors else None),
        neural_execution=False, training_performed=False, new_games=False, guard_changed=False, promotion=False,
        scope='Replay of observed-state geometric diagnostics, not neural inference. Null prediction matches do not prove accuracy. '
              'Counterfactual stopped-crouch path checks are not native outcomes or live permissions. '
              'Native error uses actual applied commands with unchanged health/armor and adjacent grounded next observation.'))
    save(args.out / 'complete.json', dict(state='complete', report_sha256=sha(args.out / 'report.json')))
    print(dict(mechanical_reconstruction_passed=mechanical, totals=dict(totals)))


if __name__ == '__main__':
    main()

"""Inspect first per-map blocked TRAIN grenade contexts using existing scalar physics."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from evaluate_registered_combat_models import read, save, sha
from audit_combat_recorded_crouch_motion import SNAPSHOT_FIELDS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workflow', type=Path, required=True)
    ap.add_argument('--preimpact', type=Path, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    assert not a.out.exists()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = pins.get(str(path)) or sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    def lines(path):
        with path.open(encoding='utf-8-sig') as stream:
            return [json.loads(line) for line in stream]

    pin(__file__)
    pin(repo/'scripts/audit_combat_recorded_crouch_motion.py')
    for path in sorted((repo/'internal/bot').glob('*.go')):
        pin(path)
    for path in sorted((repo/'internal/quake').glob('*.go')):
        pin(path)
    for name in ('q2motion-audit', 'q2support-audit'):
        pin(repo/'cmd'/name/'main.go')
    candidates, selected = [], {}
    for source in a.preimpact:
        seal = read(pin(source/'complete.json'))
        report = read(pin(source/'report.json', seal['report_sha256']))
        assert seal['state'] == report['state'] == 'complete'
        assert report['version'] == 'combat_grenade_preimpact_v1'
        assert not report['new_games'] and not report['neural_execution'] and not report['training_performed']
        assert sha(a.workflow/'complete.json') == report['inputs'][str((a.workflow/'complete.json').resolve())]
        for path, digest in report['inputs'].items():
            pin(path, digest)
        for event in sorted(report['events'], key=lambda e: (e['round'], e['seed'], e['index'], e['event_ordinal'])):
            context = event['contexts']['3']['context']
            if not (event['lethal'] and context and context['same_id_candidate_in_slots']
                    and context['same_id_candidate_velocity_known'] and context['request_reduced_near_zero']
                    and context['on_ground']):
                continue
            item = dict(arm=report['arm'], source=str(source.resolve()), event=event, context=context)
            candidates.append(item)
            selected.setdefault(event['event']['map'], item)
    assert set(selected) == {'base3', 'city1', 'jail1', 'ware2'}
    entries, labels = [], {}
    for map_name, item in selected.items():
        event, context, arm = item['event'], item['context'], item['arm']
        corpus = a.workflow/f"round-{event['round']:02d}"/arm/'corpus'
        receipt = read(pin(corpus/'complete.json'))
        pool = read(pin(receipt['source_pool_report'], receipt['pool_sha256']))
        job = next(j for j in pool['jobs'] if j['job'] == event['job'])
        assert job['seed'] == event['seed'] and not job['error']
        capture = read(pin(Path(job['root'])/'report.json'))
        manifest = read(pin(Path(job['root'])/'manifest.json'))
        assert capture['capture_complete'] and capture['provenance_valid']
        worker = Path(capture['results'][0]['root'])
        meta = read(pin(corpus/'processing-train'/f"case-{event['job']:04d}"/'cuda/report.json'))
        step_path = next((corpus/'processing-train'/f"case-{event['job']:04d}"/'native').glob('replay-*/steps.jsonl')).resolve()
        step = next(s for s in lines(pin(step_path, meta['source_sha256'][str(step_path)])) if s['index'] == context['index'])
        trace = pin(worker/'bot.jsonl', meta['source_sha256'].get(str((worker/'bot.jsonl').resolve())))
        matches = [raw for raw in lines(trace) if raw.get('observation_frame') == context['frame']
            and (raw.get('combat_policy') or {}).get('observation') == step['observation']]
        assert len(matches) == 1
        raw = matches[0]
        policy = raw['combat_policy']
        assert raw['sent_command'] == step['sent_command'] == policy['applied_command']
        assert policy['proposed'] == step['action'] and raw['self'] == step['observation']['position']
        assert raw['self_velocity'] == step['observation']['velocity']
        cleanup = read(pin(worker/'runtime/pak-cleanup.json'))
        roots, assets = set(), {}
        for asset in cleanup['removed']:
            path = pin(asset['source'], asset['sha256'].lower())
            roots.add(path.parent)
            assets[path.name] = asset['sha256'].lower()
        assert len(roots) == 1 and set(assets) == {'pak0.pak', 'pak1.pak', 'pak2.pak'}
        assert all(assets[asset['path']] == asset['sha256'].lower() for asset in manifest['bsp_assets'])
        key = f"{arm}/{map_name}/{event['seed']}/{context['index']}"
        entries.append(dict(id=key, runtime_root=str(roots.pop()),
            snapshot={k: raw[k] for k in SNAPSHOT_FIELDS if k in raw},
            proposed=policy['proposed_command'], applied=raw['sent_command'],
            logged_applied_prediction=raw.get('pmove_ground_prediction'), logged_ground_surface=raw['ground_surface']))
        labels[key] = dict(arm=arm, map=map_name, round=event['round'], seed=event['seed'], job=event['job'],
            index=context['index'], frame=context['frame'], lethal_network_frame=event['event']['frame'],
            loadout=event['loadout'], reasons=context['movement_interventions'], context=context)
    a.out.mkdir(parents=True)
    save(a.out/'request.json', dict(version='combat_recorded_motion_request_v1', entries=entries))
    save(a.out/'labels.json', labels)
    env = dict(os.environ, GOCACHE=str(repo/'workspace/build/gocache'), GOTOOLCHAIN='auto')
    for command in ('q2motion-audit', 'q2support-audit'):
        executable = a.out.resolve()/(command+'.exe')
        for label, args in (
                ('build-'+command, ['go', 'build', '-o', str(executable), './cmd/'+command]),
                (command, [str(executable), '--request', str((a.out/'request.json').resolve()),
                    '--out', str((a.out/(command+'.json')).resolve())])):
            with (a.out/(label+'.log')).open('w', encoding='utf-8') as log:
                result = subprocess.run(args, cwd=repo, env=env, stdout=log, stderr=log)
            assert result.returncode == 0, str(a.out/(label+'.log'))
    motion = read(pin(a.out/'q2motion-audit.json'))
    support = read(pin(a.out/'q2support-audit.json'))
    assert not motion['neural_execution'] and not motion['guard_changed']
    assert len(motion['results']) == len(support['results']) == len(entries) == 4
    assert all(r['applied_prediction_exact'] and r['ground_surface_exact'] for r in motion['results'])
    assert all(sha(path) == digest for path, digest in pins.items())
    save(a.out/'report.json', dict(state='complete', version='combat_grenade_motion_geometry_v1',
        candidates=len(candidates), labels=labels, motion=motion['results'], support=support['results'], inputs=pins,
        neural_execution=False, new_games=False, training_performed=False, guard_changed=False,
        scope='First qualifying state per map in declared input-arm order, then round/seed/step/event order. '
              'Only own TRAIN lethal grenade contexts, exact frame minus three, known same-ID candidate velocity, '
              'grounded and requested movement reduced near zero. Same-ID does not prove grenade lifetime. '
              'Original trace observation/proposal/integer-command and asset SHA matched; existing scalar physics only. '
              'Unknown prediction is not permission to move. No counterfactual survivability or general-quality claim.'))
    save(a.out/'complete.json', dict(state='complete', report_sha256=sha(a.out/'report.json')))
    print(dict(candidates=len(candidates), states=4, motion=motion['results']), flush=True)


if __name__ == '__main__':
    main()

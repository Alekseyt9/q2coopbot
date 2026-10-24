"""Join TRAIN corpse contacts to the exact pre-command client trace; no inference."""
import argparse
import collections
import json
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha
from report_combat_machinegun_hits import scan
from report_registered_machinegun_outcomes import outcome


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
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    for name in ('audit_combat_execution_contract_body_visibility.py', 'report_combat_machinegun_hits.py',
                 'report_registered_machinegun_outcomes.py', 'native_projectile_window.py'):
        pin(Path(__file__).with_name(name))
    seal = read(pin(args.shots/'complete.json'))
    previous = read(pin(args.shots/'report.json', seal['report_sha256']))
    assert seal['state'] == previous['state'] == 'complete' and previous['train_cases'] == 192
    for path, digest in previous['inputs'].items():
        pin(path, digest)
    root = args.workflow.resolve()
    assert str(root/'complete.json') in previous['inputs']
    counts, items, reports = collections.Counter(), [], {}
    for case in previous['cases']:
        expected = case['groups'].get('all', {}).get('outcomes', {}).get('corpse_contact', 0)
        if not expected:
            continue
        key = (case['arm'], case['round'])
        processing = root/f"round-{case['round']:02d}"/case['arm']/'corpus/processing-train'
        if key not in reports:
            reports[key] = read(pin(processing/'report.json', previous['inputs'][str(processing/'report.json')]))
        member = next(member for member in reports[key]['cases'] if member['seed'] == case['seed'])
        directory = processing/f"case-{member['job']:04d}"
        meta = read(pin(directory/'cuda/report.json', previous['inputs'][str(directory/'cuda/report.json')]))
        replay = pin(directory/'cuda/rollout.jsonl', meta['rollout_sha256'])
        with replay.open(encoding='utf-8-sig') as stream:
            accepted = {(row['index'], row['frame']) for row in map(json.loads, stream)}
        steps = pin(directory/'native/replay-0/steps.jsonl', meta['source_sha256'][str(directory/'native/replay-0/steps.jsonl')])
        windows = {}
        with steps.open(encoding='utf-8-sig') as stream:
            for step in map(json.loads, stream):
                if (step['index'], step['observation']['identity']['frame']) in accepted:
                    native = step['native_step']
                    window = tuple(native[name] for name in ('spawncount', 'begin_frame', 'actor', 'sequence'))
                    assert window not in windows
                    windows[window] = step
        assert len(windows) == len(accepted) == case['rows']
        log = next(Path(path) for path in meta['source_sha256'] if path.endswith('server.log'))
        pin(log, meta['source_sha256'][str(log)])
        with log.open(encoding='utf-8-sig') as stream:
            shots = [shot for shot in scan(stream, set(windows)) if shot['eligible'] and outcome(shot) == 'corpse_contact']
        assert len(shots) == expected
        trace = next(Path(path) for path in meta['source_sha256'] if path.endswith('bot.jsonl'))
        pin(trace, meta['source_sha256'][str(trace)])
        needed = {shot['window'] for shot in shots}
        trace_rows = {}
        with trace.open(encoding='utf-8-sig') as stream:
            for row in map(json.loads, stream):
                window = tuple(row.get(name) for name in ('spawncount', 'frame', 'self_entity', 'client_sequence'))
                if window in needed:
                    assert window not in trace_rows
                    trace_rows[window] = row
        assert set(trace_rows) == needed, 'Missing exact pre-command client trace'
        for shot in shots:
            row, step = trace_rows[shot['window']], windows[shot['window']]
            observation = step['observation']
            assert row['combat_policy']['observation']['identity'] == observation['identity']
            blocker = int(shot['fields']['target'])
            flags = {}
            for name in ('collision_bodies', 'obstacles', 'defeated', 'enemies', 'barrels'):
                flags['client_'+name] = any(item['id'] == blocker for item in row.get(name, []))
            for name in ('enemies', 'visible_props', 'remembered_threats'):
                flags['policy_'+name] = any(item['id'] == blocker for item in observation.get(name) or [])
            body = next((item for item in row.get('collision_bodies', []) if item['id'] == blocker), None)
            flags['client_nonzero_solid'] = body is not None and body.get('solid', 0) not in (0, 31)
            flags['blocker_is_selected_target'] = blocker == step['action'].get('target_entity', 0)
            counts['corpse_contacts'] += 1
            counts.update({name: int(value) for name, value in flags.items()})
            counts['any_explicit_client_body_record'] += any(flags['client_'+name] for name in ('collision_bodies', 'obstacles', 'defeated'))
            counts['any_policy_entity_or_memory_record'] += any(flags['policy_'+name] for name in ('enemies', 'visible_props', 'remembered_threats'))
            items.append(dict(arm=case['arm'], round=case['round'], seed=case['seed'], index=step['index'],
                frame=observation['identity']['frame'], shot=int(shot['fields']['shot']), blocker_id=blocker, flags=flags))
        print(dict(arm=case['arm'], round=case['round'], seed=case['seed'], corpse_contacts=len(shots)), flush=True)
    assert counts['corpse_contacts'] == 260
    assert all(sha(path) == digest for path, digest in pins.items())
    args.out.mkdir(parents=True)
    save(args.out/'report.json', dict(state='complete', version='combat_execution_contract_body_visibility_v1',
        counts=dict(counts), cases=items, inputs=pins, neural_execution=False, new_games=False,
        training_performed=False, promotion=False,
        scope='All 260 accepted TRAIN native corpse contacts; exact spawncount/frame/actor/client-sequence trace join. '
              'Client fields are recorded before the command. Presence in the network snapshot does not prove camera '
              'visibility or reliable shape; solid0/absent solid remains unknown. Current client body records and old '
              'remembered threats are counted separately. No native health/bbox enters policy or fitting.'))
    save(args.out/'complete.json', dict(state='complete', report_sha256=sha(args.out/'report.json')))


if __name__ == '__main__':
    main()

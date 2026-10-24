"""Check paired one-frame crouch actuation and replay every policy on CUDA."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from evaluate_registered_combat_models import read, save, sha


def normalized(value):
    if isinstance(value, dict):
        return {k: 0 if k == 'spawncount' else normalized(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalized(v) for v in value]
    return value


def same(a, b, tolerance):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], tolerance) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y, tolerance) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tolerance
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'root', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    repo = Path(__file__).resolve().parents[1]
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve(); digest = pins.get(str(path)) or sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    pin(__file__)
    spec = read(pin(args.protocol))
    native = read(pin(args.root / 'report.json'))
    assert native['state'] == 'complete' and native['source_unchanged'] and len(native['jobs']) == 8
    assert native['protocol_sha256'] == sha(args.protocol) and spec['new_native_games'] == spec['new_cuda_checks'] == 8
    model = pin(spec['model'], spec['model_sha256'])
    args.out.mkdir(parents=True)
    env = dict(os.environ, GOCACHE=str(repo / 'workspace/build/gocache'), GOTOOLCHAIN='auto')

    def command(label, values):
        with (args.out / (label + '.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(list(map(str, values)), cwd=repo, env=env, stdout=log, stderr=log)
        assert result.returncode == 0, str(args.out / (label + '.log'))

    exporter = args.out / 'q2ppo-data.exe'
    command('build', ['go', 'build', '-o', exporter.resolve(), './cmd/q2ppo-data'])
    traces, jobs, tasks = {}, {}, []

    def trace(path, last):
        values = {}
        for raw in map(json.loads, pin(path).open(encoding='utf-8-sig')):
            c = raw.get('combat_policy') or {}
            o = c.get('observation') or {}
            identity = o.get('identity') or {}
            frame = identity.get('frame', 0)
            if identity.get('life') == 1 and o.get('health', 0) > 0 and 98 <= frame <= last:
                assert frame not in values
                values[frame] = raw
        return values

    for job in sorted(native['jobs'], key=lambda j: j['job']):
        assert not job['error']
        case = next(c for c in spec['cases'] if c['name'] == job['case_name'])
        capture = read(pin(Path(job['root']) / 'report.json'))
        manifest = read(pin(Path(job['root']) / 'manifest.json'))
        assert capture['capture_complete'] and capture['provenance_valid']
        assert manifest['model_weights_sha256'].lower() == spec['model_sha256']
        expected_frame = case['frame'] if job['label'] == 'crouch' else 0
        assert manifest['diagnostic_stopped_crouch_frame'] == expected_frame
        result = capture['results'][0]
        assert result['dispatch_valid'] and result['capture_valid'] and result['seed_confirmed']
        key = job['case_name'], job['label'], job['repeat']
        assert key not in traces
        traces[key] = trace(Path(result['root']) / 'bot.jsonl', case['frame'] + 1)
        jobs[key] = job
        folder = args.out / f"case-{job['job']:04d}"
        folder.mkdir()
        command(f"native-{job['job']}", [exporter.resolve(), '--batch', job['root'], '--model', model,
            '--cuda-only', '--out', (folder / 'native').resolve()])
        metadata = read(pin(folder / 'native/report.json'))
        assert bool(metadata.get('diagnostic_only')) == bool(expected_frame)
        tasks.append(dict(model=str(model), model_sha256=sha(model), data=str((folder / 'native').resolve()),
            out=str((folder / 'cuda').resolve()), native_report_sha256=sha(folder / 'native/report.json')))
    gate = spec['mechanical_gate']
    tolerance = gate['prefix_float_tolerance']
    checks, details = {}, []

    def signature(row, commands=False):
        c = row['combat_policy']
        data = dict(observation=c['observation'], proposed=c['proposed'], proposed_command=c['proposed_command'],
            sample=(c.get('selection') or {}).get('sample'))
        if commands:
            data['sent_command'] = row['sent_command']
        return normalized(data)

    for case in spec['cases']:
        name, frame = case['name'], case['frame']
        reference = traces[name, 'control', 0]
        original = trace(pin(case['original_trace'], case['original_trace_sha256']), frame + 1)
        frames = list(range(98, frame + 1))
        checks[name + '/original_prefix'] = all(f in original and f in reference and
            same(signature(original[f], True), signature(reference[f], True), tolerance) for f in frames)
        for label in ('control', 'crouch'):
            for repeat in (0, 1):
                rows = traces[name, label, repeat]
                prefix = all(f in rows and f in reference and same(signature(rows[f]), signature(reference[f]), tolerance) for f in frames)
                prefix = prefix and all(f in rows and f in reference and rows[f]['sent_command'] == reference[f]['sent_command'] for f in frames if f < frame)
                checks[f'{name}/{label}/{repeat}/prefix'] = prefix
                root, following = rows.get(frame), rows.get(frame + 1)
                if not root or not following or frame not in reference:
                    checks[f'{name}/{label}/{repeat}/root_present'] = False
                    continue
                cmd = root['sent_command']; expected = copy.deepcopy(reference[frame]['sent_command'])
                if label == 'crouch':
                    expected['Up'] = case['expected_candidate_command']['Up']
                interventions = [i for i in root['combat_policy']['selection'].get('interventions') or []
                                 if i['reason'] == 'diagnostic_checked_stopped_crouch']
                o, nxt = root['combat_policy']['observation'], following['combat_policy']['observation']
                prediction = case['candidate_prediction']
                error = math.dist(nxt['position'], prediction['position'])
                velocity_error = math.dist(nxt['velocity'], prediction['velocity'])
                applied_ok = cmd == expected and cmd['Forward'] == cmd['Side'] == 0 and len(interventions) == (1 if label == 'crouch' else 0)
                checks[f'{name}/{label}/{repeat}/actuation'] = applied_ok
                checks[f'{name}/{label}/{repeat}/native_state'] = bool(nxt['on_ground'] and nxt['ducked'] == (label == 'crouch')
                    and nxt['health'] == o['health'] and nxt['armor'] == o['armor'] and error <= gate['position_tolerance'] and velocity_error <= gate['velocity_tolerance'])
                details.append(dict(case=name, label=label, repeat=repeat, prefix=prefix, applied_ok=applied_ok,
                    next_ducked=nxt['ducked'], position_error=error, velocity_error=velocity_error,
                    before_health=o['health'], next_health=nxt['health'], command=cmd))
            a, b = traces[name, label, 0], traces[name, label, 1]
            checks[f'{name}/{label}/cold_repeat'] = all(f in a and f in b and same(signature(a[f], True), signature(b[f], True), tolerance) for f in range(98, frame + 2))
    request = args.out / 'cuda-request.json'
    save(request, dict(version='combat_cuda_finalize_batch_v1', compact_finalized_streams=True, tasks=tasks))
    command('cuda', [sys.executable, 'scripts/finalize_combat_cuda_batch.py', '--request', request.resolve(),
        '--receipt-root', (args.out / 'cuda-batch').resolve()])
    for task in tasks:
        directory = Path(task['out'])
        metadata = read(pin(directory / 'report.json'))
        proof = read(pin(directory / 'cuda-verification.json', metadata['cuda_verification_sha256']))
        assert proof['state'] == 'passed' and proof['device'] == 'cuda'
        original_meta = read(Path(task['data']) / 'report.json')
        assert bool(metadata.get('diagnostic_only')) == bool(original_meta.get('diagnostic_only'))
    checks['cuda_replay_eight_cases'] = len(tasks) == 8
    assert all(sha(path) == digest for path, digest in pins.items())
    save(args.out / 'report.json', dict(state='complete', version='combat_stopped_crouch_native_probe_result_v1',
        mechanical_gate_passed=all(checks.values()), checks=checks, details=details, inputs=pins, cuda_cases=8,
        training_performed=False, guard_default_changed=False, promotion=False,
        scope=spec['scope'] + ' Float prefix tolerance1e-9; command integers and categorical choices exact. '
              'Only spawncount namespace normalized. Diagnostic trajectories excluded from ordinary PPO fitting/merge.'))
    save(args.out / 'complete.json', dict(state='complete', report_sha256=sha(args.out / 'report.json')))
    print(dict(mechanical_gate_passed=all(checks.values()), checks=checks, details=details))


if __name__ == '__main__':
    main()

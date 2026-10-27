"""Known-state native actuation proof, causal prefixes, and CUDA-only replay."""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from audit_combat_checked_ground_native_probe import normalized, same
from process_combat_architecture_pool import read, save, sha


def run(protocol, root, out):
    assert not out.exists()
    spec, native = read(protocol), read(root/'report.json')
    assert native['state'] == 'complete' and native['source_unchanged']
    assert spec['version'] == 'combat_forced_duck_native_probe_v1'
    assert len(native['jobs']) == 4 and spec['new_native_games'] == spec['new_cuda_checks'] == 4
    assert native['protocol_sha256'] == sha(protocol)
    case = spec['cases'][0]
    model = Path(spec['model'])
    assert sha(model) == spec['model_sha256']
    pins = {str(protocol.resolve()): sha(protocol), str(model): sha(model),
            str(Path(__file__).resolve()): sha(Path(__file__)), **spec['source_files']}
    assert all(sha(Path(p)) == digest for p, digest in pins.items())
    out.mkdir(parents=True)
    shutil.copyfile(__file__, out/Path(__file__).name)
    env = dict(os.environ, GOCACHE=str(Path.cwd()/'workspace/build/gocache'))

    def command(label, cmd):
        with (out/(label+'.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(list(map(str, cmd)), env=env, stdout=log, stderr=log)
        assert result.returncode == 0, label

    def traces(path):
        pins[str(path)] = sha(path)
        values = {}
        for line in path.open(encoding='utf-8-sig'):
            raw = json.loads(line)
            capture = raw.get('combat_policy') or {}
            observation = capture.get('observation') or {}
            identity = observation.get('identity') or {}
            if identity.get('life') == 1 and observation.get('health', 0) > 0:
                frame = identity['frame']
                assert frame not in values
                values[frame] = raw
        return values

    def signature(raw):
        c = raw['combat_policy']
        return normalized({k:c[k] for k in ('observation', 'proposed', 'proposed_command', 'applied_command')})

    source = traces(Path(case['original_trace']))
    assert sha(Path(case['original_trace'])) == case['original_trace_sha256']
    frame = case['frame']
    exporter = (out/'q2ppo-data.exe').resolve()
    command('build-exporter', ['go', 'build', '-o', exporter, './cmd/q2ppo-data'])
    all_traces, tasks, checks, details = {}, [], {}, []
    motion = read(Path('workspace/artifacts/combat-jail3-blocked-audit-v1-20261026/motion.json'))
    prediction = next(r['audit']['proposed_prediction'] for r in motion['results'] if r['id'] == f'{frame}-actual')
    for job in sorted(native['jobs'], key=lambda j:j['job']):
        assert not job['error']
        label, repeat = job['label'], job['repeat']
        capture = read(Path(job['root'])/'report.json')
        manifest = read(Path(job['root'])/'manifest.json')
        expected_frame = frame if label == 'request' else 0
        assert manifest.get('diagnostic_forced_duck_frame', 0) == expected_frame
        assert capture['capture_complete'] and capture['provenance_valid']
        result = capture['results'][0]
        assert result['dispatch_valid'] and result['capture_valid'] and result['seed_confirmed']
        trace = traces(Path(result['root'])/'bot.jsonl')
        all_traces[label, repeat] = trace
        prefix = sorted(f for f in source if 98 <= f < frame)
        checks[f'{label}/{repeat}/exact_original_prefix'] = bool(prefix) and all(
            f in trace and same(signature(source[f]), signature(trace[f]), 1e-9) for f in prefix)
        current = trace[frame]['combat_policy']
        expected = dict(source[frame]['combat_policy']['applied_command'])
        tags = [i for i in current['selection'].get('interventions', []) or []
                if i['reason'] == 'diagnostic_forced_duck_request']
        if label == 'request':
            requested = source[frame]['combat_policy']['proposed_command']
            expected.update(Forward=requested['Forward'], Side=requested['Side'])
        checks[f'{label}/{repeat}/command'] = current['applied_command'] == expected and len(tags) == (label == 'request')
        checks[f'{label}/{repeat}/release_preserved'] = current['applied_command']['Up'] == 0
        before = current['observation']
        after = trace[frame+1]['combat_policy']['observation']
        expected_position = prediction['position'] if label == 'request' else source[frame+1]['combat_policy']['observation']['position']
        expected_velocity = prediction['velocity'] if label == 'request' else source[frame+1]['combat_policy']['observation']['velocity']
        error = math.dist(after['position'], expected_position)
        velocity_error = math.dist(after['velocity'], expected_velocity)
        moved = math.dist(before['position'], after['position'])
        checks[f'{label}/{repeat}/native_pose_motion'] = after['ducked'] and after['on_ground'] and error <= .25 and velocity_error <= 1.
        if label == 'request':
            checks[f'{label}/{repeat}/actually_moved'] = moved > 1
        details.append(dict(label=label, repeat=repeat, position_error=error, velocity_error=velocity_error,
            actual_distance=moved, before_health=before['health'], after_health=after['health'],
            next_ducked=after['ducked'], command=current['applied_command']))
        directory = (out/f"case-{job['job']}").resolve()
        directory.mkdir()
        command(f"export-{job['job']}", [exporter, '--batch', job['root'], '--model', model, '--cuda-only', '--out', directory/'native'])
        meta = read(directory/'native/report.json')
        assert bool(meta.get('diagnostic_only')) == (label == 'request')
        tasks.append(dict(model=str(model), model_sha256=sha(model), data=str(directory/'native'),
            out=str(directory/'cuda'), native_report_sha256=sha(directory/'native/report.json')))
    for label in ('control', 'request'):
        a, b = all_traces[label, 0], all_traces[label, 1]
        checks[label+'/cold_repeat'] = all(f in b and same(signature(a[f]), signature(b[f]), 1e-9)
            for f in a if 98 <= f <= frame+1)
    save(out/'mechanical.json', dict(checks=checks, details=details))
    save(out/'cuda-request.json', dict(version='combat_cuda_finalize_batch_v1', compact_finalized_streams=True, tasks=tasks))
    command('cuda', [sys.executable, 'scripts/finalize_combat_cuda_batch.py', '--request', out/'cuda-request.json', '--receipt-root', out/'cuda-batch'])
    for task in tasks:
        meta, proof = read(Path(task['out'])/'report.json'), read(Path(task['out'])/'cuda-verification.json')
        assert proof['state'] == 'passed' and proof['device'] == 'cuda'
        if meta.get('diagnostic_forced_duck_frame'):
            assert meta['version'] == 'combat_diagnostic_forced_duck_rollout_v1'
            assert meta['diagnostic_only'] and not meta['ppo_training_ready'] and not proof['ordinary_training_eligible']
    assert all(sha(Path(p)) == digest for p, digest in pins.items())
    save(out/'report.json', dict(state='complete', mechanical_gate_passed=all(checks.values()), checks=checks,
        details=details, native_cases=4, cuda_cases=4, inputs=pins, training_performed=False, promotion=False,
        scope='Known validation one-step diagnostic, not whole-battle improvement or fresh quality. Default guard unchanged.'))
    save(out/'complete.json', dict(state='complete', report_sha256=sha(out/'report.json')))
    print(json.dumps(dict(mechanical_gate_passed=all(checks.values()), checks=checks, details=details), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('protocol', 'root', 'out'):
        p.add_argument('--'+k, required=True, type=Path)
    args = p.parse_args()
    run(args.protocol.resolve(), args.root.resolve(), args.out.resolve())

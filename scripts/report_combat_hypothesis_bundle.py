"""Close a sealed hypothesis evaluation with quality and native aim reports.

No policy execution or training. A failed child report never produces a bundle
completion seal; existing partial outputs are kept for diagnosis.
"""
import argparse
import json
import pathlib
import subprocess
import sys

from evaluate_registered_combat_models import read, save, sha


def validated_stage(spec, seal):
    stage = spec.get('hypothesis_stage')
    if stage is not None:
        assert stage['version'] == 'combat_hypothesis_stage_v1'
        assert len(spec['variants']) == 3
    else:
        stage = spec['component_ablation']
        assert stage['version'] == 'combat_component_ablation_stage_v1'
        assert stage['stage'] == 'component-ablation'
        assert len(spec['variants']) == 5
        assert sum(v['mode'] == 'rules' for v in spec['variants']) == 1
        assert set(stage['component_labels']) == {v['label'] for v in spec['variants'] if v['mode'] == 'learned'}
        assert all(v['cuda_replay'] and not v['deterministic'] for v in spec['variants'] if v['mode'] == 'learned')
    assert not stage['automatic_promotion'] and not stage['training_performed']
    assert stage['native_budget'] == seal['native_cases'] == stage['cases_per_variant'] * len(spec['variants'])
    assert stage['ordinary_cases'] + stage['recovery_cases'] == stage['cases_per_variant']
    assert seal['cuda_cases'] == sum(p['cases'] for p in seal['processed'])
    assert seal['cuda_cases'] == stage['cases_per_variant'] * sum(bool(v['cuda_replay']) for v in spec['variants'])
    return stage


def main():
    parser = argparse.ArgumentParser()
    for name in ('evaluation', 'spec', 'out'):
        parser.add_argument('--' + name, type=pathlib.Path, required=True)
    args = parser.parse_args()
    root, spec_path, out = [p.resolve() for p in
                            (args.evaluation, args.spec, args.out)]
    assert not out.exists(), 'Fresh output required'
    seal_path = root / 'complete.json'
    seal, spec = read(seal_path), read(spec_path)
    assert seal['state'] == 'complete' and not seal['training_performed']
    assert not seal['promotion'] and seal['spec_sha256'] == sha(spec_path)
    stage = validated_stage(spec, seal)
    pins = {seal_path: sha(seal_path), spec_path: sha(spec_path)}
    pool_path = root / 'pool/report.json'
    assert sha(pool_path) == seal['pool_sha256']
    jobs = {j['job']: j for j in read(pool_path)['jobs']}
    diagnostic_counts = {}
    for variant in spec['variants']:
        if not variant.get('diagnostic_nominal_aim'):
            continue
        path = root / ('execution-' + variant['label'] + '.json')
        execution = read(path)
        assert execution['state'] == 'complete' and execution['pool_sha256'] == seal['pool_sha256']
        active, changed = 0, 0
        for case in execution['cases']:
            capture_path = pathlib.Path(jobs[case['job']]['root']) / 'report.json'
            assert sha(capture_path) == case['capture_sha256']
            trace = pathlib.Path(read(capture_path)['results'][0]['root']) / 'bot.jsonl'
            assert sha(trace) == case['trace_sha256']
            pins[capture_path], pins[trace] = case['capture_sha256'], case['trace_sha256']
            with trace.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    capture = json.loads(line).get('combat_policy') or {}
                    selection = capture.get('selection') or {}
                    observation = capture.get('observation') or {}
                    identity = observation.get('identity') or {}
                    if (selection.get('owner') != 'provider' or identity.get('life') != 1
                            or observation.get('health', 0) <= 0
                            or identity.get('frame', -1) < case['network_release_frame']):
                        continue
                    marked = any(i['component'] in ('yaw', 'pitch') and
                        i['reason'] == 'diagnostic_nominal_observed_aim'
                        for i in selection.get('interventions') or [])
                    if not marked:
                        continue
                    active += 1
                    proposed, applied = capture['proposed'], capture['applied']
                    changed += any(abs((applied[key]-proposed[key]+180) % 360-180) > .01
                        for key in ('yaw_delta_degrees', 'pitch_delta_degrees'))
        assert changed > 0, 'Diagnostic never activated; no efficacy claim: ' + variant['label']
        diagnostic_counts[variant['label']] = dict(marked_frames=active, angle_changed_frames=changed)
        pins[path] = sha(path)
    out.mkdir(parents=True)
    reports = [
        ('quality', 'report_registered_combat_comparison.py', out / 'quality'),
        ('execution', 'report_registered_execution_limits.py', out / 'execution.json'),
        ('recovery', 'audit_registered_weapon_recovery.py', out / 'recovery.json'),
        ('weapon-damage', 'report_registered_weapon_damage.py', out / 'weapon-damage'),
        ('machinegun-aim', 'report_registered_machinegun_aim.py', out / 'machinegun-aim.json'),
    ]
    current = None
    try:
        outputs = []
        for current, script, target in reports:
            save(out / 'progress.json', dict(stage='reporting', report=current))
            command = [sys.executable, pathlib.Path(__file__).parent / script,
                       '--evaluation', root, '--spec', spec_path, '--out', target]
            if current == 'recovery':
                command += ['--labels', ','.join(v['label'] for v in spec['variants']
                                                 if v['mode'] == 'learned')]
            if current == 'weapon-damage':
                command += ['--recovery', out / 'recovery.json']
            with (out / (current + '.log')).open('w', encoding='utf-8') as log:
                result = subprocess.run(list(map(str, command)), stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            assert result.returncode == 0, f'{current}: inspect {out / (current + ".log")}'
            receipt = target / 'complete.json' if target.is_dir() else target
            assert read(receipt)['state'] == 'complete', current
            outputs.append(dict(kind=current, path=str(receipt), sha256=sha(receipt)))
        for path, digest in pins.items():
            assert sha(path) == digest, 'Evaluation inputs changed while reporting'
        assert sha(root / 'pool/report.json') == seal['pool_sha256']
        for item in outputs:
            assert sha(item['path']) == item['sha256']
        quality = read(out / 'quality/complete.json')
        assert quality['native_cases'] == seal['native_cases']
        assert quality['cuda_cases'] == seal['cuda_cases']
        save(out / 'complete.json', dict(state='complete',
            version='combat_hypothesis_report_bundle_v1',
            stage=stage['stage'], native_cases=seal['native_cases'],
            cuda_cases=seal['cuda_cases'], reports=outputs,
            diagnostic_variants=[v['label'] for v in spec['variants']
                                 if v.get('diagnostic_nominal_aim')],
            diagnostic_scope=spec.get('diagnostic_scope'),
            diagnostic_active_frames=diagnostic_counts,
            inputs=[dict(path=str(p), sha256=h) for p, h in pins.items()],
            script_sha256=sha(__file__), training_performed=False, promotion=False))
        save(out / 'progress.json', dict(stage='complete', reports=len(outputs)))
        print(out / 'quality/results.md')
    except BaseException as error:
        save(out / 'progress.json', dict(stage='failed', report=current, error=str(error)))
        raise


if __name__ == '__main__':
    main()

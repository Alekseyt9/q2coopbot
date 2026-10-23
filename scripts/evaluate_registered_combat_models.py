"""Sealed registry comparison of named model updates; CUDA replay, no training."""
import argparse
import hashlib
import math
import json
import pathlib
import shutil
import subprocess
import sys
import time


def read(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with pathlib.Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    parser.add_argument('--planner', type=pathlib.Path, help='Existing q2episode binary; otherwise build a private planner')
    parser.add_argument('--prepare-only', action='store_true', help='Validate and freeze plans without running battles or CUDA replay')
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    root = args.out.resolve(); assert not root.exists(), 'Fresh output required'
    spec_path = args.spec.resolve(); spec_sha = sha(spec_path); spec = read(spec_path)
    assert spec['version'] == 'registered_combat_model_comparison_v1'
    assert spec['split'] in ('validation', 'test', 'confirmation'), 'No train split in quality evaluation'
    variants = spec['variants']; labels = [v['label'] for v in variants]
    assert variants and len(set(labels)) == len(labels)
    assert all(label and all(c.isalnum() or c in '-_' for c in label) for label in labels)
    assert spec['max_instances'] == 16 and type(spec['count']) is int and 1 <= spec['count'] <= 80
    assert len(spec['comparisons']) == len({v['name'] for v in spec['comparisons']})
    assert all(v['name'] and all(c.isalnum() or c in '-_' for c in v['name']) for v in spec['comparisons'])
    assert all(v['before'] in labels and v['after'] in labels and v['before'] != v['after'] for v in spec['comparisons'])
    root.mkdir(parents=True)

    def run(label, command):
        with (root/(label+'.log')).open('w', encoding='utf-8') as output:
            result = subprocess.run(list(map(str, command)), cwd=repo, stdout=output, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert result.returncode == 0, str(root/(label+'.log'))

    def python(label, script, *arguments):
        run(label, [sys.executable, repo/'scripts'/script, *arguments])

    try:
        save(root/'progress.json', dict(stage='waiting-for-updates', spec_sha256=spec_sha, training_performed=False))
        deadline = time.monotonic() + 6*3600
        readiness = []
        for name in spec.get('ready', []):
            path = pathlib.Path(name)
            while not path.exists():
                progress = path.parent/'progress.json'
                if progress.exists():
                    assert read(progress)['stage'] != 'failed', 'Upstream failed; no restart'
                assert time.monotonic() < deadline, 'Readiness timeout; no restart'
                time.sleep(10)
            assert read(path)['state'] == 'complete'
            readiness.append(dict(path=str(path), sha256=sha(path)))
        assert sha(spec_path) == spec_sha
        minimum_free_gib=spec.get('minimum_free_gib',32)
        assert isinstance(minimum_free_gib,(int,float)) and math.isfinite(minimum_free_gib) and minimum_free_gib>=16,'Invalid storage reserve'
        assert shutil.disk_usage(root.anchor).free >= minimum_free_gib*1024**3, 'Storage guard'
        registry = pathlib.Path(spec['registry']); registry_sha = sha(registry)
        files = read(registry)['files']; assert files
        if spec.get('episodes'):
            selected = set(spec['episodes'])
            assert len(selected) == len(spec['episodes']) and selected <= {pathlib.Path(f).stem for f in files}
            files = [f for f in files if pathlib.Path(f).stem in selected]
        episodes = ','.join(pathlib.Path(f).stem for f in files)
        planner = args.planner.resolve() if args.planner else root/'q2episode.exe'
        if not args.planner:
            run('build-planner', ['go', 'build', '-o', planner, './cmd/q2episode'])
        assert planner.is_file()
        plans = []; bindings = []
        for variant in variants:
            label = variant['label']; model = None
            if variant['mode'] == 'learned':
                update = pathlib.Path(variant['update']); seal = read(update/'complete.json')
                for filename, key in [('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'), ('report.json', 'report_sha256')]:
                    assert sha(update/filename) == seal[key], filename
                report = read(update/'report.json'); assert report['device'] == 'cuda'
                model = update/'weights.json'; original = read(model)
                assert not original.get('diagnostic_joint_aim_probe'), 'Diagnostic one-frame acquisition models cannot enter quality evaluation'
                assert original['kind'] == 'combat_ppo_v1' and not original.get('deterministic', False)
                if variant['deterministic']:
                    original['deterministic'] = True; model = root/(label+'-weights.json'); save(model, original)
                    check = read(model); check.pop('deterministic', None)
                    expected = read(update/'weights.json'); expected.pop('deterministic', None)
                    assert check == expected, 'Deterministic copy changed parameters'
                if variant['cuda_replay']:
                    assert not variant['deterministic'], 'Deterministic actions are not stochastic PPO replay'
                    assert not original.get('execution_mode'), 'Inference-only execution cannot replay standard PPO'
                bindings.append(dict(label=label, update=str(update), update_seal_sha256=sha(update/'complete.json'),
                                     model=str(model), model_sha256=sha(model), cuda_replay=variant['cuda_replay']))
            else:
                assert variant['mode'] == 'rules' and not variant.get('cuda_replay')
            plan = root/(label+'-plan.json')
            command = [planner, '--registry', registry,
                       '--episodes', episodes, '--split', spec['split'], '--mode', variant['mode'],
                       '--count', spec['count'], '--seed-offset', spec['seed_offset'], '--out', plan,
                       '--root', repo, '--artifacts', root/('captures-'+label)]
            if model: command += ['--model', model]
            if variant.get('record_combat_advice', False):
                assert variant['mode'] == 'learned' and variant['record_combat_advice'] is True
                command += ['--record-combat-advice']
            if variant.get('checked_wall_slide',False):
                assert variant['mode']=='learned' and variant['checked_wall_slide'] is True
                command += ['--checked-wall-slide']
            if variant.get('diagnostic_nominal_aim', False):
                assert variant['mode'] == 'learned' and variant['diagnostic_nominal_aim'] is True
                command += ['--diagnostic-nominal-aim']
            run('plan-'+label, command); plans.append(plan)
        frozen = [read(p) for p in plans]; tasks = frozen[0]['tasks']
        assert len(tasks) == len(files)
        historical = set()
        old_plans = list((repo/'workspace/artifacts').glob('*plan*.json'))
        old_plans += [pathlib.Path(p) for p in spec.get('historical_plans', [])]
        for path in old_plans:
            for task in read(path).get('tasks', []): historical.update(task.get('seeds', []))
        for task in tasks:
            assert task['split'] == spec['split'] and len(task['seeds']) == spec['count']
            assert not historical.intersection(task['seeds']), 'Historical seed reused'
        for other in frozen[1:]:
            assert len(other['tasks']) == len(tasks)
            for left, right in zip(tasks, other['tasks']):
                for key in ('episode', 'instances', 'seeds', 'split'): assert left[key] == right[key], key
        count = len(tasks)*spec['count']; total = count*len(variants)
        stage = spec.get('hypothesis_stage')
        if stage:
            assert stage['version'] == 'combat_hypothesis_stage_v1'
            assert len(variants) == 3 and sum(v['mode'] == 'rules' for v in variants) == 1
            assert stage['cases_per_variant'] == count and stage['native_budget'] == total, 'Stage budget changed'
            recovery = sum(t['episode']['id'].endswith('-recovery') for t in tasks)*spec['count']
            assert stage['recovery_cases'] == recovery and stage['ordinary_cases'] == count-recovery
            assert stage['automatic_promotion'] is False and stage['training_performed'] is False
        save(root/'fixture-proof.json', dict(state='matched', families=len(tasks), cases_per_variant=count, native_cases=total,
             split=spec['split'], seed_offset=spec['seed_offset'], historical_seeds_checked=len(historical), models=bindings,
             spec_sha256=spec_sha, registry_sha256=registry_sha, readiness=readiness,
             plans=[dict(path=str(p), sha256=sha(p)) for p in plans]))
        if args.prepare_only:
            assert sha(spec_path) == spec_sha and sha(registry) == registry_sha
            save(root/'prepared.json', dict(state='prepared', native_cases_planned=total, native_cases_run=0,
                 spec_sha256=spec_sha, fixture_proof_sha256=sha(root/'fixture-proof.json'),
                 script_sha256=sha(pathlib.Path(__file__)), training_performed=False, promotion=False))
            save(root/'progress.json', dict(stage='prepared', native_cases_run=0, training_performed=False))
            return
        save(root/'progress.json', dict(stage='native-evaluation', cases=total, training_performed=False))
        shell = shutil.which('pwsh'); assert shell
        literal = lambda p: "'" + str(p).replace("'", "''") + "'"
        pool = root/'pool'
        command = '& '+literal(repo/'scripts/run_registered_combat_pool.ps1')+' -Plans @('+','.join(map(literal, plans))+') -MaxInstances 16 -Port '+str(spec['port'])+' -OutputRoot '+literal(pool)
        run('native-pool', [shell, '-NoProfile', '-Command', command])
        report = read(pool/'report.json'); digest = sha(pool/'report.json')
        assert report['state'] == 'complete' and report['source_unchanged'] and report['usable_captures'] == len(report['jobs']) == total
        assert all(not j['error'] for j in report['jobs'])
        python('damage-timing', 'audit_registered_damage_timing.py', '--pool', pool, '--out', root/'damage-timing.json')
        comparisons = []
        for comparison in spec['comparisons']:
            label = comparison['name']; target = root/('quality-'+label+'.json')
            python('quality-'+label, 'compare_registered_combat_pools.py', '--before', pool, '--before-plan-index', labels.index(comparison['before']),
                   '--after', pool, '--after-plan-index', labels.index(comparison['after']), '--out', target)
            comparisons.append(dict(**comparison, sha256=sha(target)))
        save(root/'native-complete.json', dict(state='complete', native_cases=total, pool_sha256=digest, comparisons=comparisons,
                                               training_performed=False, promotion=False))
        processed = []; behavior = []
        for index, variant in enumerate(variants):
            if variant['mode'] != 'learned': continue
            label = variant['label']; save(root/'progress.json', dict(stage='auditing', label=label, processed=processed))
            for script, prefix in [('audit_registered_combat_ownership.py', 'ownership'), ('audit_registered_target_choices.py', 'behavior'),
                                   ('audit_registered_weapon_choices.py', 'weapon'), ('audit_registered_execution_actions.py', 'execution')]:
                python(prefix+'-'+label, script, '--pool', pool, '--plan-index', index, '--out', root/(prefix+'-'+label+'.json'))
            behavior += ['--audit', label+'='+str(root/('behavior-'+label+'.json'))]
            if variant['cuda_replay']:
                processing = root/('processing-'+label)
                python('processing-'+label, 'process_registered_combat_pool_cuda.py', '--pool', pool, '--plan-index', index, '--out', processing)
                meta = read(processing/'report.json')
                assert meta['state'] == 'complete' and meta['device'] == 'cuda' and not meta['training_performed'] and len(meta['cases']) == count
                processed.append(dict(label=label, cases=count, rows=meta['rows'], report_sha256=sha(processing/'report.json')))
        python('action-efficiency', 'summarize_registered_action_efficiency.py', *behavior, '--damage', root/'damage-timing.json', '--out', root/'action-efficiency.json')
        assert sha(pool/'report.json') == digest and sha(spec_path) == spec_sha and sha(registry) == registry_sha
        for item in readiness: assert sha(item['path']) == item['sha256']
        for item in bindings:
            assert sha(item['model']) == item['model_sha256'] and sha(pathlib.Path(item['update'])/'complete.json') == item['update_seal_sha256']
        save(root/'complete.json', dict(state='complete', native_cases=total, cuda_cases=sum(p['cases'] for p in processed),
             spec_sha256=spec_sha, pool_sha256=digest, comparisons=comparisons, processed=processed, readiness=readiness,
             script_sha256=sha(pathlib.Path(__file__)), training_performed=False, promotion=False,
             scope='Matched fresh held-out registry fixtures across sealed updates and execution modes. CUDA replay only for explicitly stochastic PPO variants; deterministic evidence native only. No training, promotion, campaign or coop acceptance.'))
        save(root/'progress.json', dict(stage='complete', native_cases=total, processed=processed))
    except BaseException as error:
        save(root/'progress.json', dict(stage='failed', error=str(error), training_performed=False)); raise


if __name__ == '__main__':
    main()

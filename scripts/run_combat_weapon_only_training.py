"""One matched weapon-only update per arm, then a fixed five-variant screen."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from evaluate_registered_combat_models import read, save, sha, pool_command
from combat_weapon_only_scope import SCOPE
from run_combat_diverse_registry_ppo import process_alive


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'spec', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args(); repo = Path(__file__).resolve().parents[1]; root = a.out.resolve()
    protocol = read(a.protocol)
    assert protocol['version'] == SCOPE and protocol['rounds'] == 1 and protocol['cases_per_update'] == 24
    assert protocol['max_instances'] == 8 and protocol['timescale'] == 2
    configs = {str(p.resolve()): sha(p) for p in (a.protocol, a.spec)}
    if a.resume:
        run = read(root/'run.json'); state = read(root/'progress.json')
        assert not process_alive(state.get('runner_pid')) and not process_alive(state.get('child_pid'))
        assert run['configs'] == configs
        if run.get('pre_fit_revision'):
            reference = run['pre_fit_revision']; revision = read(reference['path'])
            assert sha(reference['path']) == reference['sha256']
            assert revision['version'] == 'combat_weapon_only_keep_default_recovery_v1'
            assert sha(revision['original_run']) == revision['original_run_sha256']
            original = read(revision['original_run'])
            assert original['configs'] == configs
            expected = {str((repo/'scripts'/n).resolve()) for n in
                        ('audit_combat_weapon_only_exposure.py', 'run_combat_weapon_only_training.py')}
            assert set(revision['replacements']) == expected
            revised = dict(original['source_files'])
            for path, change in revision['replacements'].items():
                assert revised[path] == change['before_sha256'] == sha(change['before_snapshot'])
                assert sha(path) == change['after_sha256']
                revised[path] = change['after_sha256']
            assert run['source_files'] == revised
            assert sha(root/'pool/report.json') == revision['pool_report_sha256']
            marker = root/'pre-fit-revision-applied.json'
            if marker.exists():
                assert read(marker)['revision_sha256'] == reference['sha256']
            else:
                assert not any((root/arm/'update').exists() for arm in ('control', 'explore'))
                save(marker, dict(revision_sha256=reference['sha256'], before_any_fitting=True))
    else:
        assert not root.exists(), 'Fresh workflow required'
        root.mkdir(parents=True)
        paths = set()
        for folder, extensions in ((repo/'scripts', ('.py', '.ps1')), (repo/'internal', ('.go',)),
                                   (repo/'cmd', ('.go',)), (repo.parent/'yquake2/src', ('.c', '.h'))):
            paths.update(p.resolve() for p in folder.rglob('*') if p.is_file() and p.suffix in extensions)
        run = dict(configs=configs, source_files={str(p):sha(p) for p in sorted(paths)},
                   native_train_budget=48, cuda_train_budget=48, native_eval_budget=40, cuda_eval_budget=32,
                   max_instances=8, promotion=False)
        save(root/'run.json', run)
    env = dict(os.environ, GOCACHE=str(repo/'workspace/build/gocache'), GOTOOLCHAIN='auto')

    def unchanged():
        assert all(sha(p) == h for p,h in configs.items()), 'Protocol changed'
        assert all(sha(p) == h for p,h in run['source_files'].items()), 'Source changed'

    def closed(target, receipt='complete.json'):
        value = read(target/receipt)
        assert value['state'] == ('prepared' if receipt == 'prepared.json' else 'complete')
        if 'report_sha256' in value:
            assert sha(target/'report.json') == value['report_sha256']
        return value

    def invoke(label, command):
        unchanged()
        with (root/(label+'.log')).open('w', encoding='utf-8') as log:
            child = subprocess.Popen(list(map(str, command)), cwd=repo, env=env, stdout=log, stderr=log,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            save(root/'progress.json', dict(stage=label, runner_pid=os.getpid(), child_pid=child.pid))
            result = child.wait()
        assert result == 0, str(root/(label+'.log'))
        unchanged()

    def step(label, command, target, receipt='complete.json'):
        unchanged()
        if (target/receipt).exists():
            closed(target, receipt); return
        assert not target.exists(), 'Partial output preserved: '+str(target)
        invoke(label, command); closed(target, receipt)

    py = lambda script,*args: [sys.executable, repo/'scripts'/script, *args]
    try:
        assert sha(protocol['initialization_protocol']) == protocol['initialization_protocol_sha256']
        assert sha(protocol['initializer_complete']) == protocol['initializer_complete_sha256']
        assert read(protocol['initializer_complete'])['state'] == 'complete'
        plans = []; sources = {}; history = list(protocol['historical_plans'])
        for index, arm in enumerate(('control', 'explore')):
            source = Path(protocol['initial_sources'][arm]); sources[arm] = source
            report = read(source/'report.json')
            assert report['profile'] == arm and report['metadata']['updates_completed'] == 0
            assert report['actor_critic_resumed_steps_exact'] and report['source_actor_value_std_exact']
            branch = root/arm; branch.mkdir(exist_ok=True)
            history_path = branch/'history.json'; value = dict(historical_plans=history[:])
            if history_path.exists():
                assert read(history_path) == value
            else:
                save(history_path, value)
            corpus = branch/'corpus'
            step('prepare-'+arm, py('collect_combat_registry_curriculum.py', '--source', source,
                '--registry', protocol['registry'], '--episodes', ','.join(protocol['episodes']),
                '--execution-contract', 'legacy_v1', '--history-spec', history_path,
                '--train-count', '1', '--train-offset', str(protocol['train_offset']+index),
                '--allow-groups', '--ppo-ready', '--train-only', '--prepare-only', '--out', corpus), corpus, 'prepared.json')
            plan = corpus/'train-plan.json'; plans.append(plan); history.append(str(plan))
        shell = shutil.which('pwsh'); assert shell
        pool = root/'pool'
        step('native-pool', [shell, '-NoProfile', '-Command', pool_command(repo, plans, 8, protocol['port'], pool)], pool, 'report.json')
        pooled = read(pool/'report.json')
        assert pooled['source_unchanged'] and pooled['usable_captures'] == len(pooled['jobs']) == 48
        updates = []
        for index, arm in enumerate(('control', 'explore')):
            branch = root/arm; corpus = branch/'corpus'
            if not (corpus/'complete.json').exists():
                assert not (corpus/'processing-train').exists() and not (corpus/'pool').exists()
                invoke('finalize-'+arm, py('finalize_combat_execution_corpus.py', '--corpus', corpus,
                    '--pool', pool, '--plan-index', index))
            receipt = closed(corpus)
            assert receipt['native_cases'] == receipt['cuda_cases'] == 24 and receipt['selected_plan_index'] == index
            native, advantage, exposure, weapon, update = (branch/name for name in
                ('native-credit', 'advantage-credit', 'mechanism-exposure', 'weapon-exposure', 'update'))
            step('native-credit-'+arm, py('audit_combat_blocked_ground_native_credit.py', '--corpus', corpus, '--out', native), native)
            step('advantage-'+arm, py('audit_combat_advantage_credit.py', '--corpus', corpus, '--native-audit', native, '--out', advantage), advantage)
            step('mechanism-exposure-'+arm, py('audit_combat_projectile_training_exposure.py', '--corpus', corpus,
                '--protocol', a.protocol.resolve(), '--round', 1, '--out', exposure), exposure)
            step('weapon-exposure-'+arm, py('audit_combat_weapon_only_exposure.py', '--corpus', corpus,
                '--source', sources[arm], '--protocol', a.protocol.resolve(), '--out', weapon), weapon)
            assert read(exposure/'report.json')['gate_passed'] and read(weapon/'report.json')['gate_passed'], 'Pre-fit coverage failed'
            step('train-'+arm, py('train_combat_weapon_only_cuda.py', '--source', sources[arm], '--corpus', corpus,
                '--native-credit', native, '--advantage-credit', advantage, '--mechanism-exposure', exposure,
                '--weapon-exposure', weapon, '--protocol', a.protocol.resolve(), '--out', update), update)
            report = read(update/'report.json')
            assert report['scope_updates'] == 1 and report['nonweapon_outputs_frozen_exact']
            assert report['disposable_resume_adam_step_exact'] and report['disposable_critic_resume_adam_step_exact']
            updates.append(dict(arm=arm, rows=report['rows'], actor_steps=report['actor_steps'], critic_steps=report['critic_steps'],
                                weights_sha256=report['weights_sha256']))
        save(root/'updates.json', dict(updates=updates))
        evaluation = root/'evaluation'
        step('evaluation', py('evaluate_registered_combat_models.py', '--spec', a.spec.resolve(), '--out', evaluation), evaluation)
        quality = root/'native-quality'
        step('native-quality', py('audit_registered_native_action_quality.py', '--evaluation', evaluation, '--out', quality), quality)
        result = root/'result'
        step('result', py('report_combat_weapon_only_training.py', '--workflow', root, '--protocol', a.protocol.resolve(),
            '--spec', a.spec.resolve(), '--out', result), result)
        unchanged()
        save(root/'complete.json', dict(state='complete', native_train_cases=48, cuda_train_cases=48,
            native_eval_cases=40, cuda_eval_cases=32, result_seal_sha256=sha(result/'complete.json'),
            source_unchanged=True, original_source_unchanged=not bool(run.get('pre_fit_revision')),
            pre_fit_revision=run.get('pre_fit_revision'), promotion=False))
        save(root/'progress.json', dict(stage='complete', runner_pid=os.getpid(), child_pid=None))
    except BaseException as error:
        save(root/'progress.json', dict(stage='failed', runner_pid=os.getpid(), error=str(error)))
        raise


if __name__ == '__main__':
    main()

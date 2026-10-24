"""Revised weapon-only admission: retain all native data and original control update."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from evaluate_registered_combat_models import read, save, sha
from run_combat_diverse_registry_ppo import process_alive


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'spec', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args(); repo = Path(__file__).resolve().parents[1]; root = a.out.resolve()
    protocol = read(a.protocol)
    assert protocol['experiment_revision'] == 'frozen_projectile_weapon_admission_v2'
    assert protocol['reuse_native_cases'] == 48 and protocol['new_native_train_cases'] == 0
    configs = {str(p.resolve()):sha(p) for p in (a.protocol, a.spec)}
    if a.resume:
        run = read(root/'run.json'); state = read(root/'progress.json')
        assert not process_alive(state.get('runner_pid')) and not process_alive(state.get('child_pid'))
        assert run['configs'] == configs
    else:
        assert not root.exists(); root.mkdir(parents=True)
        paths = set()
        for folder, extensions in ((repo/'scripts', ('.py', '.ps1')), (repo/'internal', ('.go',)),
                                   (repo/'cmd', ('.go',)), (repo.parent/'yquake2/src', ('.c', '.h'))):
            paths.update(p.resolve() for p in folder.rglob('*') if p.is_file() and p.suffix in extensions)
        run = dict(configs=configs, source_files={str(p):sha(p) for p in sorted(paths)},
                   retained_native_train_cases=48, new_native_train_cases=0, native_eval_budget=40, cuda_eval_budget=32)
        save(root/'run.json', run)

    def unchanged():
        assert all(sha(p) == h for p,h in configs.items())
        assert all(sha(p) == h for p,h in run['source_files'].items()), 'Source changed'
        assert all(sha(p) == h for p,h in retained.items()), 'Retained evidence changed'

    retained = {}

    def pin(path, expected=None):
        path = Path(path).resolve(); digest = sha(path)
        assert expected is None or expected == digest, str(path)
        retained[str(path)] = digest
        return path

    def step(label, command, target):
        unchanged()
        if not (target/'complete.json').exists():
            assert not target.exists(), 'Partial output preserved: '+str(target)
            with (root/(label+'.log')).open('w', encoding='utf-8') as log:
                child = subprocess.Popen(list(map(str, command)), cwd=repo, stdout=log, stderr=log,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                save(root/'progress.json', dict(stage=label, runner_pid=os.getpid(), child_pid=child.pid))
                result = child.wait()
            assert result == 0, str(root/(label+'.log'))
        seal = read(target/'complete.json'); assert seal['state'] == 'complete'
        if 'report_sha256' in seal:
            assert sha(target/'report.json') == seal['report_sha256']
        unchanged()

    py = lambda script,*args:[sys.executable, repo/'scripts'/script, *args]
    try:
        original = Path(protocol['original_failed_workflow'])
        failure = read(pin(original/'progress.json', protocol['original_failure_sha256']))
        assert failure['stage'] == 'failed' and failure['error'] == 'Pre-fit coverage failed'
        assert not process_alive(failure['runner_pid']) and not (original/'explore/update').exists()
        old_run = read(pin(original/'run.json'))
        assert all(sha(p) == digest for p,digest in old_run['source_files'].items())
        pool = read(pin(original/'pool/report.json'))
        assert pool['state'] == 'complete' and pool['source_unchanged'] and pool['usable_captures'] == len(pool['jobs']) == 48
        old_admission = read(pin(original/'explore/mechanism-exposure/report.json'))
        assert not old_admission['gate_passed'] and not old_admission['checks']['first_update_enemy_coverage']
        assert all(value for key,value in old_admission['checks'].items() if key != 'first_update_enemy_coverage')
        assert sha(protocol['retained_control_training_protocol']) == protocol['retained_control_training_protocol_sha256']
        control = Path(protocol['trained_updates']['control'])
        control_seal = read(pin(control/'complete.json'))
        control_report = read(pin(control/'report.json', control_seal['report_sha256']))
        assert control_report['experiment_protocol_sha256'] == protocol['retained_control_training_protocol_sha256']
        assert control_report['scope_updates'] == 1 and control_report['nonweapon_outputs_frozen_exact']
        for name,key in (('weights.json','weights_sha256'), ('checkpoint.pt','checkpoint_sha256')):
            pin(control/name, control_seal[key])
        source = Path(protocol['initial_sources']['explore']); initializer_seal = read(pin(source/'complete.json'))
        initializer = read(pin(source/'report.json', initializer_seal['report_sha256']))
        assert initializer['source_actor_value_std_exact'] and initializer['unchanged_other_outputs_exact']
        assert initializer['actor_critic_resumed_steps_exact'] and initializer['metadata']['other_actor_parameters'] == 'frozen_exact'
        assert protocol['pre_fit_exposure']['first_update_minimum_enemy_velocity_rows'] == 0
        assert protocol['pre_fit_exposure']['first_update_minimum_enemy_velocity_cases'] == 0
        corpus = original/'explore/corpus'
        receipt = read(pin(corpus/'complete.json'))
        assert receipt['native_cases'] == receipt['cuda_cases'] == 24
        pin(corpus/'processing-train/merged/report.json')
        save(root/'retained-inputs.json', dict(inputs=retained, native_train_cases=48, new_native_train_cases=0))
        branch = root/'explore'; branch.mkdir(exist_ok=True)
        mechanism, weapon, update = (branch/name for name in ('mechanism-exposure', 'weapon-exposure', 'update'))
        step('mechanism-exposure', py('audit_combat_projectile_training_exposure.py', '--corpus', corpus,
            '--protocol', a.protocol.resolve(), '--round', 1, '--out', mechanism), mechanism)
        step('weapon-exposure', py('audit_combat_weapon_only_exposure.py', '--corpus', corpus,
            '--source', source, '--protocol', a.protocol.resolve(), '--out', weapon), weapon)
        assert read(mechanism/'report.json')['gate_passed'] and read(weapon/'report.json')['gate_passed']
        step('train-explore', py('train_combat_weapon_only_cuda.py', '--source', source, '--corpus', corpus,
            '--native-credit', original/'explore/native-credit', '--advantage-credit', original/'explore/advantage-credit',
            '--weapon-exposure', weapon, '--mechanism-exposure', mechanism, '--protocol', a.protocol.resolve(), '--out', update), update)
        trained = read(update/'report.json')
        assert trained['scope_updates'] == 1 and trained['nonweapon_outputs_frozen_exact']
        assert trained['disposable_resume_adam_step_exact'] and trained['disposable_critic_resume_adam_step_exact']
        evaluation, quality, result = (root/name for name in ('evaluation', 'native-quality', 'result'))
        step('evaluation', py('evaluate_registered_combat_models.py', '--spec', a.spec.resolve(), '--out', evaluation), evaluation)
        step('native-quality', py('audit_registered_native_action_quality.py', '--evaluation', evaluation, '--out', quality), quality)
        step('result', py('report_combat_weapon_only_training_v2.py', '--workflow', root,
            '--protocol', a.protocol.resolve(), '--spec', a.spec.resolve(), '--out', result), result)
        unchanged()
        save(root/'complete.json', dict(state='complete', retained_native_train_cases=48, new_native_train_cases=0,
            retained_cuda_train_cases=48, native_eval_cases=40, cuda_eval_cases=32, original_pilot_failed=True,
            result_seal_sha256=sha(result/'complete.json'), source_unchanged=True, promotion=False))
        save(root/'progress.json', dict(stage='complete', runner_pid=os.getpid(), child_pid=None))
    except BaseException as error:
        save(root/'progress.json', dict(stage='failed', runner_pid=os.getpid(), error=str(error)))
        raise


if __name__ == '__main__':
    main()

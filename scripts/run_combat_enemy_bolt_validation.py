"""Twenty native cases: four old-world parity repeats plus sixteen fresh TRAIN cases."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from evaluate_registered_combat_models import read, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = a.out.resolve()
    assert not root.exists()
    protocol = read(a.protocol)
    assert protocol['native_budget'] == protocol['cuda_budget'] == 20
    root.mkdir(parents=True)
    paths = {a.protocol.resolve()}
    for folder, extensions in ((repo/'scripts', ('.py', '.ps1')), (repo/'internal', ('.go',)),
                               (repo/'cmd', ('.go',)), (repo.parent/'yquake2/src', ('.c', '.h'))):
        paths.update(p.resolve() for p in folder.rglob('*') if p.is_file() and p.suffix in extensions)
    paths.update(Path(p) for p in protocol['native_binaries'])
    pins = {str(p): sha(p) for p in sorted(paths)}
    save(root/'run.json', dict(protocol_sha256=sha(a.protocol), inputs=pins, native_budget=20, cuda_budget=20,
                              technical_repeat_cases=4, fresh_train_cases=16, fitting=False))
    env = dict(os.environ, GOCACHE=str(repo/'workspace/build/gocache'), GOTOOLCHAIN='auto')

    def unchanged():
        for p, digest in pins.items():
            assert sha(p) == digest, p

    def run(label, command):
        unchanged()
        with (root/(label+'.log')).open('w', encoding='utf-8') as log:
            child = subprocess.Popen(list(map(str, command)), cwd=repo, env=env, stdout=log, stderr=log,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            save(root/'progress.json', dict(stage=label, runner_pid=os.getpid(), child_pid=child.pid))
            code = child.wait()
        assert code == 0, str(root/(label+'.log'))
        unchanged()

    py = lambda name, *args: [sys.executable, repo/'scripts'/name, *args]
    try:
        fresh = root/'fresh'
        run('registry', py('prepare_combat_projectile_group_pilot.py', '--source', protocol['source'],
            '--config', protocol['config'], '--reward', protocol['reward'], '--out', fresh))
        prepared = read(fresh/'prepared.json')
        run('fresh-prepare', py('collect_combat_registry_curriculum.py', '--source', protocol['source'],
            '--registry', prepared['registry'], '--episodes', ','.join(prepared['episodes']), '--execution-contract', 'legacy_v1',
            '--history-spec', fresh/'history.json', '--train-count', 2, '--train-offset', 2,
            '--allow-groups', '--ppo-ready', '--train-only', '--prepare-only', '--out', fresh/'corpus'))
        planner = fresh/'q2episode.exe'
        reference = read(protocol['reference_plan'])
        names = [task['episode']['id'] for task in reference['tasks'] if '-site-01-' in task['episode']['id']]
        assert len(names) == 4
        parity = root/'parity-plan.json'
        run('parity-prepare', [planner, '--root', repo, '--registry', protocol['reference_registry'], '--episodes', ','.join(names),
            '--split', 'train', '--mode', 'learned', '--model', Path(protocol['source'])/'weights.json', '--count', 1,
            '--seed-offset', 0, '--checked-wall-slide', '--execution-contract', 'legacy_v1', '--out', parity,
            '--artifacts', root/'parity-captures'])
        old = {task['episode']['id']: task for task in reference['tasks']}
        for task in read(parity)['tasks']:
            before = old[task['episode']['id']]
            assert task['seeds'] == before['seeds'][:1] and task['instances'] == before['instances'][:1]
        save(root/'parity-binding.json', dict(reference_plan=protocol['reference_plan'], reference_sha256=sha(protocol['reference_plan']),
            parity_plan_sha256=sha(parity), same_seed_and_instances=True, technical_replay_only=True, fitting=False))
        plans = [parity, fresh/'corpus/train-plan.json']
        literal = lambda p: "'" + str(p).replace("'", "''") + "'"
        command = '& '+literal(repo/'scripts/run_registered_combat_pool.ps1')+' -Plans @('+','.join(map(literal, plans))+')'
        command += ' -MaxInstances 16 -Port '+str(protocol['port'])+' -OutputRoot '+literal(root/'pool')
        shell = shutil.which('pwsh')
        assert shell
        run('native-pool', [shell, '-NoProfile', '-Command', command])
        pool = read(root/'pool/report.json')
        assert pool['state'] == 'complete' and pool['source_unchanged'] and pool['usable_captures'] == len(pool['jobs']) == 20
        for index, label in enumerate(('parity', 'fresh')):
            run('process-'+label, py('process_registered_combat_pool_cuda.py', '--pool', root/'pool', '--plan-index', index,
                                    '--out', root/('processing-'+label)))
            report = read(root/('processing-'+label)/'report.json')
            assert report['state'] == 'complete' and report['device'] == 'cuda' and not report['training_performed']
            assert len(report['cases']) == (4 if index == 0 else 16)
        unchanged()
        save(root/'complete.json', dict(state='complete', version='combat_enemy_bolt_validation_collection_v1',
            native_cases=20, cuda_cases=20, technical_repeat_cases=4, fresh_train_cases=16,
            protocol_sha256=sha(a.protocol), pool_sha256=sha(root/'pool/report.json'), source_unchanged=True,
            fitting=False, promotion=False, parity_audit_pending=True, exposure_audit_pending=True))
        save(root/'progress.json', dict(stage='complete', runner_pid=os.getpid(), child_pid=None))
    except BaseException as error:
        save(root/'progress.json', dict(stage='failed', runner_pid=os.getpid(), error=str(error)))
        raise


if __name__ == '__main__':
    main()

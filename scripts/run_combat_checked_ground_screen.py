"""Run a frozen guard-only comparison and native diagnostics, without training."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from evaluate_registered_combat_models import read, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spec', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = a.out.resolve()
    assert not root.exists(), 'Fresh output required; failed attempts are preserved'
    spec = read(a.spec)
    prerequisite = Path(spec['native_prerequisite'])
    assert sha(prerequisite) == spec['native_prerequisite_sha256']
    proof = read(prerequisite)
    assert proof['mechanical_gate_passed'] and proof['cuda_cases'] == 12
    assert not proof['training_performed'] and not proof['guard_default_changed']
    root.mkdir(parents=True)
    paths = {a.spec.resolve(), prerequisite, Path(__file__).resolve()}
    for directory, extensions in ((repo/'scripts', ('.py', '.ps1')), (repo/'internal', ('.go',)),
                                  (repo/'cmd', ('.go',)), (repo.parent/'yquake2/src', ('.c', '.h'))):
        paths.update(p.resolve() for p in directory.rglob('*') if p.is_file() and p.suffix in extensions)
    pins = {str(p): sha(p) for p in sorted(paths)}
    save(root/'run.json', dict(source_files=pins, training_performed=False, promotion=False))
    env = dict(os.environ, GOCACHE=str(repo/'workspace/build/gocache'), GOTOOLCHAIN='auto')
    try:
        for label, command in (
            ('evaluation', ['scripts/evaluate_registered_combat_models.py', '--spec', a.spec.resolve(), '--out', root/'evaluation']),
            ('native-quality', ['scripts/audit_registered_native_action_quality.py', '--evaluation', root/'evaluation', '--out', root/'native-quality']),
            ('result', ['scripts/report_combat_checked_ground_screen.py', '--evaluation', root/'evaluation', '--spec', a.spec.resolve(),
                        '--quality', root/'native-quality', '--out', root/'result'])):
            assert all(sha(p) == h for p, h in pins.items()), 'Source changed'
            with (root/(label+'.log')).open('w', encoding='utf-8') as log:
                child = subprocess.Popen([sys.executable, *map(str, command)], cwd=repo, env=env,
                    stdout=log, stderr=log, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                save(root/'progress.json', dict(stage=label, runner_pid=os.getpid(), child_pid=child.pid))
                result = child.wait()
            assert result == 0, str(root/(label+'.log'))
            assert read(root/label/'complete.json')['state'] == 'complete'
        assert all(sha(p) == h for p, h in pins.items()), 'Source changed'
        save(root/'complete.json', dict(state='complete', result_seal_sha256=sha(root/'result/complete.json'),
            spec_sha256=sha(a.spec), source_unchanged=True, training_performed=False, promotion=False))
        save(root/'progress.json', dict(stage='complete', runner_pid=os.getpid()))
    except BaseException as error:
        save(root/'progress.json', dict(stage='failed', runner_pid=os.getpid(), error=str(error)))
        raise


if __name__ == '__main__':
    main()

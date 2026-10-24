"""Wait for one owned workflow, then close native diagnostics and fixed decision."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

from evaluate_registered_combat_models import read, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workflow', 'spec', 'quality', 'damage', 'result', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    args.out.mkdir(parents=True)
    save(args.out / 'progress.json', dict(stage='waiting-for-evaluation'))
    deadline = time.monotonic() + 7200
    try:
        while not (args.workflow / 'complete.json').exists():
            progress = args.workflow / 'progress.json'
            if progress.exists():
                assert read(progress)['stage'] != 'failed', 'Workflow failed; inspect original, no restart'
            assert time.monotonic() < deadline, 'Observation deadline expired; no automatic restart'
            time.sleep(5)
        evaluation = args.workflow / 'evaluation'
        operations = [
            ('native-quality', ['scripts/audit_registered_native_action_quality.py', '--evaluation', evaluation, '--out', args.quality]),
            ('damage', ['scripts/audit_combat_damage_distribution.py', '--evaluation', evaluation, '--out', args.damage]),
            ('decision', ['scripts/report_combat_blocked_ground_ppo.py', '--workflow', args.workflow, '--spec', args.spec,
                          '--quality', args.quality, '--damage', args.damage, '--out', args.result])]
        for name, command in operations:
            save(args.out / 'progress.json', dict(stage=name))
            with (args.out / (name + '.log')).open('w', encoding='utf-8') as log:
                subprocess.run([sys.executable, *map(str, command)], stdout=log, stderr=log, check=True)
        save(args.out / 'progress.json', dict(stage='complete', gate_passed=read(args.result / 'report.json')['gate_passed']))
        save(args.out / 'complete.json', dict(state='complete', result=str(args.result.resolve())))
    except BaseException as error:
        save(args.out / 'progress.json', dict(stage='failed', error=str(error)))
        raise


if __name__ == '__main__':
    main()

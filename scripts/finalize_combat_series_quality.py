"""Close extra shot-waste/comparison diagnostics after an actual evaluation process."""
import argparse
import pathlib
import sys

from process_combat_architecture_pool import read, save, sha, run
from run_combat_spatial_ppo import wait_process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    root = args.root.resolve()
    queue_sha = sha(root / 'queue.json')
    status_path = root / 'analysis-progress.json'
    try:
        if args.wait_pid:
            wait_process(args.wait_pid, status_path, 'waiting_for_native_evaluation')
        assert sha(root / 'queue.json') == queue_sha
        status = read(root / 'progress.json')
        assert status['stage'] == 'complete' and status['diagnostics_complete']
        for filename, key in (
                ('quality-report.json', 'quality_report_sha256'),
                ('diagnostics-acceptance.json', 'diagnostics_acceptance_sha256'),
                ('ownership-acceptance.json', 'ownership_acceptance_sha256'),
                ('first-shot-latency.json', 'first_native_shot_sha256'),
                ('outcome-behavior.json', 'outcome_behavior_sha256'),
                ('storage-final.json', 'physical_storage_sha256')):
            assert sha(root / filename) == status[key]
        assert read(root / 'ownership-acceptance.json')['clean']
        pool = read(root / 'pool/report.json')
        assert pool['state'] == 'complete' and pool['source_unchanged']
        assert not any(job['error'] for job in pool['jobs'])
        assert len(pool['jobs']) == read(root / 'protocol.json')['total_episodes']
        save(status_path, dict(stage='closing_extra_diagnostics', queue_sha256=queue_sha))
        for script, log, extra in (
                ('report_combat_adaptation_comparison.py', 'comparison.log', []),
                ('audit_combat_machinegun_waste.py', 'machinegun-waste.log', []),
                ('audit_combat_miss_reward.py', 'blaster-miss-credit.log', ['--min-frame', '100'])):
            run([sys.executable, repo / 'scripts' / script, '--root', root, *extra], root / log)
        receipts = dict(comparison_sha256=sha(root / 'adaptation-comparison.json'),
                        machinegun_waste_sha256=sha(root / 'machinegun-waste-audit.json'),
                        blaster_miss_credit_sha256=sha(root / 'miss-reward-audit-after-frame-100.json'))
        status.update(receipts)
        save(root / 'progress.json', status)
        save(status_path, dict(stage='complete', queue_sha256=queue_sha, **receipts,
                              scope='Read-only closed native captures; no training, recapture or promotion.'))
    except Exception as error:
        save(status_path, dict(stage='failed', error=str(error), queue_sha256=queue_sha))
        raise


if __name__ == '__main__':
    main()

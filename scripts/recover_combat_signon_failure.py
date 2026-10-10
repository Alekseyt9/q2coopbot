"""Retry empty signon failures after a terminal pool; preserve failed attempts.

Never retries gameplay losses or partial combat traces. Full native member
verification remains mandatory before producing comparative results.
"""
import argparse, datetime, pathlib, shutil, sys, time
from process_combat_architecture_pool import read, save, sha, run
from run_combat_spatial_ppo import wait_process


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=pathlib.Path, required=True)
    ap.add_argument('--wait-pid', type=int)
    ap.add_argument('--port', type=int, default=35780)
    a = ap.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    root = a.root.resolve()
    assert root.is_relative_to(repo / 'workspace/artifacts')
    assert 1024 <= a.port <= 65535
    recovery = root / 'recovery'
    recovery.mkdir(exist_ok=True)
    progress = recovery / 'signon-progress.json'
    try:
        if a.wait_pid:
            wait_process(a.wait_pid, progress)
        assert read(root / 'progress.json')['stage'] == 'failed'
        pool = read(root / 'pool/report.json')
        assert pool['state'] == 'failed' and pool['source_unchanged']
        protocol = read(root / 'protocol.json')
        paths = sorted((root / 'pool/jobs').glob('*-result.json'))
        assert len(paths) == protocol['total_episodes']
        failures = [(p, read(p)) for p in paths if read(p)['error']]
        assert failures, 'No failed jobs to recover'
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        archive = recovery / ('signon-' + stamp)
        archive.mkdir()
        receipts = []
        for path, job in failures:
            folder = pathlib.Path(job['root']).resolve()
            assert folder.is_relative_to(root) and folder != root
            report = read(folder / 'report.json')
            assert not report['capture_complete'] and len(report['results']) == 1
            result = report['results'][0]
            assert result['error'] == 'Empty command trace'
            worker = pathlib.Path(result['root'])
            assert worker.resolve().is_relative_to(folder)
            assert (worker / 'bot.jsonl').stat().st_size == 0
            server = (worker / 'server.log').read_text(errors='replace')
            assert 'SZ_GetSpace: overflow' in server and 'overflowed' in server
            assert 'command=configstrings' in server
            entry = protocol['evaluations'][job['plan_index']]
            assert sha(entry['plan']) == entry['plan_sha256']
            plan = read(entry['plan'])
            task = plan['tasks'][job['task_index']]
            si = task['seeds'].index(job['seed'])
            assert folder == pathlib.Path(entry['root']) / f"case-{job['task_index']}-{job['mode']}" / f"s-{job['seed']}"
            manifest = read(folder / 'manifest.json')
            original = archive / f"job-{job['job']}"
            assert original.resolve().is_relative_to(root)
            shutil.copy2(path, archive / path.name)
            folder.rename(original)
            started = datetime.datetime.now(datetime.timezone.utc).isoformat()
            timer = time.perf_counter()
            save(progress, dict(stage='retrying_empty_signon', job=job['job'], original=str(original)))
            run([repo / 'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe',
                 '-NoProfile', '-File', repo / 'scripts/run_registered_combat_pool_episode.ps1',
                 '-Plan', entry['plan'], '-TaskIndex', job['task_index'], '-SeedIndex', si,
                 '-Mode', job['mode'], '-Port', a.port, '-OutputRoot', folder,
                 '-BinaryBundle', manifest['binary_bundle']], archive / f"job-{job['job']}-retry.log")
            newreport = read(folder / 'report.json')
            newmanifest = read(folder / 'manifest.json')
            assert newreport['capture_complete'] and newreport['provenance_valid']
            for key in ('source_fingerprint', 'native_source_fingerprint', 'model_weights_sha256',
                        'reward_config_sha256', 'client_sha256', 'exporter_sha256'):
                assert newmanifest[key] == manifest[key]
            updated = dict(job, error='', port=a.port, slot=None, started_utc=started,
                           finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           wall_seconds=time.perf_counter()-timer,
                           original_failed_attempt=str(original), recovery_reason='empty_signon_buffer_overflow')
            save(path, updated)
            receipts.append(dict(job=job['job'], original=str(original),
                                 original_job_sha256=sha(archive / path.name),
                                 replacement_job_sha256=sha(path)))
        save(archive / 'receipt.json', dict(version='empty_signon_recovery_v1',
             protocol_sha256=sha(root / 'protocol.json'), attempts=receipts,
             scope='Only empty pre-combat signon failures retried; original pool aggregate retained.'))
        run([sys.executable, repo / 'scripts/verify_combat_evaluation_members.py', '--root', root,
             '--source-fingerprint', manifest['source_fingerprint'],
             '--native-fingerprint', manifest['native_source_fingerprint']], archive / 'verify.log')
        run([sys.executable, repo / 'scripts/report_combat_architecture_evaluation.py', '--root', root,
             '--member-proof', recovery / 'verified-members.json'], archive / 'quality.log')
        for tool in ('evaluation_strata', 'selected_target_aim', 'blaster_hits', 'aim_modes'):
            run([sys.executable, repo / f'scripts/report_combat_{tool}.py', '--root', root], archive / f'{tool}.log')
        save(root / 'progress.json', dict(stage='complete', quality_report_sha256=sha(root / 'quality-report.json'),
             recovery_receipt_sha256=sha(archive / 'receipt.json'), promotion='Not assessed'))
        run([sys.executable, repo / 'scripts/report_combat_movement.py', '--root', root], archive / 'movement.log')
        save(progress, dict(stage='complete', episodes=protocol['total_episodes'], retried=len(receipts)))
    except Exception as error:
        save(progress, dict(stage='failed', error=str(error)))
        raise


if __name__ == '__main__':
    main()

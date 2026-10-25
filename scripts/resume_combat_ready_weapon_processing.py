"""Resume only offline processing of a closed native pool after receipt-reader repair."""
import argparse
import concurrent.futures
from pathlib import Path
import subprocess
import sys
from evaluate_registered_combat_models import read, save, sha
from process_combat_architecture_pool import compact_closed_exports
from process_coop_learning_pool_cuda import verify_memory_export


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--probe', type=Path, required=True)
    ap.add_argument('--revision', default='processing-v2')
    a = ap.parse_args(); root = a.probe.resolve(); repo = Path(__file__).resolve().parents[1]
    assert a.revision == Path(a.revision).name and a.revision.startswith('processing-')
    out = root/a.revision; assert not out.exists() and not (root/'complete.json').exists()
    pool, spec = read(root/'pool/report.json'), read(root/'spec.json')
    assert pool['state'] == 'complete' and pool['source_unchanged'] and pool['usable_captures'] == 16
    assert len(pool['jobs']) == 16 and all(not j['error'] for j in pool['jobs'])
    out.mkdir(); inputs = {}
    for p in (repo/'scripts').glob('*.py'): inputs[str(p.resolve())] = sha(p)
    for p in [repo/'cmd/q2ppo-data/main.go', repo/'internal/learningenv/rng_start.go',root/'spec.json',root/'pool/report.json']:
        inputs[str(p.resolve())] = sha(p)
    def run(label, command):
        with (out/(label+'.log')).open('w', encoding='utf-8') as stream:
            r = subprocess.run(list(map(str, command)), cwd=repo, stdout=stream, stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert r.returncode == 0, str(out/(label+'.log'))
    def stage(stage): save(out/'progress.json', dict(stage=stage))
    try:
        exporter = out/'q2ppo-data.exe'
        run('build', ['go', 'build', '-o', exporter, './cmd/q2ppo-data'])
        inputs[str(exporter)] = sha(exporter)
        # Preserve every original capture and failed processing attempt.
        def export(job):
            i = job['plan_index']; v = spec['variants'][i]; capture = Path(job['root']); directory = out/f'case-{i:02d}'
            directory.mkdir(); result = read(capture/'report.json')['results'][0]
            assert result['capture_valid'] and result['dispatch_valid'] and result['generated_fixture'] == v['instance']
            assert sha(v['model']) == v['model_sha256'] == spec['source_weights_sha256']
            run(f'export-{i:02d}', [exporter, '--batch', capture, '--model', v['model'], '--cuda-only', '--out', directory/'native'])
            verify_memory_export(Path(result['root'])/'bot.jsonl', directory/'native/replay-0')
            compact_closed_exports(directory/'native')
            return dict(model=v['model'], model_sha256=v['model_sha256'], data=str(directory/'native'),
                        out=str(directory/'cuda'), native_report_sha256=sha(directory/'native/report.json'))
        stage('native-export')
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as workers:
            tasks = list(workers.map(export, sorted(pool['jobs'], key=lambda j: j['plan_index'])))
        save(out/'cuda-request.json', dict(version='combat_cuda_finalize_batch_v1', compact_finalized_streams=True, tasks=tasks))
        stage('cuda-verification')
        run('cuda-batch', [sys.executable, repo/'scripts/finalize_combat_cuda_batch.py', '--request', out/'cuda-request.json', '--receipt-root', out/'cuda-batch'])
        batch = read(out/'cuda-batch/report.json'); assert batch['state'] == 'complete' and len(batch['tasks']) == 16
        assert all(sha(p) == h for p, h in inputs.items())
        save(out/'provenance.json', dict(inputs=inputs, original_source_snapshot=str(root/'source-snapshot/manifest.json'),
            original_source_snapshot_sha256=sha(root/'source-snapshot/manifest.json'),
            change='Added Shotgun19..36 idle-frame receipt reader and loadout mapping; native capture untouched. Prior attempts retained.',
            new_native_games=0, training_performed=False))
        save(root/'complete.json', dict(state='complete', version=spec['version'], native_cases=16, cuda_cases=16,
            processing_root=a.revision, spec_sha256=sha(root/'spec.json'), pool_sha256=sha(root/'pool/report.json'),
            cuda_receipt_sha256=sha(out/'cuda-batch/report.json'), processing_provenance_sha256=sha(out/'provenance.json'),
            diagnostic_only=True, training_performed=False, promotion=False))
        stage('acceptance')
        run('acceptance', [sys.executable, repo/'scripts/audit_combat_ready_weapon_probe.py', '--probe', root, '--out', out/'acceptance'])
        stage('complete'); print(out/'acceptance/report.json')
    except BaseException as error:
        save(out/'progress.json', dict(stage='failed', error=str(error))); raise


if __name__ == '__main__': main()

"""Finalize a closed registry pool on CUDA; optionally update from train splits only."""
import argparse
import concurrent.futures
import pathlib
import subprocess
import sys

from process_combat_architecture_pool import read, save, sha
from process_coop_learning_pool_cuda import verify_memory_export


def run(command, log, repo):
    with log.open('w', encoding='utf-8') as output:
        result = subprocess.run([str(v) for v in command], cwd=repo,
                                stdout=output, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    assert result.returncode == 0, str(log)


def main():
    parser = argparse.ArgumentParser()
    for name in ('pool', 'out'):
        parser.add_argument('--'+name, type=pathlib.Path, required=True)
    for name in ('config', 'resume', 'anchor', 'bank'):
        parser.add_argument('--'+name, type=pathlib.Path)
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parent.parent
    root, out = args.pool.resolve(), args.out.resolve()
    report = read(root/'report.json')
    assert report['state'] == 'complete' and report['source_unchanged']
    assert report['usable_captures'] == len(report['jobs'])
    assert all(not job['error'] and job['mode'] == 'learned' for job in report['jobs'])
    plans = []
    for binding in report['plans']:
        path = pathlib.Path(binding['path'])
        assert sha(path) == binding['sha256']
        plans.append(read(path))
    models = {str(pathlib.Path(p['model_path']).resolve()) for p in plans}
    assert len(models) == 1
    model = pathlib.Path(models.pop())
    assert all(sha(model) == p['model_sha256'] for p in plans)
    if args.config:
        assert args.resume and args.anchor and args.bank
        assert all(t['split'] == 'train' for p in plans for t in p['tasks']), 'Validation/test data cannot train'
    assert not out.exists(), 'Fresh processing output required'
    out.mkdir(parents=True)
    pool_sha = sha(root/'report.json')
    try:
        exporter = out/'q2ppo-data.exe'
        run(['go', 'build', '-o', exporter, './cmd/q2ppo-data'], out/'build.log', repo)
        jobs = sorted(report['jobs'], key=lambda j: j['job'])
        cases = []
        for job in jobs:
            plan = plans[job['plan_index']]
            task = plan['tasks'][job['task_index']]
            assert job['seed'] in task['seeds']
            case = pathlib.Path(job['root'])
            manifest = read(case/'manifest.json')
            assert manifest['model_weights_sha256'].lower() == sha(model)
            directory = out/f"case-{job['job']:04d}"
            directory.mkdir()
            cases.append((job, case, directory))

        def native(case_info):
            job, case, directory = case_info
            run([exporter, '--batch', case, '--model', model, '--cuda-only',
                 '--out', directory/'native'], directory/'native.log', repo)
            original = read(case/'report.json')
            for i, result in enumerate(original['results']):
                verify_memory_export(pathlib.Path(result['root'])/'bot.jsonl',
                                     directory/'native'/f'replay-{i}')
            return dict(job=job['job'], seed=job['seed'], split=plans[job['plan_index']]['tasks'][job['task_index']]['split'])

        save(out/'progress.json', dict(stage='native', cases=len(cases)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as workers:
            native_cases = list(workers.map(native, cases))
        request = out/'cuda-request.json'
        save(request, dict(version='combat_cuda_finalize_batch_v1', compact_finalized_streams=True,
                          tasks=[dict(model=str(model), model_sha256=sha(model),
                                      data=str(directory/'native'), out=str(directory/'cuda'),
                                      native_report_sha256=sha(directory/'native/report.json'))
                                 for _, _, directory in cases]))
        save(out/'progress.json', dict(stage='cuda-finalize', cases=len(cases)))
        run([sys.executable, repo/'scripts/finalize_combat_cuda_batch.py', '--request', request,
             '--receipt-root', out/'cuda-batch'], out/'cuda.log', repo)
        run([exporter, '--merge', ','.join(str(d/'cuda') for _, _, d in cases),
             '--out', out/'merged'], out/'merge.log', repo)
        merged = read(out/'merged/report.json')
        result = dict(state='complete', device='cuda', pool_sha256=pool_sha,
                      model_sha256=sha(model), cases=native_cases, rows=merged['rows'],
                      training_performed=False, promotion='Not evaluated')
        if args.config:
            save(out/'progress.json', dict(stage='gpu-training', rows=merged['rows']))
            run([sys.executable, repo/'scripts/ppo_recurrent.py', '--model', model,
                 '--data', out/'merged', '--config', args.config.resolve(), '--resume', args.resume.resolve(),
                 '--anchor-model', args.anchor.resolve(), '--retention-bank', args.bank.resolve(),
                 '--retention-weight', '0', '--bank-weight', '0', '--out', out/'update'], out/'train.log', repo)
            update = read(out/'update/report.json')
            assert update['device'] == 'cuda' and update['weights_sha256'] == sha(out/'update/weights.json')
            result.update(training_performed=True, update=update)
        assert sha(root/'report.json') == pool_sha
        save(out/'report.json', result)
        save(out/'progress.json', dict(stage='complete', rows=merged['rows']))
        print(dict(state='complete', cases=len(cases), rows=merged['rows'], training=result['training_performed']), flush=True)
    except Exception as error:
        save(out/'progress.json', dict(stage='failed', error=str(error)))
        raise


if __name__ == '__main__':
    main()

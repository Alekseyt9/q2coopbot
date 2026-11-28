"""Export and fit one ordinary-GAE U1 round from a completed TRAIN pool."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

from evaluate_registered_combat_models import read, save, sha


def fit(plan_path, pool_path, model, checkpoint, config, output, gae_lambda=None):
    scripts = Path(__file__).resolve().parent
    plan, pool = read(plan_path), read(pool_path)
    plan_hash, model_hash = sha(plan_path), sha(model)
    if pool['state'] != 'complete' or not pool['source_unchanged']:
        raise ValueError('TRAIN pool must be complete with its frozen source fence')
    bound = [i for i, p in enumerate(pool['plans'])
             if p['sha256'] == plan_hash and p['model_sha256'] == model_hash]
    if len(bound) != 1 or plan['model_sha256'] != model_hash:
        raise ValueError('Pool must bind one exact plan and behavior model')
    expected = {seed for task in plan['tasks'] for seed in task['seeds']}
    jobs = sorted((job for job in pool['jobs'] if job['plan_index'] == bound[0]),
                  key=lambda job: job['seed'])
    if (any(task['split'] != 'train' or task['modes'] != ['learned'] for task in plan['tasks'])
            or len(jobs) != len(expected) or {job['seed'] for job in jobs} != expected
            or any(job['mode'] != 'learned' or job['error'] for job in jobs)):
        raise ValueError('Expected every planned TRAIN job, without outcome filtering')
    output.mkdir()
    exporter = output/'q2ppo-data.exe'
    subprocess.run(['go','build','-buildvcs=false','-o',str(exporter),'./cmd/q2ppo-data'],
                   cwd=scripts.parent, check=True)

    def export(job):
        capture = Path(job['root']).resolve()
        target = output/f"s-{job['seed']}"
        target.mkdir()
        native = target/'native'
        with (target/'native.log').open('wb') as log:
            subprocess.run([exporter, '--batch', capture, '--model', model, '--cuda-only',
                            '--out', native], check=True, stdout=log, stderr=subprocess.STDOUT)
        cuda = target/'cuda'
        task = dict(model=str(model), model_sha256=model_hash, data=str(native), out=str(cuda),
                    native_report_sha256=sha(native/'report.json'))
        return task, dict(seed=job['seed'], capture_root=str(capture), data=str(cuda))

    with ThreadPoolExecutor(4) as workers:
        tasks, cases = zip(*workers.map(export, jobs))
    cuda_request = output/'cuda-request.json'
    save(cuda_request, dict(version='combat_cuda_finalize_batch_v1', compact_finalized_streams=True,
                            tasks=tasks))
    subprocess.run([sys.executable, scripts/'finalize_combat_cuda_batch.py', '--request',
                    cuda_request, '--receipt-root', output/'cuda-batch'], check=True)
    train_request = output/'train-request.json'
    save(train_request, dict(version='combat_full_joint_train_request_v1', split='train',
                             plan=str(plan_path), plan_sha256=plan_hash, pool_report=str(pool_path),
                             pool_report_sha256=sha(pool_path), cases=cases))
    command = [sys.executable, scripts/'train_combat_full_joint_ppo_cuda.py',
               '--model', model, '--resume', checkpoint, '--config', config,
               '--request', train_request, '--out', output/'fit']
    if gae_lambda is not None:
        command.extend(['--gae-lambda', str(gae_lambda)])
    subprocess.run(command, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'pool-report', 'model', 'checkpoint', 'config', 'output'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    parser.add_argument('--gae-lambda', type=float, help='Override GAE lambda for this update')
    args = parser.parse_args()
    if args.gae_lambda is not None and not 0 <= args.gae_lambda <= 1:
        parser.error('--gae-lambda must be between 0 and 1')
    fit(*(getattr(args, name).resolve() for name in
          ('plan', 'pool_report', 'model', 'checkpoint', 'config', 'output')),
        gae_lambda=args.gae_lambda)

"""Finalize independent native corpora in one CUDA process; preserve case boundaries."""
import argparse
import json
import pathlib
import time

from process_combat_architecture_pool import read, save, sha
from finalize_combat_cuda_rollout import finalize


def batch(request_path, receipt_root):
    request_path, receipt_root = request_path.resolve(), receipt_root.resolve()
    request_digest, script_digest = sha(request_path), sha(__file__)
    request = read(request_path)
    assert request['version'] == 'combat_cuda_finalize_batch_v1' and request['tasks']
    assert not receipt_root.exists(), 'Fresh batch receipt root required'
    tasks, outputs = [], set()
    for task in request['tasks']:
        model, data, out = (pathlib.Path(task[key]).resolve() for key in ('model', 'data', 'out'))
        assert sha(model) == task['model_sha256'] and sha(data/'report.json') == task['native_report_sha256']
        assert not out.exists() and out not in outputs and out != receipt_root
        assert out != data and data not in out.parents and out not in data.parents
        assert out not in request_path.parents and out not in model.parents
        outputs.add(out)
        tasks.append((task, model, data, out))
    receipt_root.mkdir(parents=True)
    started, completed = time.perf_counter(), []
    try:
        for index, (task, model, data, out) in enumerate(tasks):
            save(receipt_root/'progress.json', dict(stage='cuda-finalize', task=index, tasks=len(tasks)))
            assert sha(request_path) == request_digest and sha(__file__) == script_digest
            assert sha(model) == task['model_sha256'] and sha(data/'report.json') == task['native_report_sha256']
            case_started = time.perf_counter()
            audit = finalize(model, data, out)
            assert audit['state'] == 'passed' and audit['device'] == 'cuda'
            meta = read(out/'report.json')
            meta['source_sha256'].update({str(request_path): request_digest, str(pathlib.Path(__file__).resolve()): script_digest})
            save(out/'report.json', meta)
            completed.append(dict(task=index, model_sha256=task['model_sha256'],
                native_report_sha256=task['native_report_sha256'], out=str(out), report_sha256=sha(out/'report.json'),
                rollout_sha256=meta['rollout_sha256'], cuda_verification_sha256=meta['cuda_verification_sha256'],
                seconds=time.perf_counter()-case_started))
        assert sha(request_path) == request_digest and sha(__file__) == script_digest
        receipt = dict(version='combat_cuda_finalize_batch_receipt_v1', state='complete', device='cuda',
            request_sha256=request_digest, script_sha256=script_digest, tasks=completed,
            seconds=time.perf_counter()-started,
            scope='One interpreter/import lifetime; identical single-case finalize function called independently. Model modules and sequence context reconstructed for each case; no memory/context carried between fights. Case reports pin batch request and entrypoint. No CPU neural fallback or training.')
        save(receipt_root/'report.json', receipt)
        save(receipt_root/'progress.json', dict(stage='complete', report_sha256=sha(receipt_root/'report.json')))
        return receipt
    except Exception as error:
        save(receipt_root/'progress.json', dict(stage='failed', completed=len(completed), error=str(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=pathlib.Path, required=True)
    parser.add_argument('--receipt-root', type=pathlib.Path, required=True)
    args = parser.parse_args()
    print(json.dumps(batch(args.request, args.receipt_root)), flush=True)

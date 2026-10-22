"""Read-only held-out value diagnostic; never fit or select a checkpoint."""
import argparse
import os
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); assert not args.out.exists()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    assert torch.cuda.is_available()
    torch.set_default_device('cuda'); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    from migrate_combat_navigation_cuda import branch
    from ppo_recurrent import prepare_context, sequence
    from ppo_combat import advantages
    import json
    root = args.root.resolve(); protocol = read(root / 'protocol.json')
    source = Path(protocol['source']); corpus = Path(protocol['corpus'])
    assert read(root / 'pair-audit.json')['state'] == 'complete'
    data = corpus / 'processing-development/merged'; meta = read(data / 'report.json')
    assert meta['model_sha256'] == sha(source / 'weights.json')
    assert meta['rollout_sha256'] == sha(data / 'rollout.jsonl')
    assert meta['sequence_sha256'] == sha(data / 'sequence.jsonl')
    rows = [json.loads(s) for s in (data / 'rollout.jsonl').read_text().splitlines()]
    context = [json.loads(s) for s in (data / 'sequence.jsonl').read_text().splitlines()]
    prepared = prepare_context(context, rows, 'cuda')
    _, returns = advantages(rows, .99, 1.)
    targets = torch.tensor(returns)
    results = {}; inputs = []
    for name, update in [('source', source), ('legacy', root / 'legacy'), ('mc', root / 'mc-before')]:
        seal = read(update / 'complete.json'); weights = update / 'weights.json'
        assert sha(weights) == seal['weights_sha256']
        net = branch(read(weights), 'value')
        with torch.no_grad():
            output = sequence(net, prepared)[0][:, 0]
            errors = (output-targets).square()
            variance = targets.var(unbiased=False)
            case_results = []
            for seed in sorted({r['seed'] for r in rows}):
                selected = torch.tensor([i for i, row in enumerate(rows) if row['seed'] == seed], dtype=torch.long)
                case_results.append(dict(seed=seed, rows=len(selected), mse=float(errors[selected].mean()),
                                         prediction_mean=float(output[selected].mean()),
                                         target_mean=float(targets[selected].mean())))
            results[name] = dict(mse=float(errors.mean()), mae=float((output-targets).abs().mean()),
                                 explained_variance=float(1-(output-targets).var(unbiased=False)/variance),
                                 prediction_mean=float(output.mean()), cases=case_results)
        inputs.append(dict(path=str(weights), sha256=sha(weights)))
        del net
    for name in ['rollout.jsonl', 'sequence.jsonl', 'report.json']:
        inputs.append(dict(path=str(data / name), sha256=sha(data / name)))
    save(args.out, dict(state='complete', device='cuda', split='development', rows=len(rows),
         cases=len({r['seed'] for r in rows}), target='gamma=.99 lambda=1 observed returns and recorded boundary bootstrap',
         models=results, inputs=inputs, training_performed=False, selection_performed=False,
         promotion=False, script_sha256=sha(__file__),
         limitation='Held-out source-policy trajectories, not new-policy value calibration or causal combat benefit. Legacy critic trained on shorter GAE targets.'))
    print({name: {k:v for k,v in result.items() if k!='cases'} for name,result in results.items()})


if __name__ == '__main__':
    main()

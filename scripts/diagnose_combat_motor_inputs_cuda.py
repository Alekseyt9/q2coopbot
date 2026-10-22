"""Read-only CUDA motor offset variability and input-band sensitivity.

These are branch-only perturbations, not policy KL, native counterfactuals,
or evidence that a feature improves combat. Development rows are never fitted.
"""
import argparse
import json
import os
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha


BANDS = dict(common=(0, 20), geometry=(20, 73), observed_enemies=(73, 169),
             typed_enemies=(466, 786), navigation=(854, 881),
             remembered_threats=(881, 1121))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from combat_motor_residual import build, inputs
    assert torch.cuda.is_available(), 'CUDA required; no CPU fallback'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    pins = {}
    def pin(path):
        pins[str(path.resolve())] = sha(path)
    pin(args.corpus / 'complete.json')
    assert read(args.corpus / 'complete.json')['state'] == 'complete'
    modules = {}
    for path in args.models:
        seal = read(path / 'complete.json')
        for filename, key in [('weights.json', 'weights_sha256'),
                              ('checkpoint.pt', 'checkpoint_sha256')]:
            assert sha(path / filename) == seal[key]
            pin(path / filename)
        pin(path / 'complete.json')
        module = build(read(path / 'weights.json'))
        assert module is not None
        modules[str(path.resolve())] = module.eval()
    reports = {}
    with torch.no_grad():
        for split in ('train', 'development'):
            folder = args.corpus / ('processing-' + split) / 'merged'
            meta = read(folder / 'report.json')
            rows_path = folder / 'rollout.jsonl'
            assert sha(rows_path) == meta['rollout_sha256']
            pin(rows_path); pin(folder / 'report.json')
            with rows_path.open(encoding='utf-8-sig') as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
            x = torch.tensor([row['features'] for row in rows], device='cuda', dtype=torch.float32)
            assert x.shape == (len(rows), 1147) and torch.isfinite(x).all()
            variants = {}
            for label, module in modules.items():
                original = module(inputs(x))
                center = original.mean(0)
                deviations = (original - center).abs().mean(0)
                sensitivity = {}
                for band, (start, end) in BANDS.items():
                    # Deterministic roll breaks alignment without changing the
                    # band's empirical marginal. Cross-band combinations may
                    # still be out of distribution, so this is descriptive.
                    shuffled = x.clone()
                    shuffled[:, start:end] = x[:, start:end].roll(1, 0)
                    zeroed = x.clone(); zeroed[:, start:end] = 0
                    sensitivity[band] = dict(
                        cyclic_permutation_mean_abs_change=(module(inputs(shuffled)) - original).abs().mean(0).tolist(),
                        zero_mean_abs_change=(module(inputs(zeroed)) - original).abs().mean(0).tolist())
                variants[label] = dict(mean_offset=center.tolist(),
                    std_offset=original.std(0, unbiased=False).tolist(),
                    mean_abs_offset=original.abs().mean(0).tolist(),
                    mean_abs_deviation_from_constant=deviations.tolist(),
                    input_sensitivity=sensitivity)
            reports[split] = dict(rows=len(rows), variants=variants)
    for path, digest in pins.items():
        assert sha(Path(path)) == digest, 'Input changed: ' + path
    save(args.out, dict(state='complete', device='cuda', splits=reports,
        axes=['forward_gaussian_mean', 'side_gaussian_mean'],
        inputs=[dict(path=p, sha256=h) for p, h in pins.items()],
        script_sha256=sha(__file__), training_performed=False, promotion=False,
        scope='Branch-only descriptive sensitivity. No full-policy inference or KL, no causal combat claim, no development fitting or checkpoint selection. Permuted and zeroed inputs may be out of distribution.'))
    print(json.dumps(dict(state='complete', rows={k: v['rows'] for k, v in reports.items()})), flush=True)


if __name__ == '__main__':
    main()

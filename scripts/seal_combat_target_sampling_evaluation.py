"""Seal inference-only target-temperature weights after full CUDA and native unit acceptance."""
import argparse
import shutil
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('models', 'compatibility', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    closed = read(args.models / 'complete.json')
    assert sha(args.models / 'report.json') == closed['report_sha256']
    prepared = read(args.models / 'report.json')
    gate = read(args.compatibility / 'report.json')
    assert sha(args.compatibility / 'report.json') == read(args.compatibility / 'complete.json')['report_sha256']
    assert prepared['device'] == 'cuda' and prepared['zero_source_full_density_exact']
    assert gate['gate_passed'] and gate['native_cases'] == gate['cuda_cases'] == 8
    candidate = prepared['models']['calibrated']
    assert sha(candidate['path']) == candidate['sha256']
    args.out.mkdir(parents=True)
    shutil.copy2(candidate['path'], args.out / 'weights.json')
    save(args.out / 'report.json', dict(state='complete', version='combat_experimental_policy_weights_v1',
         device='cuda', evaluation_only=True, ppo_checkpoint_written=False, training_performed=False, promotion=False,
         native_compatibility_report=str((args.compatibility / 'report.json').resolve()),
         native_compatibility_sha256=sha(args.compatibility / 'report.json'),
         actor_preparation_report=str((args.models / 'report.json').resolve()),
         actor_preparation_sha256=sha(args.models / 'report.json'), source_weights_sha256=prepared['source_weights_sha256'],
         scope='Only categorical target sampling/log probability calibrated. Neural parameters and other conditional '
               'inputs unchanged. Native non-unit likelihood parity and combat quality remain to be tested. Training unsupported.'))
    save(args.out / 'complete.json', dict(state='complete', weights_sha256=sha(args.out / 'weights.json'),
         report_sha256=sha(args.out / 'report.json')))
    print(dict(weights_sha256=sha(args.out / 'weights.json'), evaluation_only=True))


if __name__ == '__main__':
    main()

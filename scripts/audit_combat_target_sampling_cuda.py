"""Describe frozen target probabilities on the existing TRAIN-only CUDA actor cache."""
import argparse
from pathlib import Path

import numpy as np
from evaluate_registered_combat_models import read, save, sha
from combat_target_head import distribution
from train_combat_bc import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'source', 'baseline', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available() and not args.out.exists()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    pin(__file__)
    source, baseline = (read(pin(p / 'weights.json')) for p in (args.source, args.baseline))
    for model in (source, baseline):
        assert model['feature_version'] == 'combat_features_v10'
    source.pop('log_std')
    baseline.pop('log_std')
    assert source == baseline, 'Cached raw actor must match baseline; only exploration std may differ'
    receipt = read(pin(args.cache / 'complete.json'))
    cache = read(pin(args.cache / 'report.json', receipt['report_sha256']))
    assert cache['source_roles'] == '24TRAIN only' and len(cache['chunks']) == 24
    assert cache['inputs'][str((args.source / 'weights.json').resolve())] == sha(args.source / 'weights.json')
    stats = {str(t): dict(rows=0, multi_enemy_rows=0, previous_valid_rows=0,
                         previous_is_mode=0, previous_probability_sum=0., entropy_sum=0.) for t in (1., .5)}
    rows = 0
    for chunk in cache['chunks']:
        assert chunk['split'] == 'train'
        with np.load(pin(chunk['path'], chunk['sha256'])) as arrays:
            x = torch.tensor(arrays['features'], dtype=torch.float64, device='cuda')
            raw = torch.tensor(arrays['raw_actor'], dtype=torch.float64, device='cuda')
        assert len(x) == chunk['rows'] and raw.shape == (len(x), 81)
        previous = x[:, 845:854]
        assert bool(((previous == 0) | (previous == 1)).all()) and bool((previous.sum(-1) == 1).all())
        index = previous.argmax(-1)
        for temperature in (1., .5):
            scoring = raw[:, :45].clone()
            scoring[:, 20:29] /= temperature
            choices = distribution(scoring, x)
            multi = (torch.isfinite(choices.logits[:, 1:]).sum(-1) >= 2)
            prior_available = torch.isfinite(choices.logits.gather(1, index[:, None])[:, 0])
            valid = multi & (index != 0) & prior_available
            probabilities = choices.probs.gather(1, index[:, None])[:, 0]
            stat = stats[str(temperature)]
            stat['rows'] += len(x)
            stat['multi_enemy_rows'] += int(multi.sum())
            stat['previous_valid_rows'] += int(valid.sum())
            stat['previous_is_mode'] += int(((choices.probs.argmax(-1) == index) & valid).sum())
            stat['previous_probability_sum'] += float(probabilities[valid].sum())
            stat['entropy_sum'] += float(choices.entropy()[valid].sum())
        rows += len(x)
    for stat in stats.values():
        n = stat['previous_valid_rows']
        stat['mean_previous_probability'] = stat['previous_probability_sum'] / n if n else None
        stat['expected_switch_probability'] = 1 - stat['mean_previous_probability'] if n else None
        stat['previous_is_mode_fraction'] = stat['previous_is_mode'] / n if n else None
        stat['mean_entropy'] = stat['entropy_sum'] / n if n else None
    for path, digest in pins.items():
        assert sha(path) == digest, path
    args.out.mkdir(parents=True)
    save(args.out / 'report.json', dict(state='complete', version='combat_target_sampling_train_audit_v1',
        device='cuda', rows=rows, statistics=stats, inputs=pins, training_performed=False, new_games=False,
        promotion=False, scope='Recorded TRAIN-only CUDA actor outputs. Compare original target distribution with '
        'a hypothetical temperature0.5 on identical observed states. No runtime implementation, fitting, '
        'movement-branch modification or proof that target switching is wrong or combat improves.'))
    save(args.out / 'complete.json', dict(state='complete', report_sha256=sha(args.out / 'report.json')))
    print(stats)


if __name__ == '__main__':
    main()

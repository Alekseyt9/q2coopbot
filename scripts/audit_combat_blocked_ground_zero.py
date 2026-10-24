"""Verify live parent/zero-control trajectories, not just equal model files."""
import argparse
import copy
import json
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha


def normalized(value):
    if isinstance(value, dict):
        return {key: (0 if key == 'spawncount' else normalized(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [normalized(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest == expected
        pins[str(path)] = digest
        return path

    pin(__file__)
    closed = read(pin(args.evaluation / 'complete.json'))
    assert closed['native_cases'] == closed['cuda_cases'] == 4
    fixture = read(pin(args.evaluation / 'fixture-proof.json'))
    models = {model['label']: model for model in fixture['models']}
    assert set(models) == {'parent', 'zero'}
    assert models['parent']['model_sha256'] == models['zero']['model_sha256']
    pool = read(pin(args.evaluation / 'pool/report.json', closed['pool_sha256']))
    labels = {i: Path(item['path']).stem.removesuffix('-plan') for i, item in enumerate(fixture['plans'])}
    records = {}
    for job in pool['jobs']:
        label = labels[job['plan_index']]
        folder = args.evaluation / ('processing-' + label) / f"case-{job['job']:04d}"
        meta = read(pin(folder / 'cuda/report.json'))
        proof = read(pin(folder / 'cuda/cuda-verification.json', meta['cuda_verification_sha256']))
        assert proof['state'] == 'passed' and proof['device'] == 'cuda'
        path = next((folder / 'native').glob('replay-*/steps.jsonl')).resolve()
        pin(path, meta['source_sha256'][str(path)])
        steps = [json.loads(line) for line in path.open(encoding='utf-8-sig')]
        rows = [normalized({key: row.get(key) for key in
                ('index', 'observation', 'next_observation', 'action', 'applied_action', 'sample',
                 'interventions', 'terminal', 'truncated', 'reason', 'native_step')})
                for row in steps if row['owner'] == 'provider' and row.get('sample')]
        assert rows
        records[label, job['seed']] = rows
    seeds = sorted(seed for label, seed in records if label == 'parent')
    assert len(seeds) == 2 and len(records) == 4
    for seed in seeds:
        assert records['parent', seed] == records['zero', seed], f'Live zero-control differs at seed {seed}'
    assert all(sha(path) == digest for path, digest in pins.items())
    args.out.mkdir(parents=True)
    save(args.out / 'report.json', dict(state='complete', native_cases=4, cuda_cases=4,
        matched_seeds=seeds, matched_provider_rows=sum(len(records['parent', seed]) for seed in seeds),
        spawncount_normalized_only=True, observations_actions_samples_next_states_exact=True,
        inputs=pins, neural_execution=False, new_games=False, training_performed=False))
    save(args.out / 'complete.json', dict(state='complete', report_sha256=sha(args.out / 'report.json')))
    print(dict(matched_seeds=seeds, state='passed'))


if __name__ == '__main__':
    main()

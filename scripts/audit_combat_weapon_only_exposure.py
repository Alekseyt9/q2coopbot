"""Pre-fit scalar admission of real TRAIN weapon opportunities and sampled choices."""
import argparse
import json
from collections import Counter
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('corpus', 'protocol', 'source', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    a = ap.parse_args(); assert not a.out.exists()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve(); digest = sha(path)
        assert expected is None or expected == digest, str(path)
        pins[str(path)] = digest
        return path

    protocol = read(pin(a.protocol))
    seal = read(pin(a.source/'complete.json'))
    source = read(pin(a.source/'report.json', seal['report_sha256']))
    profile = source.get('profile', source.get('metadata', {}).get('profile'))
    assert profile in ('control', 'explore')
    closed = read(pin(a.corpus/'complete.json'))
    collection = read(pin(a.corpus/'protocol.json', closed['protocol_sha256']))
    assert collection['train_only'] and all(x['split'] == 'train' for x in collection['assignments'])
    meta = read(pin(a.corpus/'processing-train/merged/report.json'))
    assert meta['model_sha256'] == seal['weights_sha256'] and meta['numerical_verification'] == 'cuda_verified_v1'
    assert meta['rows'] > 0
    data = pin(a.corpus/'processing-train/merged/rollout.jsonl', meta['rollout_sha256'])
    counts = Counter(rows=0, legal_blaster_alternative_rows=0, alternative_weapon_samples=0, omitted_keep_rows=0)
    cases, switch_cases = set(), set()
    with data.open(encoding='utf-8-sig') as stream:
        for row in map(json.loads, stream):
            counts['rows'] += 1
            x = row['features']; assert len(x) == 1489
            counts['omitted_keep_rows'] += 'weapon' not in row['sample']
            choice = row['sample'].get('weapon', 0); assert type(choice) is int and 0 <= choice < 12
            assert x[833+choice] == 1, 'Unavailable recorded weapon choice'
            if x[1122] == 1 and any(x[835:845]):
                counts['legal_blaster_alternative_rows'] += 1; cases.add(row['seed'])
                if choice >= 2:
                    counts['alternative_weapon_samples'] += 1; switch_cases.add(row['seed'])
    assert counts['rows'] == meta['rows']
    floor = protocol['weapon_exposure']
    checks = dict(opportunity_rows=counts['legal_blaster_alternative_rows'] >= floor['minimum_opportunity_rows'],
                  opportunity_cases=len(cases) >= floor['minimum_opportunity_cases'],
                  explore_choice_coverage=(profile != 'explore' or len(switch_cases) >= floor['minimum_explore_choice_cases']))
    assert all(sha(path) == digest for path, digest in pins.items())
    a.out.mkdir(parents=True)
    save(a.out/'report.json', dict(state='complete', gate_passed=all(checks.values()), profile=profile,
        checks=checks, counts=dict(counts), opportunity_cases=len(cases), alternative_choice_cases=len(switch_cases),
        inputs=pins, training_performed=False, neural_execution=False))
    save(a.out/'complete.json', dict(state='complete', report_sha256=sha(a.out/'report.json'), gate_passed=all(checks.values())))
    print(dict(profile=profile, checks=checks, counts=dict(counts)))


if __name__ == '__main__':
    main()

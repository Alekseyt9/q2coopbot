"""Seal accepted native movement consequences, whole-seed splits and frozen admission gates."""
import argparse
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collection', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    root = a.collection.resolve()
    assert not a.out.exists()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or expected == digest, str(path)
        pins[str(path)] = digest
        return path

    closed = read(pin(root/'complete.json'))
    assert closed['state'] == 'complete' and closed['version'] == 'combat_native_movement_pairs_v1'
    spec = read(pin(root/'spec.json', closed['spec_sha256']))
    protocol = read(pin(spec['protocol'], spec['protocol_sha256']))
    assert protocol['version'] == 'combat_movement_pair_collection_v1'
    assert spec['diagnostic_only'] and not spec['ordinary_training_eligible'] and not spec['training_performed']
    records, exclusions, seed_splits = [], [], {}
    total_admitted = total_excluded = 0
    for item in spec['contexts']:
        context = item['context']
        seed, split = context['seed'], context['split']
        assert seed not in seed_splits and split in ('train', 'development')
        seed_splits[seed] = split
        folder = root/'proofs'/context['context_id']
        seal = read(pin(folder/'complete.json'))
        proof = read(pin(folder/'report.json', seal['report_sha256']))
        assert proof['state'] == 'complete' and proof['context'] == context
        assert proof['source_weights_sha256'] == protocol['source_weights_sha256']
        assert proof['reference_basis'] == 'fresh_ordinary_frozen_world'
        for path, digest in proof['inputs'].items():
            pin(path, digest)
        rows = proof['results']
        assert len(rows) == len(item['variants']) and len({r['label'] for r in rows}) == len(rows)
        assert proof['admitted'] == sum(r['admitted'] for r in rows)
        assert proof['excluded'] == sum(not r['admitted'] for r in rows)
        total_admitted += proof['admitted']
        total_excluded += proof['excluded']
        control = next(r for r in rows if r['label'] == 'control')
        accepted = [r for r in rows if r['admitted']]
        if not control['admitted'] or not any(r['label'] != 'control' for r in accepted):
            exclusions.append(dict(context=context, reason='No accepted baseline and alternative pair', proof=str(folder/'report.json')))
            continue
        variants = {v['label']:v for v in item['variants']}
        labels = []
        for row in accepted:
            useful = bool(row['label'] != 'control' and (
                (row['goal'] and not row['death'] and not control['goal']) or
                (row['goal'] and control['goal'] and not row['death'] and not control['death']
                 and row['kills'] >= control['kills'] and control['received'] - row['received'] >= 5)))
            labels.append(dict(label=row['label'], mean_shift=variants[row['label']]['mean_shift'],
                goal=row['goal'], death=row['death'], kills=row['kills'], dealt=row['dealt'], received=row['received'],
                observed_provider_return_no_bootstrap=row['observed_provider_return_no_bootstrap'],
                observed_provider_sparse_return_no_bootstrap=row['observed_provider_sparse_return_no_bootstrap'],
                useful=useful, received_delta_vs_control=row['received']-control['received'],
                measured_conditional_continuation=True, expected_q=False))
        records.append(dict(context=context, map=context['instance']['map'], split=split, seed=seed,
            labels=labels, useful=any(r['useful'] for r in labels),
            excluded_variants=[r['label'] for r in rows if not r['admitted']],
            proof=str(folder/'report.json'), proof_sha256=sha(folder/'report.json')))
    train = [r for r in records if r['split'] == 'train']
    dev = [r for r in records if r['split'] == 'development']
    useful = [r for r in train if r['useful']]
    assert not {r['seed'] for r in train} & {r['seed'] for r in dev}
    counts = dict(train_contexts=len(train), train_maps=len({r['map'] for r in train}),
        development_contexts=len(dev), development_maps=len({r['map'] for r in dev}),
        train_seeds_with_useful_variant=len({r['seed'] for r in useful}),
        train_maps_with_useful_variant=len({r['map'] for r in useful}))
    gates = {key:counts[key.removeprefix('minimum_')] >= value for key, value in protocol['dataset_admission'].items() if key.startswith('minimum_')}
    for path, digest in pins.items():
        assert sha(path) == digest, path
    a.out.mkdir(parents=True)
    save(a.out/'records.json', dict(version='combat_native_movement_pair_labels_v1', records=records,
        unknown_alternatives_are_negative=False, expected_q=False, ordinary_ppo_eligible=False))
    report = dict(state='complete', version='combat_native_movement_pair_dataset_v1',
        gate_passed=all(gates.values()), gates=gates, counts=counts, admitted_variants=total_admitted,
        excluded_variants=total_excluded, excluded_contexts=exclusions, inputs=pins,
        records_sha256=sha(a.out/'records.json'), source_weights_sha256=protocol['source_weights_sha256'],
        training_performed=False, ordinary_ppo_eligible=False, promotion=False, neural_execution=False,
        scope='Accepted observed conditional movement continuations, not expected Q or independent policy quality. Whole-seed internal development never enters fitting.')
    save(a.out/'report.json', report)
    save(a.out/'complete.json', dict(state='complete', report_sha256=sha(a.out/'report.json'), records_sha256=sha(a.out/'records.json')))
    print(dict(gate_passed=report['gate_passed'], counts=counts, admitted=total_admitted, excluded=total_excluded))


if __name__ == '__main__':
    main()

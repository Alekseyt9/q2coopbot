"""Select four archived TRAIN worlds already covered by the full CUDA unit proof."""
import argparse
import json
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    protocol = read(pin(args.protocol))
    assert protocol['version'] == 'combat_target_sampling_zero_compatibility_v1'
    models = Path(protocol['models'])
    seal = read(pin(models / 'complete.json'))
    proof = read(pin(models / 'report.json', seal['report_sha256']))
    assert proof['device'] == 'cuda' and proof['cases'] == 32
    assert proof['source_parameters_exact'] and proof['zero_source_raw_exact'] and proof['zero_source_full_density_exact']
    assert proof['source_weights_sha256'] == protocol['source_weights_sha256']
    corpus = Path(protocol['corpus']).resolve()
    closed = read(pin(corpus / 'complete.json'))
    collection = read(pin(corpus / 'protocol.json', closed['protocol_sha256']))
    assignment = collection['assignments'][0]
    assert collection['train_only'] and assignment['split'] == 'train'
    plan = read(pin(assignment['plan'], assignment['plan_sha256']))
    pool = read(pin(corpus / 'pool/report.json', closed['pool_sha256']))
    registry = pin(protocol['registry'])
    selected = {}
    for job in pool['jobs']:
        task = plan['tasks'][job['task_index']]
        slot = task['seeds'].index(job['seed'])
        instance = task['instances'][slot]
        assert instance['split'] == 'train' and job['mode'] == 'learned'
        mp = instance['map']
        if mp not in selected or job['seed'] < selected[mp][0]['seed']:
            selected[mp] = job, task, slot, instance
    assert set(selected) == {'base3', 'city1', 'jail1', 'ware2'}
    source = Path(proof['source']).resolve()
    pin(source / 'weights.json', protocol['source_weights_sha256'])
    zero = proof['models']['zero']
    model = read(pin(zero['path'], zero['sha256']))
    assert model['target_sampling'] == dict(version='combat_target_temperature_v1', temperature=1.)
    args.out.mkdir(parents=True)
    units = []
    for mp, (job, task, slot, instance) in sorted(selected.items()):
        folder = corpus / 'processing-train' / f"case-{job['job']:04d}"
        meta = read(pin(folder / 'cuda/report.json'))
        assert meta['model_sha256'] == protocol['source_weights_sha256']
        with pin(folder / 'cuda/rollout.jsonl', meta['rollout_sha256']).open(encoding='utf-8-sig') as stream:
            rows = list(map(json.loads, stream))
        assert rows
        frame = min(r['frame'] for r in rows)
        context = dict(context_id=f"target-unit-{job['seed']}-{frame}", seed=job['seed'], frame=frame,
                       job=job['job'], split='train', role='technical_unit_compatibility', episode=task['episode']['id'],
                       instance=instance, offset=assignment['offset'] + slot, directory=str(folder))
        variants = [dict(index=0, label='control', model=str(source / 'weights.json'), model_sha256=protocol['source_weights_sha256']),
                    dict(index=1, label='zero', model=zero['path'], model_sha256=zero['sha256'])]
        target = args.out / context['context_id']
        target.mkdir()
        report = dict(state='complete', version='combat_target_sampling_zero_unit_v1', device='cuda',
                      context=context, variants=variants, source=str(source), source_weights_sha256=protocol['source_weights_sha256'],
                      baseline=str(corpus), registry=str(registry), inputs=dict(pins), neural_parameters_exact=True,
                      zero_source_raw_exact=True, cuda_checked_rows=len(rows), selection_neural_execution=False,
                      actor_integration_proof=str((models / 'report.json').resolve()), actor_integration_proof_sha256=sha(models / 'report.json'),
                      training_performed=False, promotion=False)
        save(target / 'report.json', report)
        save(target / 'complete.json', dict(state='complete', report_sha256=sha(target / 'report.json')))
        units.append(dict(unit_id=context['context_id'], context=context, prepared=str(target.resolve())))
    for path, digest in pins.items():
        assert sha(path) == digest, path
    save(args.out / 'report.json', dict(state='complete', version='combat_target_sampling_units_preparation_v1', device='cuda',
         units=units, protocol_sha256=sha(args.protocol), maps=sorted(selected), native_cases=8, cuda_cases=8,
         inputs=pins, training_performed=False, promotion=False, selection_neural_execution=False))
    save(args.out / 'complete.json', dict(state='complete', report_sha256=sha(args.out / 'report.json')))
    print(dict(contexts=4, maps=sorted(selected), cases=8))


if __name__ == '__main__':
    main()

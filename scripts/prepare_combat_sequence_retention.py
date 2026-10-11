"""Pin retention rows from closed train-only own-policy captures; no NN forward."""
import argparse
import collections
import json
import pathlib
from process_combat_architecture_pool import read, save, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--series', type=pathlib.Path)
    parser.add_argument('--pool', type=pathlib.Path)
    parser.add_argument('--plan-index', type=int, default=0)
    parser.add_argument('--processing', type=pathlib.Path)
    parser.add_argument('--families', help='Comma-separated episode IDs; default is all registered train families')
    parser.add_argument('--out', type=pathlib.Path, required=True)
    args = parser.parse_args()
    assert bool(args.series) != bool(args.pool), 'Choose legacy series or registered pool'
    extra = {}
    if args.pool:
        from combat_registered_retention import VERSION, bindings
        assert args.processing
        pool_path = args.pool.resolve() / 'report.json'
        processing_path = args.processing.resolve() / 'report.json'
        plan_path, plan, seed_family, data, meta = bindings(pool_path, args.plan_index, processing_path)
        retained = sorted(set(seed_family.values()))
        extra = dict(registered_pool=str(pool_path), plan_index=args.plan_index,
                     processing_report=str(processing_path), processing_report_sha256=sha(processing_path))
    else:
        VERSION = 'combat_sequence_retention_v1'
        assert not args.processing and not args.families
        series = args.series.resolve()
        capture = series / 'round-1/capture'
        data = series / 'round-1/processing/series/rollout'
        pool_path = capture / 'pool/report.json'
        plan_path = capture / 'series/plan.json'
        plan, meta = read(plan_path), read(data / 'report.json')
        retained = [t['episode']['id'] for t in plan['tasks'] if len(t['seeds']) == 4]
    pool = read(pool_path)
    assert pool['state'] == 'complete' and pool['source_unchanged'] and not any(j['error'] for j in pool['jobs'])
    assert all(t['split'] == 'train' for t in plan['tasks'])
    assert plan['model_sha256'] == meta['model_sha256'] == sha(plan['model_path'])
    if args.families:
        selected_families = args.families.split(',')
        assert len(set(selected_families)) == len(selected_families) and set(selected_families) <= set(retained)
        retained = selected_families
    assert retained
    seed_family = {s:t['episode']['id'] for t in plan['tasks'] for s in t['seeds']}
    assert len(seed_family) == sum(len(t['seeds']) for t in plan['tasks'])
    assert sha(data / 'rollout.jsonl') == meta['rollout_sha256']
    assert sha(data / 'sequence.jsonl') == meta['sequence_sha256']
    rows = [json.loads(line) for line in (data / 'rollout.jsonl').read_text().splitlines()]
    assert all(r['seed'] in seed_family for r in rows)
    assert len({(r['seed'], r['index']) for r in rows}) == len(rows)
    selected = [r for r in rows if seed_family[r['seed']] in retained]
    count = collections.Counter(r['seed'] for r in selected)
    family_seeds = {f:{r['seed'] for r in selected if seed_family[r['seed']] == f} for f in retained}
    assert all(family_seeds.values())
    assert all(family_seeds[f] == {s for s, family in seed_family.items() if family == f} for f in retained)
    members = [dict(seed=r['seed'], index=r['index'], family=seed_family[r['seed']],
                    weight=1 / len(retained) / len(family_seeds[seed_family[r['seed']]]) / count[r['seed']])
               for r in selected]
    assert not args.out.exists()
    save(args.out, dict(version=VERSION, feature_version=meta['feature_version'],
         model_sha256=meta['model_sha256'], rollout_sha256=meta['rollout_sha256'],
         sequence_sha256=meta['sequence_sha256'], training_plan=str(plan_path),
         training_plan_sha256=sha(plan_path), pool_sha256=sha(pool_path), **extra,
         retained_families=retained, members=members,
         scope='Own frozen behavior on train-only actual histories; all eligible rows in retained families. Equal family/seed mass. No validation states or teacher actions.'))
    print('Pinned', len(members), 'rows from', len(retained), 'train-only families')


if __name__ == '__main__':
    main()

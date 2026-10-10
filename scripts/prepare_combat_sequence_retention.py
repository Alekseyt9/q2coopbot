"""Pin retention rows from closed train-only own-policy captures; no NN forward."""
import argparse
import collections
import json
import pathlib
from process_combat_architecture_pool import read, save, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--series', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    args = parser.parse_args()
    series = args.series.resolve()
    capture = series / 'round-1/capture'
    data = series / 'round-1/processing/series/rollout'
    pool = read(capture / 'pool/report.json')
    assert pool['state'] == 'complete' and pool['source_unchanged'] and not any(j['error'] for j in pool['jobs'])
    plan_path = capture / 'series/plan.json'
    plan, meta = read(plan_path), read(data / 'report.json')
    assert all(t['split'] == 'train' for t in plan['tasks'])
    assert plan['model_sha256'] == meta['model_sha256'] == sha(plan['model_path'])
    retained = [t['episode']['id'] for t in plan['tasks'] if len(t['seeds']) == 4]
    seed_family = {s:t['episode']['id'] for t in plan['tasks'] for s in t['seeds']}
    assert len(seed_family) == sum(len(t['seeds']) for t in plan['tasks'])
    assert sha(data / 'rollout.jsonl') == meta['rollout_sha256']
    assert sha(data / 'sequence.jsonl') == meta['sequence_sha256']
    rows = [json.loads(line) for line in (data / 'rollout.jsonl').read_text().splitlines()]
    selected = [r for r in rows if seed_family[r['seed']] in retained]
    count = collections.Counter(r['seed'] for r in selected)
    family_seeds = {f:{r['seed'] for r in selected if seed_family[r['seed']] == f} for f in retained}
    assert all(family_seeds.values())
    members = [dict(seed=r['seed'], index=r['index'], family=seed_family[r['seed']],
                    weight=1 / len(retained) / len(family_seeds[seed_family[r['seed']]]) / count[r['seed']])
               for r in selected]
    assert not args.out.exists()
    save(args.out, dict(version='combat_sequence_retention_v1', feature_version=meta['feature_version'],
         model_sha256=meta['model_sha256'], rollout_sha256=meta['rollout_sha256'],
         sequence_sha256=meta['sequence_sha256'], training_plan=str(plan_path),
         training_plan_sha256=sha(plan_path), pool_sha256=sha(capture / 'pool/report.json'),
         retained_families=retained, members=members,
         scope='Own frozen behavior on train-only actual histories; all eligible rows in retained families. Equal family/seed mass. No validation states or teacher actions.'))
    print('Pinned', len(members), 'rows from', len(retained), 'train-only families')


if __name__ == '__main__':
    main()

"""Compare native one-frame immediate labels with subsequent observed returns."""
import argparse
import json
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha

SPARSE = ('monster_damage', 'received_damage', 'self_damage_extra',
          'friendly_damage', 'death', 'monster_kill', 'tick')
EPS = 1e-6


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('probe', 'proof', 'protocol', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args()
    assert not a.out.exists()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    seal = read(pin(a.probe / 'complete.json'))
    spec = read(pin(a.probe / 'spec.json', seal['spec_sha256']))
    prepared = read(pin(Path(spec['prepared']) / 'report.json', spec['prepared_report_sha256']))
    assert prepared['version'] in ('combat_one_frame_aim_batch_preparation_v1','combat_one_frame_aim_batch_preparation_v2_policy')
    proof_seal = read(pin(a.proof / 'complete.json'))
    proof = read(pin(a.proof / 'report.json', proof_seal['report_sha256']))
    assert proof['one_frame_probe'] and proof['total_variants'] == len(spec['variants'])
    assert proof['inputs'][str((a.probe / 'complete.json').resolve())] == seal_hash(a.probe)
    pin(a.protocol)
    contexts = {c['context_id']: c for c in prepared['contexts']}
    variants = {(v['context_id'], v['category']): i for i, v in enumerate(spec['variants'])}
    data = []
    excluded = []

    def trajectory(index, frame):
        folder = a.probe / f'case-{index:02d}'
        cuda = read(pin(folder / 'cuda/report.json'))
        path = pin(folder / 'cuda/rollout.jsonl', cuda['rollout_sha256'])
        with path.open(encoding='utf-8-sig') as stream:
            row = next((r for r in map(json.loads, stream) if r['frame'] == frame), None)
        assert row is not None
        start = row['index']
        native = folder / 'native/replay-0'
        reward_config = read(pin(native / 'reward-config.json'))
        with pin(native / 'rewards.jsonl').open(encoding='utf-8-sig') as stream:
            rewards = list(map(json.loads, stream))
        ids = [r['step'] for r in rewards]
        assert ids == sorted(set(ids)) and start in ids
        assert all(r['version'] == reward_config['version'] for r in rewards)
        unavailable = [r for r in rewards if not r['available']]
        rewards = [r for r in rewards if r['available']]
        assert rewards and any(r['step'] == start for r in rewards)
        # Existing captures contain incomplete post-terminal transitions. Never
        # impute their reward or silently bridge an unavailable internal gap.
        assert all(r['step'] > rewards[-1]['step'] and r['reason'] == 'incomplete_transition'
                   for r in unavailable), 'Unavailable internal reward transition'
        assert all(abs(sum(r['components'].values()) - r['score']) < EPS for r in rewards)
        tail = [r for r in rewards if r['step'] >= start]
        return dict(start_index=start, config=reward_config,
                    prefix=[(r['step'], r['score'], r['components']) for r in rewards if r['step'] < start],
                    full_return=sum(r['score'] for r in tail),
                    sparse_return=sum(sum(r['components'][k] for k in SPARSE) for r in tail),
                    reward_steps=len(tail), last_reward_step=tail[-1]['step'],
                    unavailable_tail_steps=len(unavailable))

    for context in proof['contexts']:
        cid = context['context_id']
        accepted = [s for s in context['samples'] if s['accepted']]
        if not any(s['category'] == 0 for s in accepted):
            excluded.append(dict(context_id=cid, reason='baseline prefix rejected'))
            continue
        base = trajectory(variants[cid, 0], context['frame'])
        samples = []
        for s in accepted:
            cur = base if s['category'] == 0 else trajectory(variants[cid, s['category']], context['frame'])
            assert cur['config'] == base['config']
            # Rewards before the intervention must also have the same causal history.
            before, reference = cur['prefix'], base['prefix']
            same = len(before) == len(reference) and all(
                x[0] == y[0] and abs(x[1] - y[1]) < EPS and x[2].keys() == y[2].keys()
                and all(abs(x[2][k] - y[2][k]) < EPS for k in x[2])
                for x, y in zip(before, reference))
            if not same or cur['start_index'] != base['start_index']:
                excluded.append(dict(context_id=cid, category=s['category'], reason='reward prefix/index differs'))
                continue
            samples.append(dict(category=s['category'], immediate_hp=s['health_damage'],
                                full_return=cur['full_return'], sparse_return=cur['sparse_return'],
                                full_delta=cur['full_return'] - base['full_return'],
                                sparse_delta=cur['sparse_return'] - base['sparse_return'],
                                reward_steps=cur['reward_steps'], last_reward_step=cur['last_reward_step'],
                                unavailable_tail_steps=cur['unavailable_tail_steps']))
        baseline_hp = next(s['immediate_hp'] for s in samples if s['category'] == 0)
        best = {key: [s['category'] for s in samples if s[key] >= max(v[key] for v in samples) - EPS]
                for key in ('immediate_hp', 'full_return', 'sparse_return')}
        data.append(dict(context_id=cid, seed=context['seed'], frame=context['frame'],
                         split=contexts[cid]['split'], baseline_hp=baseline_hp,
                         samples=samples, best=best))
    statistics = {}
    for split in ('train', 'development'):
        group = [c for c in data if c['split'] == split]
        statistics[split] = dict(
            contexts=len(group), seeds=sorted({c['seed'] for c in group}),
            immediate_improvable=sum(any(s['immediate_hp'] > c['baseline_hp'] for s in c['samples']) for c in group),
            full_improvable=sum(any(s['full_delta'] > EPS for s in c['samples']) for c in group),
            sparse_improvable=sum(any(s['sparse_delta'] > EPS for s in c['samples']) for c in group),
            best_sets_differ=sum(any(c['best'][k] != c['best']['immediate_hp'] for k in ('full_return', 'sparse_return')) for c in group),
            no_common_best=sum(not set(c['best']['immediate_hp']).intersection(c['best']['full_return'], c['best']['sparse_return']) for c in group),
            immediate_better_full_worse=sum(any(s['immediate_hp'] > c['baseline_hp'] and s['full_delta'] < -EPS for s in c['samples']) for c in group),
            immediate_better_sparse_worse=sum(any(s['immediate_hp'] > c['baseline_hp'] and s['sparse_delta'] < -EPS for s in c['samples']) for c in group))
    for path, digest in pins.items():
        assert sha(path) == digest, path
    a.out.mkdir(parents=True)
    save(a.out / 'report.json', dict(state='complete', version='combat_one_frame_observed_returns_v1',
         contexts=data, statistics=statistics, excluded=excluded, inputs=pins, script_sha256=sha(__file__),
         training_performed=False, promotion=False, new_games=0,
         scope='Single conditional finite first-life captures; not expected Q, independent validation, or PPO data.'))
    save(a.out / 'complete.json', dict(state='complete', report_sha256=sha(a.out / 'report.json')))
    print(json.dumps(dict(statistics=statistics, excluded=excluded), indent=2))


def seal_hash(root):
    return sha(root / 'complete.json')


if __name__ == '__main__':
    main()

"""Read-only TRAIN reward/GAE audit. No neural inference or training."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def lines(path):
    with path.open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), 'Preserve existing audit outputs'
    source = Path(__file__).with_name('ppo_combat.py')
    tree = ast.parse(source.read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'advantages')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    advantages = namespace['advantages']
    cases = []
    all_values = Counter()
    components = Counter()
    max_reward_error = 0.0
    for job in read(args.corpus / 'pool/report.json')['jobs']:
        if job['plan_index'] != 0:
            continue
        native = args.corpus / 'processing-train' / ('case-%04d' % job['job']) / 'native'
        rows = lines(native / 'native-rollout.jsonl')
        rewards = {row['step']: row for row in lines(native / 'replay-0/rewards.jsonl')}
        assert rows and all(row['seed'] == job['seed'] for row in rows)
        total = Counter()
        kill_steps = []
        for row in rows:
            reward = rewards[row['index']]
            error = abs(reward['score'] - row['reward'])
            max_reward_error = max(max_reward_error, error)
            assert error < 1e-9, (job['job'], row['index'], error)
            assert abs(sum(reward['components'].values()) - row['reward']) < 1e-8
            total.update(reward['components'])
            all_values[row['sample']['value']] += 1
            if reward['components']['monster_kill'] > 0:
                kill_steps.append(row['index'])
        components.update(total)
        gae, _ = advantages(rows, .99, .95)
        mc, _ = advantages(rows, .99, 1.0)
        cases.append(dict(job=job['job'], seed=job['seed'], rows=len(rows),
                          reward=sum(row['reward'] for row in rows), components=dict(total),
                          kills=total['monster_kill'] / 5, deaths=-total['death'] / 5,
                          first_kill_step=kill_steps[0] if kill_steps else None,
                          first_advantage_gae=gae[0], first_advantage_lambda1=mc[0],
                          early20_mean_gae=statistics.mean(gae[:20]),
                          early20_mean_lambda1=statistics.mean(mc[:20]),
                          terminal=rows[-1]['terminal'], truncated=rows[-1]['truncated'],
                          bootstrap_zero=rows[-1].get('bootstrap_zero')))
    assert len(cases) == 32 and len({case['seed'] for case in cases}) == 32
    merged = lines(args.corpus / 'processing-train/merged/rollout.jsonl')
    assert len(merged) == sum(case['rows'] for case in cases)
    assert Counter(row['sample']['value'] for row in merged) == all_values
    groups = {}
    for name, predicate in [('with_kills', lambda c: c['kills'] > 0),
                            ('no_kills', lambda c: c['kills'] == 0),
                            ('death', lambda c: c['deaths'] > 0)]:
        selected = [case for case in cases if predicate(case)]
        groups[name] = dict(cases=len(selected), mean_reward=statistics.mean(c['reward'] for c in selected) if selected else None,
                            mean_early20_gae=statistics.mean(c['early20_mean_gae'] for c in selected) if selected else None,
                            mean_early20_lambda1=statistics.mean(c['early20_mean_lambda1'] for c in selected) if selected else None)
    result = dict(version='combat_reward_credit_audit_v1', split='TRAIN', cases=cases,
                  rows=len(merged), value_counts=dict(all_values), reward_components=dict(components),
                  max_native_reward_error=max_reward_error, groups=groups,
                  gae=dict(gamma=.99, lam=.95, effective_horizon_steps=1 / (1 - .99 * .95),
                           half_life_steps=math.log(.5) / math.log(.99 * .95),
                           future_reward_weights={str(n): (.99 * .95) ** n for n in [10, 20, 50, 100]}),
                  ppo_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  merged_sha256=hashlib.sha256((args.corpus / 'processing-train/merged/rollout.jsonl').read_bytes()).hexdigest(),
                  neural_computation=False, training=False,
                  limitation='Descriptive TRAIN audit, not proof that a different baseline or lambda improves combat.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ['rows', 'value_counts', 'reward_components', 'groups', 'max_native_reward_error', 'gae']}, indent=2))


if __name__ == '__main__':
    main()

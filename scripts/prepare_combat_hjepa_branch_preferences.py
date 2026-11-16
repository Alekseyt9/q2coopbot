"""Build initial-state preferences from verified diagnostic weapon-policy outcomes."""
import argparse
import collections
import hashlib
import json
import math
import pathlib
from audit_combat_hjepa_weapons import NAMES, canonical


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outcome_score(result):
    life = result['first_life']
    damage = life['damage']
    hp = sum(e['health'] for e in result['generated_start']['entities'] if e['class'].startswith('monster_'))
    dealt = sum(e['health_damage'] for e in life['outgoing_by_target_class'])
    progress = min(1., dealt / max(1, hp))
    death = bool(result['death_stop'])
    win = bool(result['goal_stop']) and not death
    return 100 * win - 100 * death + 10 * progress - .1 * damage['received_health_damage'] - .2 * damage['self_health_damage']


def preference(scores, mask):
    # Fixed change cost and temperature are documented before fitting and are
    # never selected using validation or held-out combat outcomes.
    values = {i: (v - (2 if i else 0)) / 15 for i, v in scores.items() if mask[i]}
    if set(values) != {i for i, yes in enumerate(mask) if yes}:
        raise ValueError('not all available options measured')
    top = max(values.values())
    weights = {i: math.exp(v - top) for i, v in values.items()}
    total = sum(weights.values())
    return [weights.get(i, 0.) / total for i in range(12)]


def outcome_improvement(scores, outcomes):
    baseline = outcomes[0]
    candidates = [i for i, o in outcomes.items() if i and
                  (o['win'] and not baseline['win'] or baseline['death'] and not o['death'])]
    best = max(candidates, key=lambda i: scores[i]) if candidates else 0
    return [float(i == best) for i in range(12)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--matrix', type=pathlib.Path, required=True)
    ap.add_argument('--states', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    ap.add_argument('--outcome-improvement', action='store_true', help='hard labels only for actual win gain/death avoidance over keep')
    a = ap.parse_args()
    if a.out.exists() or pathlib.Path(str(a.out) + '.meta.json').exists():
        raise ValueError('fresh preference output required')
    complete = json.loads((a.matrix / 'complete.json').read_text(encoding='utf-8-sig'))
    if complete['state'] != 'complete' or not complete.get('diagnostic_weapon_branches'):
        raise ValueError('completed diagnostic branches required')
    if sha(a.matrix / 'plan.json') != complete['source_plan_sha256']:
        raise ValueError('changed source plan')
    meta = json.loads(pathlib.Path(str(a.states) + '.meta.json').read_text())
    for field in ['sources_sha256', 'native_receipts_sha256']:
        for name, digest in meta[field].items():
            if sha(pathlib.Path(name)) != digest:
                raise ValueError('changed source/receipt')
    if sha(a.matrix / 'complete.json') != meta['complete_sha256']:
        raise ValueError('changed completion receipt')
    variants = {v['name']: v for v in complete['models']}
    parent_variant = next(v for v in variants.values() if v['diagnostic_weapon_slot'] == 0)
    parent = json.loads(pathlib.Path(parent_variant['model']).read_text())
    for v in variants.values():
        if sha(pathlib.Path(v['model'])) != v['sha256']:
            raise ValueError('changed branch weights')
        model = json.loads(pathlib.Path(v['model']).read_text())
        frozen = lambda d: {k: value for k, value in d.items() if k not in ['weapon_actor', 'weapon_head', 'weapon_mode', 'weapon_sampling_seed']}
        if frozen(model) != frozen(parent):
            raise ValueError('non-weapon branch change')
        slot = v['diagnostic_weapon_slot']
        if slot:
            layers = model['weapon_actor']
            if len(layers) != 1 or any(any(w != 0 for w in row) for row in layers[0]['weight']):
                raise ValueError('branch is not a constant diagnostic head')
            if layers[0]['bias'] != [50. if j == slot else 0. for j in range(12)] or model.get('weapon_sampling_seed') is not None:
                raise ValueError('invalid diagnostic head')
    states = [json.loads(line) for line in a.states.read_text().splitlines()]
    if len(states) != len(complete['jobs']) or len(states) != meta['rows']:
        raise ValueError('incomplete branch state coverage')
    grouped = collections.defaultdict(dict)
    receipts = dict(meta['native_receipts_sha256'])
    splits = set()
    for state in states:
        v = variants[state['variant']]
        slot = v['diagnostic_weapon_slot']
        seed = state['seed']
        if seed not in v['seeds'] or slot in grouped[seed]:
            raise ValueError('duplicate or unexpected branch')
        report_path = pathlib.Path(state['case_report'])
        report = json.loads(report_path.read_text())
        receipts[str(report_path)] = sha(report_path)
        if not report['capture_complete'] or not report['provenance_valid'] or len(report['results']) != 1:
            raise ValueError('invalid native capture')
        result = report['results'][0]
        if result['seed'] != seed or not result['capture_valid'] or not result['generated_start']['confirmed']:
            raise ValueError('wrong seed or initial state')
        if result['generated_fixture']['split'] != state['split'] or state['split'] not in ['train', 'validation']:
            raise ValueError('preference data cannot use test/confirmation')
        splits.add(state['split'])
        grouped[seed][slot] = (state, result)
    if len(splits) != 1:
        raise ValueError('mixed splits')
    rows, rejected = [], []
    keep_wins = best_wins = 0
    for seed, arms in sorted(grouped.items()):
        base, baseline = arms[0]
        if any(s['identity'] != base['identity'] or s['features'] != base['features'] or
               s.get('weapon_features') != base.get('weapon_features') for s, _ in arms.values()):
            rejected.append(dict(seed=seed, reason='initial_observation_mismatch'))
            continue
        slot = NAMES.index(canonical(base['weapon']))
        measured = set(arms) | {slot}
        available = {j for j, yes in enumerate(base['availability']) if yes}
        if measured != available:
            rejected.append(dict(seed=seed, reason='initial_availability_mismatch', measured=sorted(measured), available=sorted(available)))
            continue
        scores = {j: outcome_score(result) for j, (_, result) in arms.items()}
        scores[slot] = scores[0]  # requesting the already equipped initial weapon is a no-op class
        outcomes = {j: dict(win=bool(result['goal_stop']) and not bool(result['death_stop']), death=bool(result['death_stop']))
                    for j, (_, result) in arms.items()}
        outcomes[slot] = dict(outcomes[0])
        y = outcome_improvement(scores, outcomes) if a.outcome_improvement else preference(scores, base['availability'])
        keep_wins += bool(baseline['goal_stop']) and not bool(baseline['death_stop'])
        best_wins += any(bool(r['goal_stop']) and not bool(r['death_stop']) for _, r in arms.values())
        rows.append(dict(seed=seed, split=base['split'], identity=base['identity'], features=base['features'],
                         weapon_features=base.get('weapon_features'),
                         availability=base['availability'], preference_probabilities=y, branch_scores=scores,
                         branch_outcomes=outcomes,
                         label_kind='native_weapon_policy_preference_v1'))
    if not rows:
        raise ValueError('no common initial states')
    a.out.write_text(''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in rows))
    result = dict(version='hjepa_native_weapon_preference_data_v1', split=next(iter(splits)), rows=len(rows),
                  sources_sha256=meta['sources_sha256'], native_receipts_sha256=receipts,
                  branch_states=str(a.states.resolve()), branch_states_sha256=sha(a.states),
                  completion=str((a.matrix / 'complete.json').resolve()), completion_sha256=sha(a.matrix / 'complete.json'),
                  parent_source=parent_variant['model'], parent_sha256=parent_variant['sha256'],
                  rejected=rejected, keep_wins=keep_wins, best_branch_wins=best_wins,
                  score='100 win -100 death +10 capped progress -0.1 receivedHP -0.2 selfHP',
                  temperature=15, change_cost=2, label_kind='native_weapon_policy_preference_v1', diagnostic=True,
                  target_mode='outcome_improvement' if a.outcome_improvement else 'soft_score',
                  scope='Actual diagnostic policy contrasts at common initial client states. Not per-frame optimal labels, '
                        'not PPO, not an autonomous oracle; outcomes never enter features.')
    pathlib.Path(str(a.out) + '.meta.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(rows=len(rows), rejected=rejected, keep_wins=keep_wins, best_branch_wins=best_wins)))


if __name__ == '__main__':
    main()

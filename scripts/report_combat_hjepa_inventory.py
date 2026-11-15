"""Join verified paired native outcomes to generated encounter and resource strata."""
import argparse
import collections
import json
import pathlib

from report_combat_hjepa_matrix import summarize


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--matrix', type=pathlib.Path, required=True)
    ap.add_argument('--difficulty', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    a = ap.parse_args()
    complete = json.loads((a.matrix / 'complete.json').read_text(encoding='utf-8-sig'))
    assert complete['state'] == 'complete'
    plan = json.loads((a.matrix / 'plan.json').read_text(encoding='utf-8-sig'))
    conditions = {i['engine_seed']: i for t in plan['tasks'] for i in t['instances']}
    difficulty = {i['seed']: i for i in json.loads(a.difficulty.read_text())['cases']}
    variants = {}
    for variant in complete['models']:
        name = variant['name']
        paths = sorted((a.matrix / name).glob('case-*/report.json'))
        result = summarize(paths)
        assert set(conditions) == {e['seed'] for e in result['episodes']}
        ownership = collections.Counter()
        for path in paths:
            for trace in path.parent.glob('worker-*/bot.jsonl'):
                for line in trace.open(encoding='utf-8'):
                    b = json.loads(line).get('combat_policy', {})
                    if b.get('observation', {}).get('identity', {}).get('frame', 0) >= 100:
                        s = b['selection']
                        ownership[s['owner'] + ':' + s.get('fallback', '')] += 1
        result['post_release_captured_ownership'] = dict(ownership)
        details = {}
        for path in paths:
            for r in json.loads(path.read_text())['results']:
                seed = r['seed']
                assert r['generated_fixture'] == conditions[seed], 'fixture differs from frozen plan'
                assert r['generated_start']['confirmed'], 'native start not verified'
                life = r['first_life']
                details[seed] = dict(self_hp=life['damage']['self_health_damage'],
                                     teammate_hp=life['damage']['teammate_health_damage'],
                                     monster_damage=sum(x['health_damage'] for x in life['outgoing_by_target_class']),
                                     kills=sum(x['kills'] for x in life['outgoing_by_target_class']),
                                     weapon_requests=life['weapon_requests'])
        strata = collections.defaultdict(lambda: collections.defaultdict(list))
        for e in result['episodes']:
            i = conditions[e['seed']]
            e.update(details[e['seed']])
            e.update(initial_weapon=i['inventory']['initial_weapon'],
                     inventory=i['inventory'], monsters=i['monsters'],
                     difficulty=difficulty[e['seed']]['tier'], skill=i['skill'])
            for dimension, value in [('map', e['map']), ('initial_weapon', e['initial_weapon']),
                                     ('difficulty', e['difficulty']), ('monster_count', str(len(i['monsters'])))]:
                strata[dimension][value].append(e)
        def metrics(rows):
            return dict(episodes=len(rows), **{k: sum(r[k] for r in rows) for k in
                        ['win', 'death', 'timeout', 'received_hp', 'self_hp', 'teammate_hp',
                         'monster_damage', 'kills', 'guard_interventions']})
        result['strata'] = {dim: {key: metrics(rows) for key, rows in groups.items()}
                            for dim, groups in strata.items()}
        result['totals'] = metrics(result['episodes'])
        variants[name] = result
    report = dict(version='hjepa_inventory_native_screen_v1', promotion=False,
                  scope='Paired system comparison at skill 1, 300-frame budget; H-JEPA keeps equipped weapon '
                        'and engine may switch on empty ammo. U1 and rules choose weapons. '
                        'Resource tiers are pre-outcome estimates, not solvability certificates. '
                        'Frozen diagnostic conditions on previously seen maps; partial earlier runs exposed '
                        'runtime admission issues. Not a fresh final test or held-out map generalization.',
                  variants=variants)
    a.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v['totals'] for k, v in variants.items()}, indent=2))


if __name__ == '__main__':
    main()

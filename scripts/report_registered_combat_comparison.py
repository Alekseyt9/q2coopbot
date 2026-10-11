"""Report sealed matched evaluation outcomes and behavior; no neural computation."""
import argparse
import collections
import hashlib
import json
import pathlib
import random


def read(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with pathlib.Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def paired_summary(pairs):
    assert pairs, 'Empty comparison'
    keys = [(p['episode'], p['seed']) for p in pairs]
    assert len(set(keys)) == len(keys), 'Duplicate matched case'
    clusters = collections.defaultdict(list)
    changes = collections.Counter()
    totals = {side: collections.Counter() for side in ('before', 'after')}
    for pair in pairs:
        a, b = (pair['metrics'][side] for side in ('before', 'after'))
        assert isinstance(a['goal'], bool) and isinstance(b['goal'], bool)
        clusters[pair['episode']].append(int(b['goal']) - int(a['goal']))
        changes['gained' if b['goal'] and not a['goal'] else
                'lost' if a['goal'] and not b['goal'] else
                'both_won' if a['goal'] else 'both_lost'] += 1
        for side in totals:
            totals[side]['battles'] += 1
            totals[side].update(pair['metrics'][side])
    assert len({len(v) for v in clusters.values()}) == 1, 'Unbalanced family blocks'
    means = [sum(v) / len(v) for _, v in sorted(clusters.items())]
    rng = random.Random(20261011)
    samples = sorted(sum(rng.choice(means) for _ in means) / len(means)
                     for _ in range(5000))
    return dict(cases=len(pairs), families=len(clusters), totals=totals,
                paired_goal_changes=dict(changes),
                win_rate_delta_pp=100 * sum(means) / len(means),
                exploratory_family_bootstrap_95_interval_pp=[
                    100 * samples[int(.025 * 4999)], 100 * samples[int(.975 * 4999)]],
                bootstrap_seed=20261011, bootstrap_replicates=5000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evaluation', type=pathlib.Path, required=True)
    ap.add_argument('--spec', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    args = ap.parse_args()
    root, out = args.evaluation.resolve(), args.out.resolve()
    assert not out.exists(), 'Fresh report directory required'
    complete = read(root / 'complete.json')
    assert complete['state'] == 'complete' and not complete['training_performed']
    assert not complete['promotion']
    assert sha(args.spec) == complete['spec_sha256']
    spec = read(args.spec)
    assert spec['version'] == 'registered_combat_model_comparison_v1'
    digest = sha(root / 'pool/report.json')
    assert digest == complete['pool_sha256']
    pool = read(root / 'pool/report.json')
    assert pool['state'] == 'complete' and pool['source_unchanged']
    assert pool['usable_captures'] == len(pool['jobs']) == complete['native_cases']
    assert all(not job['error'] for job in pool['jobs'])
    labels = [v['label'] for v in spec['variants']]
    assert len(labels) == len(set(labels))
    pins = {root / 'complete.json': sha(root / 'complete.json'),
            args.spec: sha(args.spec), root / 'pool/report.json': digest}
    efficiency = read(root / 'action-efficiency.json')
    assert efficiency['state'] == 'complete' and efficiency['pool_sha256'] == digest
    assert efficiency['behavior_clock_version'] == 'registered_target_intent_audit_v4'
    pins[root / 'action-efficiency.json'] = sha(root / 'action-efficiency.json')
    assert sha(root / 'damage-timing.json') == efficiency['damage_sha256']
    pins[root / 'damage-timing.json'] = efficiency['damage_sha256']
    for label, item in efficiency['variants'].items():
        assert label in labels
        path = root / ('behavior-' + label + '.json')
        assert sha(path) == item['audit_sha256']
        pins[path] = item['audit_sha256']
    comparisons, variants, fixture_keys = {}, {}, None
    assert [{k: item[k] for k in ('name', 'before', 'after')}
            for item in complete['comparisons']] == spec['comparisons']
    for binding in complete['comparisons']:
        path = root / ('quality-' + binding['name'] + '.json')
        assert sha(path) == binding['sha256']
        pins[path] = binding['sha256']
        value = read(path)
        assert value['state'] == 'complete'
        assert value['before_pool_sha256'] == value['after_pool_sha256'] == digest
        for side in ('before', 'after'):
            assert value[side + '_plan_index'] == labels.index(binding[side])
        pairs = value['pairs']
        keys = {(p['episode'], p['seed'], p['map']) for p in pairs}
        if fixture_keys is None: fixture_keys = keys
        assert keys == fixture_keys, 'Comparisons use different fixtures'
        grouped = {'all': paired_summary(pairs)}
        assert grouped['all']['totals'] == value['totals']
        for group in ('ordinary', 'recovery'):
            rows = [p for p in pairs if p['episode'].endswith('-recovery') == (group == 'recovery')]
            if rows: grouped[group] = paired_summary(rows)
        comparisons[binding['name']] = dict(before=binding['before'], after=binding['after'], groups=grouped)
        for side in ('before', 'after'):
            label = binding[side]
            totals = {group: result['totals'][side] for group, result in grouped.items()}
            if label in variants: assert variants[label]['outcomes'] == totals
            else: variants[label] = dict(outcomes=totals, behavior=efficiency['variants'].get(label))
    assert set(variants) == set(labels), 'An evaluated variant lacks a comparison'
    assert len(fixture_keys) == efficiency['cases_per_variant']
    assert len(labels) * len(fixture_keys) == complete['native_cases']
    assert complete['cuda_cases'] == sum(item['cases'] for item in complete['processed'])
    for label, item in variants.items():
        if item['behavior']:
            b = item['behavior']['totals']
            assert b['cases'] == item['outcomes']['all']['battles']
    for path, expected in pins.items(): assert sha(path) == expected
    out.mkdir(parents=True)
    scope = ('Descriptive matched held-out comparisons. Family bootstrap is exploratory: '
             'families share maps and each reward branch has one training seed. '
             'No multiple-comparison correction, promotion, campaign or coop acceptance. '
             'Attack commands are not physical shots; turns and standing are not automatically waste.')
    result = dict(state='complete', version='registered_combat_comparison_report_v1',
                  native_cases=complete['native_cases'], cuda_cases=complete['cuda_cases'],
                  variants=variants, comparisons=comparisons,
                  inputs=[dict(path=str(p), sha256=v) for p, v in pins.items()],
                  script_sha256=sha(pathlib.Path(__file__)), training_performed=False, promotion=False, scope=scope)
    (out / 'complete.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# Парное сравнение моделей', '',
             '| Вариант | Победы / бои | Ordinary | Recovery | Урон получен | Смены цели /100 кадров | Повороты ≥30° | Стояние при clear enemy | Attack без цели |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label in labels:
        v = variants[label]; t = v['outcomes']['all']
        split = lambda name: str(v['outcomes'][name]['goal']) + '/' + str(v['outcomes'][name]['battles']) if name in v['outcomes'] else '—'
        behavior = v['behavior']['totals'] if v['behavior'] else {}
        switches = behavior.get('target_changes_per_100_frames')
        switches = f'{switches:.2f}' if switches is not None else '—'
        fraction = lambda key: f"{100*behavior[key]:.2f}%" if behavior.get(key) is not None else '—'
        lines.append(f"| {label} | {t['goal']}/{t['battles']} | {split('ordinary')} | {split('recovery')} | {t['received']} | {switches} | {fraction('large_yaw_command_fraction')} | {fraction('stationary_clear_enemy_fraction')} | {fraction('attack_without_target_fraction')} |")
    lines += ['', '| Сравнение | Группа | Новые победы | Потерянные победы | Разница, п.п. | Описательный интервал 95%, п.п. |',
              '|---|---|---:|---:|---:|---:|']
    for name, comparison in comparisons.items():
        for group, stats in comparison['groups'].items():
            changes = stats['paired_goal_changes']; lo, hi = stats['exploratory_family_bootstrap_95_interval_pp']
            lines.append(f"| {name} | {group} | {changes.get('gained',0)} | {changes.get('lost',0)} | {stats['win_rate_delta_pp']:.2f} | [{lo:.2f}; {hi:.2f}] |")
    lines += ['', scope, '']
    (out / 'results.md').write_text('\n'.join(lines), encoding='utf-8')
    print(out / 'results.md')


if __name__ == '__main__':
    main()

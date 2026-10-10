"""Paired miss-objective evaluation with repeated-parent and firing diagnostics."""
import argparse
import collections
import json
import pathlib
import statistics

from process_combat_architecture_pool import read, save, sha
from run_combat_spatial_ppo import wait_process


def paired(left, right):
    """Positive deltas mean right minus left; keys must describe the same fights."""
    assert left.keys() == right.keys() and left
    keys = sorted(left)
    gained = sum(not left[k]['win'] and right[k]['win'] for k in keys)
    lost = sum(left[k]['win'] and not right[k]['win'] for k in keys)
    return dict(episodes=len(keys), gained_wins=gained, lost_wins=lost, win_delta=gained-lost,
        both_won=sum(left[k]['win'] and right[k]['win'] for k in keys),
        both_lost=sum(not left[k]['win'] and not right[k]['win'] for k in keys),
        mean_received_damage_delta=statistics.mean(right[k]['received_damage']-left[k]['received_damage'] for k in keys),
        mean_outgoing_damage_delta=statistics.mean(right[k]['outgoing_damage']-left[k]['outgoing_damage'] for k in keys),
        death_delta=sum(right[k]['death']-left[k]['death'] for k in keys))


def report(root):
    filenames = ('protocol.json', 'quality-report.json', 'quality-episodes.json',
                 'recovery/verified-members.json', 'miss-reward-audit.json', 'first-attack.json')
    hashes = {name: sha(root/name) for name in filenames}
    inputs = {name: read(root/name) for name in filenames}
    protocol, quality, proof = (inputs[name] for name in filenames[:2]+('recovery/verified-members.json',))
    assert protocol['version'] == 'combat_miss_reward_ab_validation_v1'
    assert quality['state'] == proof['state'] == 'complete' and proof['source_unchanged']
    assert quality['protocol_sha256'] == proof['protocol_sha256'] == hashes['protocol.json']
    assert quality['pool_sha256'] == hashes['recovery/verified-members.json']
    assert len(inputs['quality-episodes.json']) == protocol['total_episodes'] == 480
    attack, audit = inputs['first-attack.json'], inputs['miss-reward-audit.json']
    assert attack['state'] == 'complete'
    assert attack['protocol_sha256'] == audit['protocol_sha256'] == hashes['protocol.json']
    assert attack['proof_sha256'] == audit['member_proof_sha256'] == hashes['recovery/verified-members.json']
    plans, recipe_by_family = {}, {}
    for entry in protocol['evaluations']:
        assert sha(entry['plan']) == entry['plan_sha256']
        label = entry['model']+'-'+entry['label']
        plans[label] = read(entry['plan'])
    reference = plans['miss-ab-control-before']
    for task in reference['tasks']:
        recipe_by_family[task['episode']['id']] = task['episode']['recipe']['loadout']
    for plan in plans.values():
        assert len(plan['tasks']) == len(reference['tasks'])
        for left, right in zip(reference['tasks'], plan['tasks']):
            assert left['episode']['id'] == right['episode']['id']
            assert left['seeds'] == right['seeds'] and left['instances'] == right['instances']
            assert left['episode']['recipe'] == right['episode']['recipe'], 'Evaluation recipe mismatch'
    parents = [e for e in protocol['evaluations'] if e['label'] == 'before']
    assert len(parents) == 2 and parents[0]['deterministic_weights_sha256'] == parents[1]['deterministic_weights_sha256']
    groups = collections.defaultdict(dict)
    for row in inputs['quality-episodes.json']:
        key = (row['episode'], row['seed'])
        assert key not in groups[row['variant']]
        groups[row['variant']][key] = row
    assert set(groups) == set(plans)
    quality_variants = {row['variant']: row for row in quality['variants']}
    assert set(quality_variants) == set(groups)
    for name, rows in groups.items():
        assert len(rows) == protocol['episodes_per_model'] == 80
        expected_keys = {(task['episode']['id'], seed) for task in plans[name]['tasks'] for seed in task['seeds']}
        assert rows.keys() == expected_keys
        assert rows.keys() == groups['miss-ab-control-before'].keys()
        assert sum(row['win'] for row in rows.values()) == quality_variants[name]['wins']
        assert sum(row['death'] for row in rows.values()) == quality_variants[name]['deaths']
        for index, task in enumerate(plans[name]['tasks']):
            for seed in task['seeds']:
                member = pathlib.Path(plans[name]['output_root'])/f"case-{index}-{task['modes'][0]}"/f's-{seed}'
                assert rows[(task['episode']['id'], seed)]['report_sha256'] == proof['members'][str(member)]['report_sha256']
        seeds = {key[1] for key in rows}
        assert len(seeds) == len(rows), 'Diagnostics require globally unique seeds'
        assert {e['seed'] for e in attack['groups'][name]['episodes']} == seeds
        assert {e['seed'] for e in audit['groups'][name]['episodes']} == seeds
    comparisons = {}
    pairs = [('parent_repeat', 'miss-ab-control-before', 'miss-ab-miss-before'),
             ('control_update', 'miss-ab-control-before', 'miss-ab-control-after'),
             ('miss_update', 'miss-ab-miss-before', 'miss-ab-miss-after'),
             ('after_miss_vs_control', 'miss-ab-control-after', 'miss-ab-miss-after'),
             ('miss_vs_rules', 'rules-baseline', 'miss-ab-miss-after'),
             ('miss_vs_firebc', 'firebc-baseline', 'miss-ab-miss-after')]
    for label, left, right in pairs:
        by_loadout = {}
        for loadout in sorted(set(recipe_by_family.values())):
            subset = lambda name: {k: v for k, v in groups[name].items() if recipe_by_family[k[0]] == loadout}
            by_loadout[loadout] = paired(subset(left), subset(right))
        comparisons[label] = dict(left=left, right=right, all=paired(groups[left], groups[right]), by_loadout=by_loadout)
    variants = {}
    for name, rows in groups.items():
        counts = audit['groups'][name]['counts']
        launches, misses = counts.get('eligible_launches', 0), counts.get('confirmed_misses', 0)
        variants[name] = dict(wins=sum(r['win'] for r in rows.values()), deaths=sum(r['death'] for r in rows.values()),
            eligible_blaster_launches=launches, confirmed_blaster_misses=misses,
            confirmed_miss_fraction_lower=misses/launches if launches else None,
            unknown_blaster_endings=counts.get('unresolved', 0)+counts.get('freed', 0),
            creditable_misses=counts.get('creditable_misses', 0),
            first_attack_by_initial_weapon=attack['groups'][name]['by_initial_weapon'],
            requested_attack_delay=attack['groups'][name]['delays']['requested_attack'],
            sent_attack_delay=attack['groups'][name]['delays']['sent_attack'])
    delta = comparisons['miss_update']['all']['win_delta']-comparisons['control_update']['all']['win_delta']
    assert hashes == {name: sha(root/name) for name in filenames}, 'Inputs changed during report'
    output = dict(version='combat_miss_objective_paired_report_v1', state='complete', variants=variants,
        comparisons=comparisons, difference_of_update_win_deltas=delta, source_sha256=hashes,
        scope='Paired development validation, common reward/fixture/guard. Repeated identical parents measure native repeat variation. Difference of update win deltas is descriptive, not causal proof or significance. Actual trajectories and eligible training transitions may differ. Confirmed misses are a lower bound with unknown endings explicit; damage/corpse contact is not counted as a geometry miss. Machinegun launch is unavailable: initial-weapon ammo decrement is only a proxy. No automatic promotion or untouched final-test claim.')
    save(root/'miss-ab-comparison.json', output)
    lines = ['# Miss penalty A/B', '', output['scope'], '',
             '| Variant | Wins /80 | Deaths | Blaster launches | Confirmed misses | Unknown endings |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, row in variants.items():
        lines.append(f"| {name} | {row['wins']} | {row['deaths']} | {row['eligible_blaster_launches']} | {row['confirmed_blaster_misses']} | {row['unknown_blaster_endings']} |")
    lines += ['', '| Pair (right minus left) | Win delta | Gained | Lost | Received damage delta |',
              '| --- | ---: | ---: | ---: | ---: |']
    for label, comparison in comparisons.items():
        row = comparison['all']
        lines.append(f"| {label} | {row['win_delta']:+d} | {row['gained_wins']} | {row['lost_wins']} | {row['mean_received_damage_delta']:+.2f} |")
    lines += ['', f'Difference of update win deltas: {delta:+d}; descriptive only.',
              'Per-loadout pairs and initial-weapon firing delays are retained in miss-ab-comparison.json.', '']
    (root/'miss-ab-comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    return dict(state='complete', report=str(root/'miss-ab-comparison.json'), comparisons=comparisons)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.wait_pid:
        wait_process(args.wait_pid, root/'miss-ab-comparison-progress.json', stage='waiting_for_sealed_firing_diagnostics')
    try:
        result = report(root)
        save(root/'miss-ab-comparison-progress.json', result)
        print(json.dumps(result), flush=True)
    except Exception as error:
        save(root/'miss-ab-comparison-progress.json', dict(stage='failed', error=str(error)))
        raise

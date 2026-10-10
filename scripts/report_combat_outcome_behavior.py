"""Join sealed native diagnostics by episode; descriptive associations, no NN replay."""
import argparse
import collections
import pathlib

from process_combat_architecture_pool import read, save, sha
from report_combat_first_attack import distribution
from run_combat_spatial_ppo import wait_process


def summarize(rows):
    counts = collections.Counter()
    for row in rows:
        counts.update(row['counts'])
    fraction = lambda a, b: counts[a] / counts[b] if counts[b] else None
    return dict(episodes=len(rows), counts=dict(counts),
        stationary_fraction=fraction('stationary', 'alive_native_commands'),
        firing_stationary_fraction=fraction('held_attack_stationary', 'held_attack'),
        movement_cancelled_fraction=fraction('provider_movement_stopped', 'provider_requested_movement'),
        mean_episode_stationary_fraction=(sum(r['counts']['stationary'] / r['counts']['alive_native_commands'] for r in rows) / len(rows)) if rows else None,
        first_attack={key: distribution([r['delay_frames'][key] for r in rows])
                      for key in rows[0]['delay_frames']} if rows else {})


def report(root):
    files = ['protocol.json', 'quality-report.json', 'recovery/verified-members.json',
             'diagnostics-acceptance.json', 'first-attack.json', 'first-life-movement.json']
    hashes = {name: sha(root / name) for name in files}
    protocol, quality, proof, seal, attacks, movements = [read(root / name) for name in files]
    assert quality['state'] == proof['state'] == seal['state'] == 'complete'
    assert quality['episodes'] == protocol['total_episodes'] == len(proof['members'])
    for item in (quality, proof, attacks, movements):
        assert item['protocol_sha256'] == hashes['protocol.json']
    assert seal['quality_report_sha256'] == hashes['quality-report.json']
    for name in ('first-attack.json', 'first-life-movement.json'):
        assert seal['reports'][name] == hashes[name]
    assert attacks['groups'].keys() == movements['groups'].keys()
    groups = {}
    for name, movement in movements['groups'].items():
        # Worker is case-N/s-SEED/learned (or rules), retained by first-attack.
        indexed = {}
        for attack in attacks['groups'][name]['episodes']:
            worker = pathlib.Path(attack['worker'])
            key = (worker.parent.parent.name, attack['seed'])
            assert key not in indexed, ('Duplicate episode identity', name, key)
            indexed[key] = attack
        rows = []
        for item in movement['episodes_detail']:
            key = (item['case'], item['seed'])
            attack = indexed.pop(key)
            assert not (item['win'] and item['death']), 'Win/death overlap requires explicit handling'
            outcome = 'win' if item['win'] else 'death' if item['death'] else 'unfinished'
            rows.append(dict(case=item['case'], seed=item['seed'], outcome=outcome,
                initial_weapon=attack['initial_weapon'], counts=item['counts'],
                delay_frames=attack['delay_frames'],
                attack_guard_reasons=attack['attack_guard_reasons_before_first_sent'],
                movement_reasons=item['movement_reasons']))
        assert not indexed and len(rows) == protocol['episodes_per_model']
        groups[name] = dict(all=summarize(rows),
            by_outcome={outcome: summarize([r for r in rows if r['outcome'] == outcome])
                        for outcome in ('win', 'death', 'unfinished')},
            by_initial_weapon={weapon: {outcome: summarize([r for r in rows
                if r['initial_weapon'] == weapon and r['outcome'] == outcome])
                for outcome in ('win', 'death', 'unfinished')}
                for weapon in sorted({r['initial_weapon'] for r in rows})}, episodes=rows)
    assert hashes == {name: sha(root / name) for name in files}, 'Diagnostic inputs changed'
    scope = ('Native first-life diagnostic association by outcome and initial weapon; '
        'not a causal effect of standing or delayed fire. Held attack does not prove discharge. '
        'Pooled fractions weight longer fights more; episode mean is also retained. '
        'Missing first attacks stay unknown, unfinished fights remain separate. No neural execution.')
    save(root / 'outcome-behavior.json', dict(state='complete', version='combat_outcome_behavior_v1',
        protocol_sha256=hashes['protocol.json'], source_sha256=hashes, groups=groups, scope=scope))
    lines = ['# Combat behavior by outcome', '',
        '| Variant | Outcome | Episodes | Stationary | Held attack stationary | Movement cancelled | First sent attack mean, frames | Unknown attack |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    percentage = lambda x: 'n/a' if x is None else f'{100*x:.2f}%'
    for name, group in groups.items():
        for outcome, result in group['by_outcome'].items():
            delay = result['first_attack'].get('sent_attack', {})
            mean = delay.get('mean')
            lines.append(f"| {name} | {outcome} | {result['episodes']} | {percentage(result['stationary_fraction'])} | {percentage(result['firing_stationary_fraction'])} | {percentage(result['movement_cancelled_fraction'])} | {'n/a' if mean is None else f'{mean:.2f}'} | {delay.get('unknown', 0)} |")
    (root / 'outcome-behavior.md').write_text('\n'.join(lines + ['', scope, '']), encoding='utf-8')
    return dict(state='complete', episodes=sum(g['all']['episodes'] for g in groups.values()),
                report=str(root / 'outcome-behavior.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.wait_pid:
            wait_process(args.wait_pid, root / 'outcome-behavior-progress.json')
        result = report(root)
        save(root / 'outcome-behavior-progress.json', result)
        print(result, flush=True)
    except Exception as error:
        save(root / 'outcome-behavior-progress.json', dict(state='failed', error=str(error)))
        raise

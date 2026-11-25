"""Summarize a completed paired native battle matrix."""
import argparse
from collections import Counter
import json
import re
from pathlib import Path

from experiment import read, save


FIELD = re.compile(r"(\w+)=([^ ]+)")


def episode(result, fixture, temporal, rocket_policy=None):
    if not all(result[key] for key in ('capture_valid', 'native_objective_required',
                                       'frame_budget_valid', 'seed_confirmed')):
        raise ValueError(f"Invalid native receipt for seed {result['seed']}")
    if result['generated_fixture'] != fixture:
        raise ValueError(f"Native receipt has the wrong fixture for seed {result['seed']}")
    root = Path(result['root'])
    life = result['first_life']
    generation = life['spawncount']
    actor = life['bot_entity']
    start, end = 100, life['end_frame']
    objectives, self_damage, monster_damage = {}, Counter(), Counter()
    with (root/'server.log').open(errors='replace') as log:
        for line in log:
            if not line.startswith(('sv_test_objective ', 'sv_test_damage ')):
                continue
            fields = dict(FIELD.findall(line))
            frame = int(fields['server_frame'])
            if not start <= frame <= end or int(fields['spawncount']) != generation:
                continue
            if line.startswith('sv_test_objective '):
                objectives[frame] = int(fields['alive']) + int(fields['pending'])
            elif int(fields['attacker']) == actor:
                mod = int(fields['mod']) & ~0x8000000
                lost = max(0, int(fields['health_before'])) - max(0, int(fields['health_after']))
                if int(fields['target']) == actor:
                    self_damage[mod] += lost
                elif fields['target_class'].startswith('monster_'):
                    monster_damage[mod] += lost
    clear_frame = None
    owners = Counter()
    with (root/'bot.jsonl').open(encoding='utf-8') as trace:
        for line in trace:
            row = json.loads(line)
            frame = row['frame']
            if row['spawncount'] != generation or not start <= frame <= end or row.get('health', 0) <= 0:
                continue
            if frame not in objectives:
                raise ValueError(f"Missing native objective at {root}, frame {frame}")
            if objectives[frame]:
                selection = (row.get('combat_policy') or {}).get('selection') or {}
                owners[selection.get('owner', 'unobserved')] += 1
            elif clear_frame is None:
                clear_frame = frame
    refreshes = modes = rocket_checks = rocket_defers = 0
    with (root/'bot.err').open(errors='replace') as log:
        for line in log:
            refreshes += line.startswith('lewm_refresh ')
            modes += line.startswith('lewm_mode ')
            if line.startswith('lewm_gate '):
                rocket_checks += 1
                rocket_defers += 'defer=true' in line
    if not rocket_policy and rocket_checks:
        raise ValueError(f"Unexpected rocket gate at {root}")
    if not temporal and refreshes:
        raise ValueError(f"Unexpected temporal controller at {root}")
    if temporal and owners['provider'] and not refreshes:
        raise ValueError(f"Missing temporal controller at {root}")
    win, death = bool(result.get('goal_stop')), bool(result.get('death_stop'))
    if win and (death or clear_frame is None):
        raise ValueError(f"Unconfirmed native win at {root}")
    return dict(seed=result['seed'], map=fixture['map'], weapon=fixture['inventory']['initial_weapon'],
                win=win, death=death, timeout=not (win or death), clear=clear_frame is not None,
                clear_frame=clear_frame, self_hp=life['damage']['self_health_damage'],
                rocket_hp=self_damage[9], hand_grenade_hp=sum(self_damage[k] for k in (15, 16, 24)),
                monster_hp=sum(monster_damage.values()), rocket_monster_hp=monster_damage[8]+monster_damage[9],
                owners=dict(owners), fallback=any(owner not in ('provider', 'unobserved') for owner in owners),
                unobserved=owners['unobserved'],
                refreshes=refreshes, mode_choices=modes, rocket_checks=rocket_checks,
                rocket_defers=rocket_defers)


def report(root):
    manifest = read(root/'complete.json')
    if manifest['state'] != 'complete':
        raise ValueError('Matrix did not complete')
    fixtures = {f['engine_seed']: f for t in read(root/'plan.json')['tasks'] for f in t['instances']}
    rows = {model['name']: [] for model in manifest['models']}
    for job in manifest['jobs']:
        receipt = read(Path(job['out'])/'report.json')
        if not (receipt['capture_complete'] and receipt['provenance_valid']):
            raise ValueError(f"Incomplete capture at {job['out']}")
        for result in receipt['results']:
            rows[job['variant']].append(episode(result, fixtures[result['seed']],
                                                job.get('temporal_policy'), job.get('rocket_policy')))
    totals = {}
    for name, cases in rows.items():
        if Counter(r['seed'] for r in cases) != Counter(fixtures.keys()):
            raise ValueError(f'Unpaired or duplicate outcomes for {name}')
        totals[name] = {key: sum(row[key] for row in cases) for key in
                        ('win', 'death', 'timeout', 'clear', 'self_hp', 'rocket_hp', 'hand_grenade_hp', 'fallback', 'unobserved')}
        for key in ('rocket_checks', 'rocket_defers', 'monster_hp', 'rocket_monster_hp'):
            totals[name][key] = sum(row[key] for row in cases)
    comparisons = {}
    for base in ('u1', 'warm-s1729'):
        if base not in rows:
            continue
        reference = {row['seed']: row for row in rows[base]}
        comparisons[base] = {}
        for name, cases in rows.items():
            if name == base:
                continue
            gained = [row['seed'] for row in cases if row['clear'] and not reference[row['seed']]['clear']]
            lost = [row['seed'] for row in cases if reference[row['seed']]['clear'] and not row['clear']]
            comparisons[base][name] = dict(gained_clearances=gained, lost_clearances=lost)
    save(root/'summary.json', dict(totals=totals, comparisons=comparisons, episodes=rows))
    print(totals)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('matrix', type=Path)
    report(parser.parse_args().matrix)

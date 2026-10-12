"""Native damage by weapon and observed MG/Blaster firing events; no inference."""
import argparse
import collections
import pathlib
import re

from audit_registered_damage_timing import damage_events
from process_combat_architecture_pool import read, save, sha
from report_combat_machinegun_hits import FIRE, fields, validate_shot

WEAPONS = {1: 'Blaster', 2: 'Shotgun', 3: 'Super Shotgun', 4: 'Machinegun'}
SPAWN = re.compile(r'^sv_test_projectile spawncount=(-?\d+) server_frame=(\d+) '
    r'g_test_projectile version=1 event=spawn (.*)$')


def eligible(event, life, release):
    return (event['map'] == life['map'] and event['spawncount'] == life['spawncount']
        and life['start_frame'] <= event['frame'] <= life['end_frame']
        and event['game_frame'] >= release and event['attacker'] == life['bot_entity'])


def weapon_damage(events, life, release, observed_supported_frame=None):
    totals = collections.defaultdict(collections.Counter)
    for event in events:
        if not eligible(event, life, release) or event['attacker_class'] != 'player':
            continue
        if not event['target_class'].startswith('monster_'):
            continue
        take = max(0, event['health_before']) - max(0, event['health_after'])
        if take <= 0:
            continue
        result = totals[str(event['mod'])]
        result['health_damage'] += take
        result['positive_damage_contacts'] += 1
        result['kills'] += int(event['health_before'] > 0 and event['health_after'] <= 0)
        if observed_supported_frame is not None and event['frame'] >= observed_supported_frame:
            result['health_damage_after_supported_observation'] += take
    return {k: dict(v) for k, v in totals.items()}


def physical_events(lines, life, release):
    counts = collections.Counter()
    ids = set()
    projectile_ready = False
    for line in lines:
        line = line.rstrip('\r\n')
        if line == 'g_test_projectile ready version=1':
            projectile_ready = True
        match = FIRE.fullmatch(line)
        if match:
            generation, frame, tail = match.groups()
            f = fields(tail)
            validate_shot(f)
            key = ('hitscan', int(generation), int(f['shot']))
            assert key not in ids
            ids.add(key)
            event = dict(map=f['map'], spawncount=int(generation), frame=int(frame),
                game_frame=int(f['frame']), attacker=int(f['actor']))
            if eligible(event, life, release):
                counts['machinegun_fire_events'] += 1
                counts['machinegun_monster_trace_contacts'] += int(
                    f['target_class'].startswith('monster_') and int(f['health_before']) > 0)
                counts['machinegun_muzzle_blocked_events'] += int(f['muzzle_blocked'])
        match = SPAWN.fullmatch(line)
        if match:
            generation, frame, tail = match.groups()
            f = fields(tail)
            assert projectile_ready
            key = ('projectile', int(generation), int(f['shot']))
            assert key not in ids
            ids.add(key)
            event = dict(map=f['map'], spawncount=int(generation), frame=int(frame),
                game_frame=int(f['frame']), attacker=int(f['attacker']))
            if eligible(event, life, release) and int(f['mod']) == 1:
                counts['blaster_spawn_events'] += 1
    assert projectile_ready, 'Projectile telemetry readiness absent'
    return dict(counts)


def main():
    ap = argparse.ArgumentParser()
    for name in ('evaluation', 'spec', 'recovery', 'out'):
        ap.add_argument('--'+name, type=pathlib.Path, required=True)
    args = ap.parse_args()
    root = args.evaluation.resolve()
    assert not args.out.exists()
    seal = read(root/'complete.json')
    spec = read(args.spec)
    assert seal['state'] == 'complete' and not seal['training_performed']
    assert sha(args.spec) == seal['spec_sha256']
    pool_path = root/'pool/report.json'
    pool = read(pool_path)
    assert sha(pool_path) == seal['pool_sha256'] and pool['state'] == 'complete'
    assert pool['source_unchanged'] and pool['usable_captures'] == len(pool['jobs'])
    recovery = read(args.recovery)
    assert recovery['state'] == 'complete' and recovery['pool_sha256'] == seal['pool_sha256']
    recovery_cases = {c['job']: c for v in recovery['variants'].values() for c in v['cases']}
    pins = {root/'complete.json': sha(root/'complete.json'), args.spec: sha(args.spec),
            args.recovery: sha(args.recovery), pool_path: sha(pool_path)}
    variants = {}
    for index, variant in enumerate(spec['variants']):
        plan_path = pathlib.Path(pool['plans'][index]['path'])
        assert sha(plan_path) == pool['plans'][index]['sha256']
        pins[plan_path] = sha(plan_path)
        plan = read(plan_path)
        cases = []
        for job in pool['jobs']:
            if job['plan_index'] != index:
                continue
            task = plan['tasks'][job['task_index']]
            if not task['episode']['id'].endswith('-recovery'):
                continue
            assert not job['error'] and job['seed'] in task['seeds']
            case_root = pathlib.Path(job['root'])
            cp, mp = case_root/'report.json', case_root/'manifest.json'
            capture, manifest = read(cp), read(mp)
            assert capture['capture_complete'] and capture['provenance_valid']
            assert len(capture['results']) == 1
            result = capture['results'][0]
            life = result['first_life']
            assert result['capture_valid'] and result['dispatch_valid'] and result['seed_confirmed']
            assert manifest['release_game_frame'] > 0 and manifest['synchronous']
            log = pathlib.Path(result['root'])/'server.log'
            for p in (cp, mp, log):
                pins[p] = sha(p)
            with log.open(encoding='utf-8-sig') as stream:
                lines = list(stream)
            events = list(damage_events(lines))
            before = weapon_damage(events, life, 1)
            assert sum(v['health_damage'] for v in before.values()) == sum(
                m['monster_health_damage'] for m in life['damage']['by_mod'])
            observed = None
            if variant['mode'] == 'learned':
                rec = recovery_cases[job['job']]
                assert rec['capture_sha256'] == pins[cp] and rec['seed'] == job['seed']
                if rec['first_supported_delay_frames'] is not None:
                    observed = rec['initial']['frame'] + rec['first_supported_delay_frames']
            cases.append(dict(job=job['job'], seed=job['seed'], episode=task['episode']['id'],
                goal=bool(result['goal_stop']), first_supported_observed_network_frame=observed,
                damage=weapon_damage(events, life, manifest['release_game_frame'], observed),
                firing_events=physical_events(lines, life, manifest['release_game_frame']),
                server_log_sha256=pins[log], capture_sha256=pins[cp]))
        expected = sum(len(t['seeds']) for t in plan['tasks'] if t['episode']['id'].endswith('-recovery'))
        assert len(cases) == expected > 0
        totals = collections.defaultdict(collections.Counter)
        for c in cases:
            for mod, count in c['damage'].items():
                totals[mod].update(count)
        variants[variant['label']] = dict(cases=cases, wins=sum(c['goal'] for c in cases),
            damage={k: dict(v) for k, v in totals.items()},
            firing_events=dict(sum((collections.Counter(c['firing_events']) for c in cases), collections.Counter())))
    for p, digest in pins.items():
        assert sha(p) == digest, str(p)
    args.out.mkdir(parents=True)
    save(args.out/'complete.json', dict(state='complete', version='registered_weapon_damage_v1',
        variants=variants, inputs=[dict(path=str(p), sha256=d) for p, d in pins.items()],
        script_sha256=sha(__file__), training_performed=False, promotion=False,
        scope='Native first-life health damage capped at remaining monster health, by native MOD; not gun-frame shot inference. MG fire and Blaster spawn event counts only; no Shotgun/Super Shotgun shot counts. Effects after observed equipment can originate from earlier projectiles; boundaries are network frames. Descriptive associations, no counterfactual or causal weapon-choice claim.'))
    lines = ['# Recovery: native weapon damage', '',
        '| Variant | Wins | Blaster damage | Shotgun damage | Super Shotgun damage | Machinegun damage | MG fire events | Blaster spawn events |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label, v in variants.items():
        cells = [str(v['damage'].get(str(mod), {}).get('health_damage', 0)) for mod in WEAPONS]
        events = v['firing_events']
        lines.append('| '+label+' | '+str(v['wins'])+'/'+str(len(v['cases']))+' | '+
            ' | '.join(cells)+' | '+str(events.get('machinegun_fire_events', 0))+' | '+
            str(events.get('blaster_spawn_events', 0))+' |')
    lines += ['', 'MOD health effects and native firing events have different denominators. '
        'This is not accuracy or proof a different weapon would win. No model execution or training.', '']
    (args.out/'results.md').write_text('\n'.join(lines), encoding='utf-8')
    print(args.out/'results.md')


if __name__ == '__main__':
    main()

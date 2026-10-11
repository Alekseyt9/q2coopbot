"""Native first health damage timing; no inference, shot count or attribution."""
import argparse
import pathlib
import re

from process_combat_architecture_pool import read, save, sha

PATTERN = re.compile(
    r'^sv_test_damage spawncount=(-?\d+) server_frame=(\d+) '
    r'g_test_damage version=1 map=(\w+) frame=(\d+) attacker=(\d+) '
    r'target=(\d+) inflictor=(\d+) mod=(\d+) health_before=(-?\d+) '
    r'health_after=(-?\d+) take=(\d+) armor=(\d+) power=(\d+) '
    r'protection=(\d+) target_class=(\w+) attacker_class=(\w+)(?: shot=(\d+))?$')
FIELDS = ('spawncount', 'frame', 'map', 'game_frame', 'attacker', 'target',
          'inflictor', 'mod', 'health_before', 'health_after', 'take', 'armor',
          'power', 'protection', 'target_class', 'attacker_class', 'shot')


def damage_events(lines):
    ready = False
    previous = {}
    for line in lines:
        line = line.rstrip('\r\n')
        if line == 'g_test_damage ready version=1':
            ready = True
            continue
        if not line.startswith(('sv_test_damage ', 'g_test_damage ')):
            continue
        match = PATTERN.fullmatch(line)
        assert ready and match, 'Missing ready marker or malformed native damage'
        event = {name: (value if name in ('map', 'target_class', 'attacker_class')
                        else int(value or 0)) for name, value in zip(FIELDS, match.groups())}
        assert event['frame'] >= 1
        assert event['health_before'] - event['take'] == event['health_after']
        world = event['map'], event['spawncount']
        assert event['frame'] >= previous.get(world, 0), 'Nonmonotonic server frame'
        previous[world] = event['frame']
        yield event
    assert ready, 'Native damage telemetry not confirmed'


def summarize_events(events, life, release):
    assert release > 0 and life['start_frame'] <= life['end_frame']
    first = None
    damage = kills = contacts = 0
    for event in events:
        # Native game_frame and network server_frame are different clocks.
        # Use server_frame for the sealed first-life bounds; game_frame for release.
        if (event['map'] != life['map'] or event['spawncount'] != life['spawncount']
                or not life['start_frame'] <= event['frame'] <= life['end_frame']
                or event['game_frame'] < release
                or event['attacker'] != life['bot_entity']
                or event['attacker_class'] != 'player'
                or not event['target_class'].startswith('monster_')):
            continue
        take = max(0, event['health_before']) - max(0, event['health_after'])
        if take <= 0:
            continue
        damage += take
        contacts += 1
        kills += int(event['health_before'] > 0 and event['health_after'] <= 0)
        first = event['game_frame'] if first is None else min(first, event['game_frame'])
    return dict(first_monster_health_damage_game_frame=first,
                first_monster_health_damage_delay_native_frames=(first-release if first is not None else None),
                no_monster_health_damage=first is None, post_release_monster_health_damage=damage,
                post_release_monster_damage_contacts=contacts, post_release_monster_kills=kills)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    pool_path = args.pool / 'report.json'
    pool_hash = sha(pool_path)
    pool = read(pool_path)
    assert pool['state'] == 'complete' and pool['source_unchanged']
    assert pool['usable_captures'] == len(pool['jobs']) and not any(j['error'] for j in pool['jobs'])
    plans = []
    for binding in pool['plans']:
        assert sha(binding['path']) == binding['sha256']
        plans.append(read(binding['path']))
    cases = []
    for job in sorted(pool['jobs'], key=lambda j:j['job']):
        root = pathlib.Path(job['root'])
        capture_path, manifest_path = root/'report.json', root/'manifest.json'
        capture_hash, manifest_hash = sha(capture_path), sha(manifest_path)
        capture, manifest = read(capture_path), read(manifest_path)
        assert capture['capture_complete'] and capture['provenance_valid'] and len(capture['results']) == 1
        result = capture['results'][0]
        assert result['capture_valid'] and result['dispatch_valid'] and result['seed_confirmed']
        assert manifest['synchronous'] and manifest['fixed_world_hold'] and manifest['release_game_frame'] > 0
        plan = plans[job['plan_index']]
        assert job['mode'] in plan['tasks'][job['task_index']]['modes'] and manifest['provider'] == job['mode']
        if job['mode'] == 'learned':
            assert manifest['model_weights_sha256'].lower() == plan['model_sha256'] == sha(plan['model_path'])
        else:
            assert job['mode'] == 'rules' and not manifest.get('model_weights_sha256')
        log = pathlib.Path(result['root'])/'server.log'
        log_hash = sha(log)
        with log.open(encoding='utf-8-sig') as stream:
            events = list(damage_events(stream))
        case = summarize_events(events, result['first_life'], manifest['release_game_frame'])
        assert sha(log) == log_hash
        total = sum(m['monster_health_damage'] for m in result['first_life']['damage']['by_mod'])
        all_life = summarize_events(events, result['first_life'], 1)
        assert all_life['post_release_monster_health_damage'] == total, 'Raw native damage differs from sealed first-life summary'
        assert case['post_release_monster_health_damage'] <= total
        assert sha(capture_path) == capture_hash and sha(manifest_path) == manifest_hash
        case.update(job=job['job'], plan_index=job['plan_index'], seed=job['seed'],
                    episode=result['generated_fixture']['episode_id'], map=result['first_life']['map'],
                    mode=job['mode'], release_game_frame=manifest['release_game_frame'],
                    first_life_end_server_frame=result['first_life']['end_frame'],
                    pre_release_monster_health_damage=total-case['post_release_monster_health_damage'],
                    goal=bool(result['goal_stop']), server_log_sha256=log_hash,
                    capture_sha256=capture_hash, manifest_sha256=manifest_hash)
        cases.append(case)
    variants = {}
    for index in range(len(plans)):
        selected = [c for c in cases if c['plan_index'] == index]
        assert selected
        delays = [c['first_monster_health_damage_delay_native_frames'] for c in selected if not c['no_monster_health_damage']]
        variants[str(index)] = dict(cases=len(selected), cases_without_monster_health_damage=len(selected)-len(delays),
            first_damage_delay_native_frames_mean=sum(delays)/len(delays) if delays else None,
            first_damage_delay_native_frames_max=max(delays, default=None),
            post_release_monster_health_damage=sum(c['post_release_monster_health_damage'] for c in selected))
    assert sha(pool_path) == pool_hash
    save(args.out, dict(version='registered_native_damage_timing_v1', state='complete',
        pool_sha256=pool_hash, script_sha256=sha(__file__), cases=cases, variants=variants,
        scope='First-life native monster health damage after the native game release frame, not first attack intent or first shot. Server/game clocks kept separate. Positive health damage capped at remaining health; corpses, other worlds/lives/actors and pre-release effects excluded. Delay mean/max conditional on cases with damage; no-damage cases reported separately, never imputed. Effects may come from earlier projectiles; no attribution to the latest action, shot accuracy, model execution or training.'))
    print(variants, flush=True)


if __name__ == '__main__':
    main()

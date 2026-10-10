"""Measure first-life attack delays from sealed native evaluation, without NN execution."""
import argparse
import collections
import json
import pathlib
import re
import statistics

from native_projectile_window import records
from process_combat_architecture_pool import read, save, sha
from run_combat_spatial_ppo import wait_process

RELEASE = re.compile(r'^sv_test_combat spawncount=(-?\d+) server_frame=(\d+) g_test_combat_start game_frame=\d+ ready=1 seed=(\d+)$')
SPAWN = re.compile(r'^sv_test_projectile spawncount=(-?\d+) server_frame=(\d+) g_test_projectile version=1 event=spawn map=\w+ frame=\d+ shot=\d+ entity=\d+ (.*)$')


def measure(worker, seed):
    steps, log = worker / 'dataset/steps.jsonl', worker / 'server.log'
    hashes = dict(steps=sha(steps), server_log=sha(log))
    rows = []
    with steps.open(encoding='utf-8-sig') as stream:
        for line in stream:
            row = json.loads(line)
            obs, native, execution = row['observation'], row.get('native_step'), row.get('server_execution', {})
            if obs['identity']['life'] == 1 and obs['health'] > 0 and native and execution.get('matched') and execution.get('window_exclusive'):
                rows.append(row)
    assert rows, 'No exclusive first-life alive commands'
    windows = {(n['spawncount'], n['begin_frame'], n['actor'], n['sequence']) for n in (r['native_step'] for r in rows)}
    releases, launches = [], []
    with log.open(encoding='utf-8-sig') as stream:
        for window, line in records(stream):
            match = RELEASE.match(line)
            if match and int(match[3]) == seed:
                releases.append((int(match[1]), int(match[2])))
            match = SPAWN.match(line)
            if match and window in windows:
                fields = dict(v.split('=', 1) for v in match[3].split())
                if int(fields['attacker']) == window[2] and int(fields['mod']) == 1:
                    launches.append(int(match[2]))
    first = rows[0]['native_step']
    assert releases == [(first['spawncount'], first['begin_frame'])], 'Release differs from first exported command'
    release = first['begin_frame']
    first_frame = lambda predicate: next((r['native_step']['begin_frame'] for r in rows if predicate(r)), None)
    requested = first_frame(lambda r: r['action']['attack'])
    sent = first_frame(lambda r: bool(r['sent_command']['Buttons'] & 1))
    clear = first_frame(lambda r: any(e.get('clear_shot') for e in r['observation']['enemies']))
    ammo = first_frame(lambda r: r['observation']['weapon'] in ('Machinegun', 'models/weapons/v_machn/tris.md2')
        and r['next_observation']['identity']['life'] == 1 and r['next_observation']['health'] > 0
        and r['next_observation']['weapon'] == r['observation']['weapon']
        and r['observation']['ammo'] - r['next_observation']['ammo'] == 1)
    launch = min(launches) if launches else None
    reasons = collections.Counter()
    for row in rows:
        if sent is not None and row['native_step']['begin_frame'] >= sent:
            break
        for intervention in row.get('interventions', []):
            if intervention['component'] == 'attack':
                reasons[intervention['reason']] += 1
    assert hashes == dict(steps=sha(steps), server_log=sha(log)), 'Inputs changed'
    delay = lambda frame: None if frame is None else frame - release
    return dict(seed=seed, release_frame=release, initial_weapon=rows[0]['observation']['weapon'],
        delay_frames=dict(requested_attack=delay(requested), sent_attack=delay(sent), clear_target=delay(clear),
                          native_blaster_launch=delay(launch), machinegun_ammo_proxy=delay(ammo)),
        request_to_sent_frames=None if requested is None or sent is None else sent-requested,
        sent_to_blaster_launch_frames=None if sent is None or launch is None else launch-sent,
        attack_guard_reasons_before_first_sent=reasons, source_sha256=hashes, worker=str(worker))


def distribution(values):
    known = [v for v in values if v is not None]
    return dict(known=len(known), unknown=len(values)-len(known),
        mean=statistics.mean(known) if known else None, median=statistics.median(known) if known else None,
        maximum=max(known) if known else None, over_five_frames=sum(v > 5 for v in known))


def report(root):
    protocol, proof = read(root / 'protocol.json'), read(root / 'recovery/verified-members.json')
    assert proof['state'] == 'complete' and proof['protocol_sha256'] == sha(root / 'protocol.json')
    groups = {}
    for entry in protocol['evaluations']:
        episodes = []
        for path in sorted(pathlib.Path(entry['root']).glob('case-*/s-*/report.json')):
            capture = read(path)
            assert proof['members'][str(path.parent)]['report_sha256'] == sha(path)
            assert capture['capture_complete'] and capture['provenance_valid'] and len(capture['results']) == 1
            result = capture['results'][0]
            assert result['capture_valid'] and result['dispatch_valid'] and result['dataset']['command_proof']['accepted']
            episodes.append(measure(pathlib.Path(result['root']), result['seed']))
        assert len(episodes) == protocol['episodes_per_model']
        delays = lambda subset: {name: distribution([e['delay_frames'][name] for e in subset]) for name in episodes[0]['delay_frames']}
        weapons = sorted({e['initial_weapon'] for e in episodes})
        groups[entry['model']+'-'+entry['label']] = dict(episodes=episodes, delays=delays(episodes),
            by_initial_weapon={weapon: dict(episodes=sum(e['initial_weapon'] == weapon for e in episodes),
                delays=delays([e for e in episodes if e['initial_weapon'] == weapon])) for weapon in weapons})
    save(root / 'first-attack.json', dict(version='combat_first_attack_v1', state='complete', groups=groups,
        protocol_sha256=sha(root / 'protocol.json'), proof_sha256=sha(root / 'recovery/verified-members.json'),
        scope='Delays in native server frames from verified combat release; preparation and later lives excluded. Sent attack is a command, actual Blaster launch is native mod1 projectile telemetry. Machinegun ammo decrement is an observation proxy, not per-shot native proof. Unknown values retained; weapon-specific launch denominators differ. No neural execution.'))
    return {name: group['delays'] for name, group in groups.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.wait_pid:
        wait_process(args.wait_pid, root / 'first-attack-progress.json', stage='waiting_for_sealed_evaluation')
    result = report(root)
    save(root / 'first-attack-progress.json', dict(state='complete', report=str(root / 'first-attack.json')))
    print(json.dumps(result), flush=True)

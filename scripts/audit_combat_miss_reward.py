"""Native Blaster miss-credit audit, no training or policy execution.

Only geometry/sky endings with no damage contact are confirmed misses.
Unresolved/freed projectiles and events outside an alive first-life command
window cannot generate the proposed delayed reward. Server events are labels.
"""
import argparse, collections, json, pathlib, re
from process_combat_architecture_pool import read, save, sha
from native_projectile_window import records

PROJECTILE = re.compile(r'^sv_test_projectile spawncount=(-?\d+) server_frame=(\d+) g_test_projectile version=1 event=(spawn|end) map=\w+ frame=\d+ shot=(\d+) entity=(\d+) (.*)$')
DAMAGE = re.compile(r'^sv_test_damage spawncount=(-?\d+) server_frame=(\d+) g_test_damage version=1 map=\w+ frame=\d+ (.*)$')


def episode(worker, min_frame=0):
    steps = worker / 'dataset/steps.jsonl'
    log = worker / 'server.log'
    windows = {}
    with steps.open(encoding='utf-8-sig') as stream:
        for line in stream:
            step = json.loads(line)
            obs = step['observation']
            native = step.get('native_step')
            execution = step.get('server_execution') or {}
            if (native and obs['identity']['life'] == 1 and obs['health'] > 0
                    and obs['identity']['frame'] > min_frame
                    and execution.get('matched') and execution.get('window_exclusive')
                    and execution.get('recovery_commands', 0) == 0):
                key = (native['spawncount'], native['begin_frame'], native['actor'], native['sequence'])
                assert key not in windows
                windows[key] = step['index']
    shots, contacts = {}, {}
    ready = False
    with log.open(encoding='utf-8-sig') as stream:
        for window, line in records(stream):
            if line == 'g_test_projectile ready version=1':
                ready = True
            match = PROJECTILE.match(line)
            if match:
                generation, frame, event, shot, entity, tail = match.groups()
                generation, frame, shot, entity = map(int, (generation, frame, shot, entity))
                key = (generation, shot)
                fields = dict(v.split('=', 1) for v in tail.split())
                if event == 'spawn':
                    assert key not in shots
                    shots[key] = dict(entity=entity, mod=int(fields['mod']), attacker=int(fields['attacker']),
                                      launch_frame=frame, launch_window=window, outcome='unresolved')
                else:
                    assert key in shots and shots[key]['outcome'] == 'unresolved'
                    assert shots[key]['entity'] == entity
                    shots[key].update(outcome=fields['outcome'], end_frame=frame, end_window=window)
            match = DAMAGE.match(line)
            if match:
                generation, _, tail = match.groups()
                fields = dict(v.split('=', 1) for v in tail.split())
                shot = int(fields.get('shot', 0))
                if shot:
                    key = (int(generation), shot)
                    assert key not in contacts
                    contacts[key] = fields
    assert ready
    counts = collections.Counter()
    events = []
    for key, shot in shots.items():
        launch = shot['launch_window']
        if shot['mod'] != 1 or launch not in windows or launch[2] != shot['attacker']:
            continue
        counts['eligible_launches'] += 1
        contact = contacts.get(key)
        if contact:
            assert int(contact['attacker']) == shot['attacker']
            assert int(contact['inflictor']) == shot['entity'] and int(contact['mod']) == 1
        if shot['outcome'] not in ('geometry', 'sky'):
            counts[shot['outcome']] += 1
            continue
        assert contact is None, 'Contradictory miss and native damage contact'
        counts['confirmed_misses'] += 1
        end = shot['end_window']
        if end not in windows or end[2] != shot['attacker']:
            counts['misses_without_alive_first_life_end_window'] += 1
            continue
        assert end[0] == launch[0] and shot['end_frame'] >= shot['launch_frame']
        counts['creditable_misses'] += 1
        events.append(dict(spawncount=key[0], shot=key[1], outcome=shot['outcome'],
                           launch_step=windows[launch], end_step=windows[end],
                           launch_frame=shot['launch_frame'], end_frame=shot['end_frame'],
                           delay_frames=shot['end_frame']-shot['launch_frame']))
    return dict(counts=dict(counts), events=events, steps=str(steps), steps_sha256=sha(steps),
                server_log=str(log), server_sha256=sha(log))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=pathlib.Path, required=True)
    ap.add_argument('--penalty', type=float, default=-0.02)
    ap.add_argument('--min-frame', type=int, default=0,
                    help='Exclusive client frame cutoff; 100 matches the weapon quality reports')
    a = ap.parse_args()
    assert -0.1 <= a.penalty < 0
    assert a.min_frame >= 0
    root = a.root.resolve()
    protocol = read(root / 'protocol.json')
    proof = read(root / 'recovery/verified-members.json')
    assert proof['state'] == 'complete' and proof['protocol_sha256'] == sha(root / 'protocol.json')
    groups, sources = {}, []
    for entry in protocol['evaluations']:
        counts = collections.Counter()
        episodes = []
        for path in sorted(pathlib.Path(entry['root']).glob('case-*/s-*/report.json')):
            assert proof['members'][str(path.parent)]['report_sha256'] == sha(path)
            report = read(path)
            assert report['capture_complete'] and report['provenance_valid']
            worker = pathlib.Path(report['results'][0]['root'])
            data = episode(worker, a.min_frame)
            counts.update(data['counts'])
            episodes.append(dict(seed=report['results'][0]['seed'], **data))
        assert len(episodes) == protocol['episodes_per_model']
        groups[entry['model']+'-'+entry['label']] = dict(counts=dict(counts), episodes=episodes,
            proposed_total_reward_delta=counts['creditable_misses']*a.penalty,
            proposed_mean_episode_reward_delta=counts['creditable_misses']*a.penalty/len(episodes))
    filename = 'miss-reward-audit.json' if a.min_frame == 0 else f'miss-reward-audit-after-frame-{a.min_frame}.json'
    save(root / filename, dict(version='native_blaster_miss_credit_audit_v1',
         groups=groups, penalty=a.penalty, protocol_sha256=sha(root / 'protocol.json'),
         exclusive_client_frame_cutoff=a.min_frame,
         member_proof_sha256=sha(root / 'recovery/verified-members.json'),
         scope='Descriptive proposed reward only. No weights, stored rewards or training changed. '
               'Credit at native impact/end command, not launch. Unknown/freed and post-life endings '
               'have no penalty. No Machinegun accounting or selected-target credit.'))
    print(json.dumps({k:dict(counts=v['counts'], mean_reward_delta=v['proposed_mean_episode_reward_delta'])
                      for k,v in groups.items()}), flush=True)


if __name__ == '__main__':
    main()

"""Measure first actual native fire after scene release and visible opportunity.

Unlike hit-rate reports, this includes the first frames after verified release.
No NN inference; no wall-time/timescale conversion or attack-command proxy.
"""
import argparse
import json
import pathlib
import re
import statistics

from native_projectile_window import records
from process_combat_architecture_pool import read, save, sha

RELEASE = re.compile(r'^sv_test_combat spawncount=(-?\d+) server_frame=(\d+) g_test_combat_start game_frame=(\d+) ready=1 seed=(\d+)$')
HITSCAN = re.compile(r'^sv_test_hitscan spawncount=(-?\d+) server_frame=(\d+) g_test_hitscan version=1 event=fire (.*)$')
PROJECTILE = re.compile(r'^sv_test_projectile spawncount=(-?\d+) server_frame=(\d+) g_test_projectile version=1 event=spawn (.*)$')


def measure(member):
    root = pathlib.Path(member['root'])
    log = root/'server.log'
    release = None
    fires = []
    with log.open(encoding='utf-8-sig') as stream:
        for window,line in records(stream):
            m = RELEASE.match(line)
            if m:
                assert release is None, 'Multiple scene releases'
                generation,frame,game_frame,seed = map(int,m.groups())
                assert seed==member['seed']
                release = (generation,frame)
            for pattern,mod,actor_field in ((HITSCAN,4,'actor'),(PROJECTILE,1,'attacker')):
                m = pattern.match(line)
                if m and window:
                    generation,frame,tail = m.groups()
                    fields = dict(word.split('=',1) for word in tail.split())
                    if int(fields['mod'])==mod and int(fields[actor_field])==window[2]:
                        assert int(generation)==window[0]
                        fires.append((window,int(frame)))
    assert release is not None, 'No native scene release'
    valid = set()
    opportunities = []
    with (root/'dataset/steps.jsonl').open(encoding='utf-8-sig') as stream:
        for line in stream:
            step = json.loads(line); o = step['observation']; native = step.get('native_step')
            if (not native or o['identity']['life']!=1 or o['health']<=0 or
                    not step.get('server_execution',{}).get('matched')): continue
            key = tuple(native[k] for k in ('spawncount','begin_frame','actor','sequence'))
            if key[0]!=release[0] or key[1]<release[1]: continue
            valid.add(key)
            if any(e.get('clear_shot') is True for e in o['enemies']):
                opportunities.append(key[1])
    first_visible = min(opportunities) if opportunities else None
    matched = [(window,frame) for window,frame in fires if window in valid]
    first_fire = min((frame for _,frame in matched),default=None)
    return dict(seed=member['seed'],release_frame=release[1],first_visible_frame=first_visible,
                first_native_fire_frame=first_fire,native_fires=len(matched),
                release_to_first_fire_game_seconds=(first_fire-release[1])*.1 if first_fire is not None else None,
                visible_to_first_fire_game_seconds=(first_fire-first_visible)*.1 if first_fire is not None and first_visible is not None and first_fire>=first_visible else None,
                source_server_sha256=sha(log),source_steps_sha256=sha(root/'dataset/steps.jsonl'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);a=ap.parse_args()
    proof=read(a.root/'recovery/verified-members.json'); protocol=read(a.root/'protocol.json')
    assert proof['state']=='complete' and proof['protocol_sha256']==sha(a.root/'protocol.json')
    rows=[];groups={}
    for entry in protocol['evaluations']:
        label=entry['model']+'-'+entry['label'];members=[]
        for path in pathlib.Path(entry['root']).glob('case-*/s-*/report.json'):
            assert sha(path)==proof['members'][str(path.parent)]['report_sha256']
            row=measure(read(path)['results'][0]);row['variant']=label;members.append(row);rows.append(row)
        latency=[r['visible_to_first_fire_game_seconds'] for r in members if r['visible_to_first_fire_game_seconds'] is not None]
        groups[label]=dict(episodes=len(members),with_native_fire=sum(r['native_fires']>0 for r in members),
            visible_without_native_fire=sum(r['first_visible_frame'] is not None and r['native_fires']==0 for r in members),
            measured_latency_episodes=len(latency),mean_visible_to_first_fire_game_seconds=statistics.mean(latency) if latency else None,
            median_visible_to_first_fire_game_seconds=statistics.median(latency) if latency else None,
            max_visible_to_first_fire_game_seconds=max(latency) if latency else None)
    save(a.root/'first-shot-latency.json',dict(state='complete',groups=groups,members=rows,
         member_proof_sha256=sha(a.root/'recovery/verified-members.json'),
         scope='Actual native mod1/mod4 fire joined to verified alive first-life command windows after native scene release. All release frames included. Game seconds at10Hz, no x2 wall scaling. Descriptive opportunity timing, not proof any earlier shot was safe or optimal.'))
    print(json.dumps(groups))


if __name__=='__main__': main()

"""Describe subsequent native trajectories; do not infer an isolated cause."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from native_projectile_window import records
from report_combat_machinegun_hits import DAMAGE, fields, scan
from process_combat_architecture_pool import read,save,sha


def wrap(x):return (x+180)%360-180


def observed_error(o,a):
    e=next((e for e in o['enemies'] if e['id']==a.get('target_entity')),None)
    if not e or not e.get('clear_shot') or o.get('kick_angles_degrees') is None:return None
    solid=e.get('observed_solid',0);bottom=-((solid>>5)&31)*8;top=((solid>>10)&63)*8-32
    if solid in (0,31) or not solid&31 or top<=bottom:return None
    height=(bottom+top)/2 if top-bottom<16 else min(max(22,bottom+8),top-8)
    p=list(e['relative']);p[2]+=height-(-2 if o['ducked'] else 22)+8
    vp,vy=(x*360/65536 for x in o['view_angles'][:2]);kick=o['kick_angles_degrees']
    desired=[wrap(math.degrees(math.atan2(p[1],p[0]))-vy-kick[1]),
        wrap(-math.degrees(math.atan2(p[2],math.hypot(*p[:2])))-vp-kick[0])]
    dy=wrap(a['yaw_delta_degrees']-desired[0]);dp=wrap(a['pitch_delta_degrees']-desired[1])
    return dict(yaw=dy,pitch=dp,approx_degrees=math.hypot(dy,dp))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--first-life',action='store_true',help='Exclude post-terminal task actions')
    ap.add_argument('--include-rules',action='store_true',help='Describe hybrid continuations with explicit owner labels')
    args=ap.parse_args();root=args.root.resolve();out=root/('continuation-first-life.json' if args.first_life else 'continuation.json');assert not out.exists()
    complete=read(root/'complete.json');assert complete['state']=='complete'
    assert sha(root/'comparison.json')==complete['comparison_sha256']
    spec,pool=read(root/'spec.json'),read(root/'pool/report.json');result=[];pins={}
    for j in sorted(pool['jobs'],key=lambda j:j['plan_index']):
        assignment=spec['assignments'][j['plan_index']]
        if assignment['repeat']!=0 or assignment['mode']=='identity':continue
        report=Path(j['root'])/'report.json';r=read(report)['results'][0];native=Path(r['root'])
        step_path,trace_path,log_path=(native/name for name in ('dataset/steps.jsonl','bot.jsonl','server.log'))
        steps=[json.loads(line) for line in step_path.open(encoding='utf-8-sig')]
        selected=[s for s in steps if s['observation']['identity']['frame']>=assignment['branch_frame']]
        if args.first_life:
            end=r['first_life']['end_frame'];terminal=bool(r.get('goal_stop') or r.get('death_stop'))
            selected=[s for s in selected if s['observation']['identity']['life']==1 and
                (s['observation']['identity']['frame']<end or not terminal and s['observation']['identity']['frame']==end)]
        assert selected and selected[0]['observation']['identity']['frame']==assignment['branch_frame']
        assert all(b['observation']['identity']['frame']==a['observation']['identity']['frame']+1 for a,b in zip(selected,selected[1:]))
        windows={}
        for s in selected:
            n=s['native_step'];k=(n['spawncount'],n['begin_frame'],n['actor'],n['sequence'])
            assert k not in windows;windows[k]=s['observation']['identity']['frame']
        traces={}
        for line in trace_path.open(encoding='utf-8-sig'):
            c=json.loads(line).get('combat_policy') or {};o=c.get('observation')
            if o and o['identity']['frame'] in windows.values() and c.get('selection',{}).get('owner') in (('provider','rules') if args.include_rules else ('provider',)):
                f=o['identity']['frame'];assert f not in traces;traces[f]=c
        assert set(traces)==set(windows.values())
        damage=defaultdict(Counter)
        for window,line in records(log_path.open(encoding='utf-8-sig')):
            m=DAMAGE.fullmatch(line)
            if m and window in windows:
                d=fields(m[3]);hp=int(d['health_before']);lost=min(max(hp,0),max(0,hp-int(d['health_after'])))
                f=windows[window]
                if int(d['attacker'])==window[2] and d['target_class'].startswith('monster_'):
                    damage[f]['dealt']+=lost;damage[f]['kills']+=int(hp>0 and int(d['health_after'])<=0)
                if int(d['target'])==window[2]:damage[f]['received']+=lost
        shots=defaultdict(list)
        for shot in scan(log_path.open(encoding='utf-8-sig'),set(windows)):
            if shot['eligible']:shots[windows[shot['window']]].append(shot)
        timeline=[]
        for s in selected:
            o=s['observation'];a=s['action'];f=o['identity']['frame'];c=traces[f];cmd=s['sent_command']
            reasons=[v['reason'] for v in c['selection'].get('interventions',[]) if v['component']=='movement']
            nxt=s.get('next_observation')
            moved=math.dist(o['position'],nxt['position']) if nxt is not None else None
            useful=sum(min(int(v['damage']['health_before']),max(0,int(v['damage']['health_before'])-int(v['damage']['health_after'])))
                for v in shots[f] if v['damage'] and v['fields']['target_class'].startswith('monster_') and int(v['damage']['health_before'])>0)
            assert s['owner']==c['selection']['owner']
            timeline.append(dict(frame=f,owner=s['owner'],health=o['health'],on_ground=o['on_ground'],target=a.get('target_entity',0),
                mode=a.get('aim_mode',0),weapon=o['weapon'],request=[a['yaw_delta_degrees'],a['pitch_delta_degrees']],
                attack=a['attack'],requested_movement=[a['forward'],a['side'],a['vertical']],
                sent_movement=[cmd[k] for k in ('Forward','Side','Up')],movement_reasons=reasons,
                next_displacement=moved,current_mg_anchor_error=observed_error(o,a) if 'machn' in o['weapon'].lower() or o['weapon']=='Machinegun' else None,
                native_mg_bullets=len(shots[f]),native_mg_live_damage=useful,native_damage=dict(damage[f])))
        horizons=[]
        for count in (1,3,5,10,20,len(timeline)):
            rows=timeline[:count];reasons=Counter(v for t in rows for v in t['movement_reasons'])
            horizons.append(dict(requested_rows=count,available_rows=len(rows),complete=len(rows)==count,
                provider_rows=sum(t['owner']=='provider' for t in rows),rules_rows=sum(t['owner']=='rules' for t in rows),
                learned_only=all(t['owner']=='provider' for t in rows),
                dealt=sum(t['native_damage'].get('dealt',0) for t in rows),received=sum(t['native_damage'].get('received',0) for t in rows),
                kills=sum(t['native_damage'].get('kills',0) for t in rows),mg_bullets=sum(t['native_mg_bullets'] for t in rows),
                mg_damage=sum(t['native_mg_live_damage'] for t in rows),target_changes=sum(a['target']!=b['target'] for a,b in zip(rows,rows[1:])),
                coarse_frames=sum(t['mode']==0 for t in rows),requested_yaw_travel=sum(abs(t['request'][0]) for t in rows),
                suppressed_movement=sum(max(abs(x) for x in t['requested_movement'][:2])>=.25 and t['sent_movement']==[0,0,0] for t in rows),
                airborne_frames=sum(not t['on_ground'] for t in rows),
                observed_displacement_rows=sum(t['next_displacement'] is not None for t in rows),
                observed_path_length=sum(t['next_displacement'] for t in rows if t['next_displacement'] is not None),
                movement_reasons=dict(reasons)))
        result.append(dict(**assignment,timeline=timeline,horizons=horizons))
        for p in (report,step_path,trace_path,log_path):pins[str(p)]=sha(p)
    for p,h in pins.items():assert sha(p)==h
    save(out,dict(state='complete',source_complete_sha256=sha(root/'complete.json'),cases=result,pins=pins,
        training_performed=False,promotion=False,first_life_only=args.first_life,includes_rules=args.include_rules,
        scope='Subsequent trajectories after an accepted exact-prefix intervention. Horizons count matched network frames [root,root+N), nominal0.1 game seconds each. Native health damage joined by exact command windows; no invented values for incomplete horizons. Optional first-life cutoff excludes post-terminal actions; owner is explicit, hybrid rule frames are NOT learned-policy outcomes or PPO data. Requested rotation is not physical miss rate. Suppressed movement is not lack of displacement. Subsequent decisions/state diverge together; no isolated attribution to movement, target switching or noise.'))
    print(json.dumps([dict(context=c['context_id'],mode=c['mode'],horizons=[h for h in c['horizons'] if h['requested_rows'] in (1,5,10)]) for c in result]))


if __name__=='__main__':main()

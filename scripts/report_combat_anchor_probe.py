"""Native branch-shot outcomes after a closed observed-anchor diagnostic."""
import argparse
import json
import math
from pathlib import Path
import re
from process_combat_architecture_pool import read,save,sha
from report_combat_machinegun_hits import scan,fields,vector


def intersects(start,direction,pose):
    origin,lo,hi=(vector(pose[k]) for k in ('origin','mins','maxs'))
    near,far=0.,float('inf')
    for i in range(3):
        a,b=origin[i]+lo[i]-start[i],origin[i]+hi[i]-start[i]
        if abs(direction[i])<1e-12:
            if not a<=0<=b:return False
        else:
            a,b=sorted((a/direction[i],b/direction[i]));near=max(near,a);far=min(far,b)
            if far<near:return False
    return far>=near


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    a=ap.parse_args();root=a.root.resolve();assert not (root/'branch-effects.json').exists()
    complete=read(root/'complete.json');assert complete['state']=='complete'
    assert sha(root/'comparison.json')==complete['comparison_sha256']
    spec,pool=read(root/'spec.json'),read(root/'pool/report.json');results=[];pins={}
    pattern=re.compile(r'^sv_test_aim_pose spawncount=(-?\d+) server_frame=(\d+) g_test_aim_pose version=1 (.*)$')
    for j in pool['jobs']:
        assignment=spec['assignments'][j['plan_index']]
        r=read(Path(j['root'])/'report.json')['results'][0];native=Path(r['root'])
        steps=native/'dataset/steps.jsonl';log=native/'server.log'
        s=next(json.loads(line) for line in steps.open(encoding='utf-8-sig') if json.loads(line)['observation']['identity']['frame']==assignment['branch_frame'])
        n=s['native_step'];window=(n['spawncount'],n['begin_frame'],n['actor'],n['sequence'])
        with log.open(encoding='utf-8-sig') as stream:shots=[v for v in scan(stream,{window}) if v['eligible']]
        assert len(shots)==1,'Expected same selected MG fire event'
        shot=shots[0];f=shot['fields'];d=shot['damage'];poses=[]
        for line in log.open(encoding='utf-8-sig'):
            m=pattern.fullmatch(line.strip())
            if m and (int(m[1]),int(m[2]))==(shot['generation'],shot['server_frame']):
                p=fields(m[3])
                if (int(p['shot']),int(p['actor']),int(p['entity']))==(int(f['shot']),n['actor'],s['action']['target_entity']):poses.append(p)
        assert len(poses)==1
        p=poses[0];start,direction=vector(f['start']),vector(f['aim'])
        origin,lo,hi=(vector(p[k]) for k in ('origin','mins','maxs'))
        height=(lo[2]+hi[2])/2 if hi[2]-lo[2]<16 else min(max(22,lo[2]+8),hi[2]-8)
        target=[origin[i]+(height if i==2 else 0)-start[i] for i in range(3)]
        cosine=sum(x*y for x,y in zip(direction,target))/math.sqrt(sum(x*x for x in direction)*sum(x*x for x in target))
        useful=bool(d and f['target_class'].startswith('monster_') and int(d['health_before'])>0 and int(d['health_after'])<int(d['health_before']))
        results.append(dict(**assignment,selected_target=s['action']['target_entity'],
            requested_degrees=[s['action']['yaw_delta_degrees'],s['action']['pitch_delta_degrees']],
            applied_degrees=[s['applied_action']['yaw_delta_degrees'],s['applied_action']['pitch_delta_degrees']],
            movement_controls=[s['sent_command'][k] for k in ('Forward','Side','Up')],
            observed_position=s['observation']['position'],next_position=s['next_observation']['position'],
            actual_muzzle=list(start),actual_view=vector(f['view']),actual_recoil=vector(f['recoil']),
            selected_native_bbox_intersection=intersects(start,direction,p),
            selected_native_aimpoint_error_degrees=math.degrees(math.acos(max(-1,min(1,cosine)))),
            actual_contact_class=f['target_class'],actual_contact_entity=int(f['target']),
            live_monster_health_damage=min(int(d['health_before']),max(0,int(d['health_before'])-int(d['health_after']))) if useful else 0,
            actual_spread=vector(f['spread'],2),muzzle_blocked=bool(int(f['muzzle_blocked']))))
        pins[str(steps)]=sha(steps);pins[str(log)]=sha(log)
    for p,h in pins.items():assert sha(p)==h
    save(root/'branch-effects.json',dict(state='complete',records=results,pins=pins,
        source_complete_sha256=sha(root/'complete.json'),training_performed=False,promotion=False,
        scope='Actual same-frame native MG shot, current action and next observed motion. Native shot-time pose is an offline label only. Full delayed fight outcomes remain in comparison.json; repeated seeds are not independent quality samples.'))
    print(json.dumps([{k:r[k] for k in ('context_id','mode','repeat','live_monster_health_damage','selected_native_bbox_intersection','selected_native_aimpoint_error_degrees')} for r in results if r['repeat']==0]))


if __name__=='__main__':main()

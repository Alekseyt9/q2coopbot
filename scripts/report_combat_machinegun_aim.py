"""Compare real shot rays to the explicitly chosen client-observed target."""
import argparse,collections,json,math,pathlib
from process_combat_architecture_pool import read,save,sha
from report_combat_machinegun_hits import scan,vector
from report_combat_selected_target_aim import ray,wrap

def angular(a,b):
    lengths=math.sqrt(sum(v*v for v in a)*sum(v*v for v in b))
    assert lengths>0
    return math.degrees(math.acos(max(-1,min(1,sum(v*w for v,w in zip(a,b))/lengths))))

def measure(shot,step):
    o=step['observation'];action=step['action'];entity=action.get('target_entity',0)
    if step['owner']!='provider':return dict(status='no_declared_target')
    if not entity:
        available=0
        for e in o['enemies']:
            solid=e.get('observed_solid',0);bottom=-((solid>>5)&31)*8;top=((solid>>10)&63)*8-32
            available+=bool(e.get('clear_shot') and solid not in (0,31) and solid&31 and top>bottom)
        return dict(status='provider_no_target_with_visible_bbox' if available else 'provider_no_target_without_visible_bbox')
    enemy=next((e for e in o['enemies'] if e['id']==entity and e.get('observed_track',0)==action.get('target_track',0)),None)
    if not enemy or not enemy.get('clear_shot'):return dict(status='selected_target_unavailable')
    solid=enemy.get('observed_solid',0);bottom=-((solid>>5)&31)*8;top=((solid>>10)&63)*8-32
    if solid in (0,31) or solid&31==0 or top<=bottom:return dict(status='unknown_bbox')
    height=(top+bottom)/2 if top-bottom<16 else min(max(22,bottom+8),top-8)
    f=shot['fields'];start=vector(f['start']);aim=vector(f['aim']);view=vector(f['view']);kick=vector(f['recoil'])
    desired=[o['position'][axis]+enemy['relative'][axis]-start[axis] for axis in range(3)];desired[2]+=height
    if sum(v*v for v in desired)<1e-12:return dict(status='degenerate_target')
    desired_pitch=-math.degrees(math.atan2(desired[2],math.hypot(*desired[:2])))
    applied=step['applied_action'];client_view=[o['view_angles'][i]*360/65536+applied[k] for i,k in ((0,'pitch_delta_degrees'),(1,'yaw_delta_degrees'))]
    damage=shot['damage'];selected_hit=bool(damage and int(damage['target'])==entity and int(damage['health_after'])<int(damage['health_before']) and int(damage['health_before'])>0)
    return dict(status='measured',entity=entity,track=action.get('target_track',0),distance=math.sqrt(sum(v*v for v in desired)),
        pre_recoil_degrees=angular(ray(view[1],view[0]),desired),post_recoil_degrees=angular(aim,desired),
        pre_pitch_error=wrap(view[0]-desired_pitch),post_pitch_error=wrap(view[0]+kick[0]-desired_pitch),
        native_vs_applied_pitch=abs(wrap(view[0]-client_view[0])),native_vs_applied_yaw=abs(wrap(view[1]-client_view[1])),
        native_recoil_pitch=kick[0],observed_kick_pitch=(o.get('kick_angles_degrees') or [None])[0],selected_target_damage=selected_hit)

def summary(rows):
    return dict(shots=len(rows),**{key:sum(r[key] for r in rows)/len(rows) if rows else None for key in ('pre_recoil_degrees','post_recoil_degrees','pre_pitch_error','post_pitch_error','native_vs_applied_pitch','native_vs_applied_yaw')},
        recoil_worsened_shots=sum(r['post_recoil_degrees']>r['pre_recoil_degrees']+1 for r in rows),
        selected_target_damage=sum(r['selected_target_damage'] for r in rows))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);a=ap.parse_args()
    protocol=read(a.root/'protocol.json');proof=read(a.root/'recovery/verified-members.json');assert proof['state']=='complete' and proof['protocol_sha256']==sha(a.root/'protocol.json')
    groups={};sources=[]
    for entry in protocol['evaluations']:
        counts=collections.Counter();values=[]
        for path in pathlib.Path(entry['root']).glob('case-*/s-*/report.json'):
            assert proof['members'][str(path.parent)]['report_sha256']==sha(path)
            root=pathlib.Path(read(path)['results'][0]['root']);steps=root/'dataset/steps.jsonl';log=root/'server.log';by_window={}
            with steps.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    s=json.loads(line);o=s['observation'];n=s.get('native_step')
                    if n and o['identity']['life']==1 and o['identity']['frame']>100 and o['health']>0 and s.get('server_execution',{}).get('matched'):
                        key=(n['spawncount'],n['begin_frame'],n['actor'],n['sequence']);assert key not in by_window;by_window[key]=s
            with log.open(encoding='utf-8-sig') as stream:shots=scan(stream,set(by_window))
            for shot in shots:
                if not shot['eligible']:continue
                item=measure(shot,by_window[shot['window']]);counts[item['status']]+=1
                if item['status']=='measured':values.append(item)
            sources.append(dict(steps_sha256=sha(steps),server_sha256=sha(log),server_log=str(log)))
        strata={}
        for label,predicate in [('recoil_above_6',lambda r:abs(r['native_recoil_pitch'])>6),('recoil_at_most_1.5',lambda r:abs(r['native_recoil_pitch'])<=1.5),('distance_below_128',lambda r:r['distance']<128),('distance_128_to_256',lambda r:128<=r['distance']<256),('distance_at_least_256',lambda r:r['distance']>=256)]:
            strata[label]=summary([r for r in values if predicate(r)])
        groups[entry['model']+'-'+entry['label']]=dict(status_counts=dict(counts),all=summary(values),descriptive_strata=strata)
    save(a.root/'machinegun-selected-target-aim.json',dict(state='complete',protocol_sha256=sha(a.root/'protocol.json'),groups=groups,sources=sources,
        scope='Actual native muzzle/view/aim at each shot versus explicit chosen target from pre-command client snapshot and known bbox. No nearest-target substitution or hidden target coordinates. Snapshot staleness, enemy motion, random spread and geometry affect outcomes; angular error is descriptive and not hit probability. Rules without declared target remain unmeasured. No policy execution.'))
    print(json.dumps(groups),flush=True)

if __name__=='__main__':main()

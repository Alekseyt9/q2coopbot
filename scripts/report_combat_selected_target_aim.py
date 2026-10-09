"""Actual selected-target ray errors; never substitutes nearest enemy as intent."""
import argparse,json,math,pathlib
from process_combat_architecture_pool import read,save,sha

def wrap(angle):return (angle+180)%360-180

def ray(yaw,pitch):
    yaw,pitch=map(math.radians,(yaw,pitch))
    return (math.cos(pitch)*math.cos(yaw),math.cos(pitch)*math.sin(yaw),-math.sin(pitch))

def measure(capture):
    o=capture['observation'];selection=capture.get('selection') or {};candidate=selection.get('candidate') or {}
    if selection.get('owner')!='provider' or o['identity']['life']!=1 or o['identity']['frame']<=100 or o['health']<=0:return None
    entity=candidate.get('target_entity',0)
    if not entity:return dict(status='no_selected_target')
    enemy=next((e for e in o['enemies'] if e['id']==entity and (e.get('observed_track') or 0)==candidate.get('target_track',0)),None)
    if not enemy or not enemy.get('clear_shot'):return dict(status='selected_target_unavailable')
    solid=enemy.get('observed_solid',0);bottom=-((solid>>5)&31)*8;top=((solid>>10)&63)*8-32
    if solid in (0,31) or solid&31==0 or top<=bottom:return dict(status='unknown_bbox')
    height=(top+bottom)/2 if top-bottom<16 else min(max(22,bottom+8),top-8)
    x,y,z=enemy['relative'];z+=height-(-2 if o['ducked'] else 22)
    desired_yaw=math.degrees(math.atan2(y,x));desired_pitch=-math.degrees(math.atan2(z,math.hypot(x,y)))
    applied=capture['applied'];yaw=o['view_angles'][1]*360/65536+applied['yaw_delta_degrees'];pitch=o['view_angles'][0]*360/65536+applied['pitch_delta_degrees']
    a,b=ray(yaw,pitch),ray(desired_yaw,desired_pitch)
    angular=math.degrees(math.acos(max(-1,min(1,sum(u*v for u,v in zip(a,b))))))
    return dict(status='measured',entity=entity,track=candidate.get('target_track',0),frame=o['identity']['frame'],yaw=abs(wrap(yaw-desired_yaw)),pitch=abs(wrap(pitch-desired_pitch)),angular=angular,firing=applied['attack'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);a=ap.parse_args()
    protocol=read(a.root/'protocol.json');quality=read(a.root/'quality-report.json');assert quality['state']=='complete' and quality['protocol_sha256']==sha(a.root/'protocol.json')
    groups={}
    for entry in protocol['evaluations']:
        counters={};values=[];firing=[];switches=0;adjacent=0
        for path in sorted(pathlib.Path(entry['root']).glob('case-*/s-*/report.json')):
            report=read(path);assert report['capture_complete'] and report['provenance_valid'] and len(report['results'])==1
            result=report['results'][0];assert result['capture_valid'] and result['dispatch_valid'];previous=None
            trace=pathlib.Path(result['root'])/'bot.jsonl'
            with trace.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    capture=json.loads(line).get('combat_policy');item=measure(capture) if capture else None
                    if item is None:previous=None;continue
                    counters[item['status']]=counters.get(item['status'],0)+1
                    if item['status']!='measured':previous=None;continue
                    values.append(item)
                    if item['firing']:firing.append(item)
                    if previous and item['frame']==previous['frame']+1:
                        adjacent+=1;switches+=int((item['entity'],item['track'])!=(previous['entity'],previous['track']))
                    previous=item
        def stats(rows):
            return dict(frames=len(rows),**{key+'_mean_degrees':sum(v[key] for v in rows)/len(rows) if rows else None for key in ('yaw','pitch','angular')},above_10_degrees=sum(v['angular']>10 for v in rows)/len(rows) if rows else None)
        groups[entry['model']+'-'+entry['label']]=dict(counters=counters,all=stats(values),firing=stats(firing),target_switches=switches,adjacent_selected_frames=adjacent)
    save(a.root/'selected-target-aim.json',dict(protocol_sha256=sha(a.root/'protocol.json'),groups=groups,scope='First-life provider frames after fixed release. Applied yaw/pitch to explicitly selected observed bbox. No lead/recoil correction in this diagnostic; frame counts are not bullet counts, angular proximity is not hit rate. Legacy controls have no declared target and remain unmeasured.'))

if __name__=='__main__':main()

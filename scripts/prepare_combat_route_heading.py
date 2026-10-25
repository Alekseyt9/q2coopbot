"""Observed static waypoint heading labels, not executed expert actions or PPO."""
import argparse
import math
from pathlib import Path
import numpy as np
from combat_route_yaw_auxiliary_data import admit
from audit_combat_contact_loss_probe import load_steps
from evaluate_registered_combat_models import read,save,sha

LABEL='combat_observed_static_route_heading_v1'


def heading(o):
    if o['health']<=0 or o.get('enemies') or not 0<=o.get('age_ms',0)<=300:return None
    frame=o['identity']['frame'];group=o.get('remembered_static_routes')or{}
    if group.get('version')!='combat_observed_memory_static_routes_v1' or group.get('frame')!=frame:return None
    choices=[]
    for h in group.get('hints',[]):
        age=h['memory_age_frames'];point=h.get('next_static_waypoint_relative')
        if not(h['static_geometry_known'] and h['static_route_found'] and 0<age<=200 and point is not None):continue
        if not all(math.isfinite(v) for v in point) or math.hypot(*point[:2])<8:continue
        if not any(m['id']==h['observed_id'] and m['age_frames']==age for m in o.get('remembered_threats',[])):continue
        choices.append((age,h['static_horizontal_path_length'],h['observed_id'],point))
    if not choices:return None
    age,distance,identity,point=min(choices)
    yaw=o['view_angles'][1]*2*math.pi/65536
    turn=math.atan2(point[1],point[0])-yaw;turn=math.atan2(math.sin(turn),math.cos(turn))
    return dict(turn_radians=turn,observed_id=identity,memory_age=age,waypoint_relative=point,path_length=distance)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-model',required=True,type=Path);ap.add_argument('--train',required=True,type=Path,nargs='+')
    ap.add_argument('--diagnostic',required=True,type=Path,nargs='+');ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--diagnostic-exclude-seeds',type=int,nargs='*',default=[],
                    help='Explicit overlapping diagnostic worlds to exclude before labels are read')
    a=ap.parse_args();assert not a.out.exists();a.out.mkdir(parents=True);model_hash=sha(a.source_model)
    episodes=[];pins={str(a.source_model.resolve()):model_hash};worlds={s:set() for s in ('train','diagnostic')}
    for split,roots in [('train',a.train),('diagnostic',a.diagnostic)]:
        for root in roots:
            root=root.resolve();spec=read(root/'spec.json');score=read(root/'acceptance.json')
            assert score['state']=='passed'
            for name in ('spec.json','acceptance.json','pool/report.json'):pins[str(root/name)]=sha(root/name)
            for v in spec['variants']:
                if v['model_sha256']!=model_hash:continue
                if split=='diagnostic' and v['context']['seed'] in a.diagnostic_exclude_seeds:continue
                assert v['context']['instance']['split']=='train'
                d=root/f'case-{v["index"]:02d}';rows,context,token,meta=admit(d/'cuda',model_hash,[])
                path=next((d/'native').glob('replay-*/steps.jsonl'));native={s['observation']['identity']['frame']:s for s in load_steps(path)}
                indices=[];targets=[];labels=[]
                for i,r in enumerate(rows):
                    step=native[r['frame']];o=step['observation'];label=heading(o)
                    if label is None:continue
                    assert step['owner']=='provider' and step['native_step'] and step['server_execution']['matched'] and step['server_execution']['window_exclusive']
                    assert o['identity']['life']==1
                    indices.append(i);targets.append(label['turn_radians']);labels.append(dict(frame=r['frame'],**label))
                if not indices:continue
                key=(v['context']['map'],v['context']['seed']);assert key not in worlds[split]
                worlds[split].add(key)
                dest=a.out/f'episode-{len(episodes):02d}';dest.mkdir();pack=dest/'heading.npz'
                np.savez_compressed(pack,indices=np.asarray(indices,dtype=np.int64),turn=np.asarray(targets,dtype=np.float64))
                save(dest/'labels.json',labels)
                episodes.append(dict(map=key[0],seed=key[1],split=split,rows=len(indices),pack=str(pack.resolve()),pack_sha256=sha(pack),
                    labels=str((dest/'labels.json').resolve()),labels_sha256=sha(dest/'labels.json'),source_cuda=str((d/'cuda').resolve()),
                    source_model_sha256=model_hash,native_rollout_sha256=token))
                for p in (path,d/'cuda/report.json',d/'cuda/rollout.jsonl',d/'cuda/sequence.jsonl',d/'cuda/cuda-verification.json'):pins[str(p.resolve())]=sha(p)
    assert worlds['train'] and worlds['diagnostic'] and not(worlds['train']&worlds['diagnostic'])
    assert all(sha(p)==h for p,h in pins.items())
    save(a.out/'report.json',dict(version=LABEL,state='passed',episodes=episodes,inputs=pins,
        counts={s:dict(rows=sum(e['rows'] for e in episodes if e['split']==s),worlds=len(worlds[s])) for s in worlds},
        diagnostic_excluded_seeds=a.diagnostic_exclude_seeds,
        ppo_eligible=False,executed_demonstration=False,neural_execution=False,
        selection='All matching fixed-policy TRAIN contexts, no outcome filter; nearest path within freshest matched observed memories, waypoint norm >=8.',
        scope='Geometric auxiliary target from current observation only; not an optimal action, executed teacher, future entity location or proven traversability.'))
    print({s:dict(rows=sum(e['rows'] for e in episodes if e['split']==s),worlds=len(worlds[s])) for s in worlds},flush=True)


if __name__=='__main__':main()

"""Observed command/physics diagnostics during route availability; no NN or teacher."""
import argparse
import math
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from audit_combat_contact_loss_probe import load_steps


def command_yaw(observation, applied):
    assert applied['identity']==observation['identity']
    short=int(observation['view_angles'][1])+round(applied['yaw_delta_degrees']*65536/360)
    return (short+32768)%65536-32768


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--screen',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args();assert not a.out.exists();root=a.screen.resolve();spec=read(root/'spec.json');items=[];pins={}
    for v in spec['variants']:
        if v['label']!='zero':continue
        path=next((root/f'case-{v["index"]:02d}/native').glob('replay-*/steps.jsonl'))
        rollout=root/f'case-{v["index"]:02d}/cuda/rollout.jsonl'
        admitted={r['frame'] for r in load_steps(rollout)}
        steps=load_steps(path);count=0;alignments=[];moving=away=stationary=ducked=0;longest=run=0;previous=None;distances=[];yaw_checks=0
        for s in steps:
            o,n=s['observation'],s['next_observation'];hints=(o.get('remembered_static_routes') or {}).get('hints',[])
            points=[h['next_static_waypoint_relative'] for h in hints if h['static_geometry_known'] and h['static_route_found']]
            if o['identity']['frame'] not in admitted or s['owner']!='provider' or n is None or o['health']<=0 or n['health']<=0 or o['enemies'] or not points:continue
            dispatch=s.get('server_execution') or {}
            assert dispatch.get('matched') and dispatch.get('window_exclusive')
            frame=o['identity']['frame'];count+=1;ducked+=bool(o.get('ducked'))
            dx,dy=(n['position'][j]-o['position'][j] for j in (0,1));distance=math.hypot(dx,dy);distances.append(distance)
            cmd=s['sent_command'];fwd,side=cmd['Forward'],cmd['Side'];magnitude=math.hypot(fwd,side)
            applied=s['applied_action'];effective=command_yaw(o,applied)
            assert effective==n['view_angles'][1], 'Native next view differs from effective command yaw'
            yaw_checks+=1
            assert abs(applied['forward']-fwd/400)<1e-6 and abs(applied['side']-side/400)<1e-6
            yaw=effective*2*math.pi/65536
            wx,wy=fwd*math.cos(yaw)+side*math.sin(yaw),fwd*math.sin(yaw)-side*math.cos(yaw)
            candidates=[(wx*p[0]+wy*p[1])/(magnitude*math.hypot(p[0],p[1])) for p in points if magnitude>1 and math.hypot(p[0],p[1])>1]
            if candidates:
                moving+=1;best=max(candidates);alignments.append(best);away+=best<-.1
            stopped=magnitude>=100 and distance<1
            stationary+=stopped
            run=run+1 if stopped and previous is not None and frame==previous+1 else int(stopped)
            longest=max(longest,run);previous=frame
        items.append(dict(map=v['context']['map'],seed=v['context']['seed'],loadout=v['context']['loadout'],route_rows=count,
            movement_rows=moving,commands_away_from_all_available_waypoints=away,stationary_with_command_rows=stationary,
            longest_stationary_command_run=longest,crouched_rows=ducked,
            mean_best_waypoint_alignment=sum(alignments)/len(alignments) if alignments else None,
            mean_native_horizontal_displacement=sum(distances)/len(distances) if distances else None,native_effective_yaw_checks=yaw_checks))
        pins[str(path.resolve())]=sha(path)
        pins[str(rollout.resolve())]=sha(rollout)
    save(a.out,dict(version='combat_route_motion_applied_view_task_rows_v3',state='passed',results=items,inputs=pins,training_performed=False,
        scope='Observed native displacement and matched forward/right-side commands. Alignment uses best of all available static waypoints; it is a diagnostic, not proof of optimal action, route traversability or target life state.'))
    for item in items:
        if item['route_rows']:print(item,flush=True)


if __name__=='__main__':main()

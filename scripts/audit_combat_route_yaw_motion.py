"""Native camera and world-command diagnostics; no neural computation."""
import argparse
import math
from pathlib import Path
from prepare_combat_route_heading import heading
from audit_combat_route_motion import command_yaw
from audit_combat_contact_loss_probe import load_steps
from evaluate_registered_combat_models import read,save,sha


def summarize(records):
    rows=len(records)
    moving=[r for r in records if r['alignment'] is not None]
    return dict(rows=rows,mean_absolute_yaw_degrees=sum(abs(r['yaw']) for r in records)/rows if rows else None,
        net_camera_turns=sum(r['yaw'] for r in records)/360,absolute_camera_turns=sum(abs(r['yaw']) for r in records)/360,
        large_yaw_frames=sum(abs(r['yaw'])>=30 for r in records),
        mean_waypoint_heading_error_degrees=sum(r['heading_error'] for r in records)/rows if rows else None,
        movement_frames=len(moving),mean_waypoint_command_alignment=sum(r['alignment'] for r in moving)/len(moving) if moving else None,
        away_movement_frames=sum(r['alignment']<-.1 for r in moving),
        stopped_command_frames=sum(r['stopped'] for r in records),crouched_frames=sum(r['crouched'] for r in records))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--screen',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args();assert not a.out.exists();root=a.screen.resolve()
    spec=read(root/'spec.json');quality=read(root/'acceptance.json')
    assert quality['state']=='passed' and quality['numerical_verification_complete']
    pins={str(root/'spec.json'):sha(root/'spec.json'),str(root/'acceptance.json'):sha(root/'acceptance.json')}
    cases=[];groups={'source':[],'candidate':[]}
    for v in spec['variants']:
        if v['label'] not in ('source','zero'):continue
        d=root/f'case-{v["index"]:02d}'
        meta=read(d/'cuda/report.json');assert meta['model_sha256']==v['model_sha256']
        rp=d/'cuda/rollout.jsonl';assert sha(rp)==meta['rollout_sha256']
        admitted={r['frame'] for r in load_steps(rp)}
        path=next((d/'native').glob('replay-*/steps.jsonl'));records=[]
        for s in load_steps(path):
            o=s['observation'];frame=o['identity']['frame']
            if frame<100 or frame not in admitted or s['owner']!='provider' or o['health']<=0:continue
            h=heading(o)
            if h is None:continue
            assert s['native_step'] and s['server_execution']['matched'] and s['server_execution']['window_exclusive']
            actual_yaw=command_yaw(o,s['applied_action'])
            if s['next_observation'] is not None:assert actual_yaw==s['next_observation']['view_angles'][1]
            angle=actual_yaw*2*math.pi/65536;fwd,side=s['sent_command']['Forward'],s['sent_command']['Side']
            wx,wy=fwd*math.cos(angle)+side*math.sin(angle),fwd*math.sin(angle)-side*math.cos(angle)
            point=h['waypoint_relative'];norm=math.hypot(fwd,side);alignment=None
            if norm>=100:alignment=(wx*point[0]+wy*point[1])/(norm*math.hypot(*point[:2]))
            yaw=s['applied_action']['yaw_delta_degrees'];error=yaw*math.pi/180-h['turn_radians']
            records.append(dict(frame=frame,yaw=yaw,heading_error=abs(math.atan2(math.sin(error),math.cos(error)))*180/math.pi,
                alignment=alignment,stopped=norm<1,crouched=bool(o.get('ducked'))))
        role='candidate' if v['label']=='zero' else 'source';groups[role].extend(records)
        cases.append(dict(map=v['context']['map'],seed=v['context']['seed'],role=role,**summarize(records)))
        pins[str(path.resolve())]=sha(path);pins[str(rp.resolve())]=sha(rp)
    totals={r['controller']:r for r in quality['totals']}
    advance=(totals['candidate']['wins']>=totals['source']['wins']+1
        and totals['candidate']['wins']>=totals['rules']['wins']
        and totals['candidate']['ownership_failures']<=totals['source']['ownership_failures'])
    save(a.out,dict(version='combat_route_native_yaw_world_command_audit_v1',state='passed',cases=cases,
        totals={k:summarize(v) for k,v in groups.items()},advance_confirmation=advance,promotion=False,inputs=pins,
        scope='Executed route-context camera/command diagnostics on different closed-loop trajectories, not same-state causal accuracy, optimal navigation or full campaign acceptance.'))
    print({k:summarize(v) for k,v in groups.items()},flush=True)


if __name__=='__main__':main()

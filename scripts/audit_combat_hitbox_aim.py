"""Offline native ray/AABB audit; no policy execution or fitting."""
import argparse
import collections
import math
import pathlib
import json
from process_combat_architecture_pool import read, save, sha
from report_combat_machinegun_hits import scan, vector
from report_combat_machinegun_aim import measure, angular
from export_combat_aim_feedback import poses


def intersects(start, direction, lower, upper, maximum=math.inf, epsilon=.05):
    near,far=0.,maximum
    for s,d,lo,hi in zip(start,direction,lower,upper):
        if abs(d)<1e-12:
            if s<lo-epsilon or s>hi+epsilon:return False
            continue
        a,b=(lo-epsilon-s)/d,(hi+epsilon-s)/d
        near=max(near,min(a,b));far=min(far,max(a,b))
        if near>far:return False
    return far>=0


def ballistic_basis(direction):
    # game/g_utils.c vectoangles truncates atan2 degrees to int BEFORE
    # fire_lead calls AngleVectors. f['aim'] is the pre-quantization input.
    x,y,z=direction;assert x*x+y*y+z*z>0
    if x==0 and y==0:
        yaw=0.;pitch=-(90. if z>0 else 270.)
    else:
        yaw=math.trunc(math.degrees(math.atan2(y,x))) if x else (90 if y>0 else -90)
        if yaw<0:yaw+=360
        elevation=math.trunc(math.degrees(math.atan2(z,math.hypot(x,y))))
        if elevation<0:elevation+=360
        pitch=-elevation
    p,w=math.radians(pitch),math.radians(yaw)
    cp,sp,cy,sy=math.cos(p),math.sin(p),math.cos(w),math.sin(w)
    return [cp*cy,cp*sy,-sp],[sy,-cy,0.],[sp*cy,sp*sy,cp]


def spread_ray(direction, spread):
    forward,right,up=ballistic_basis(direction)
    return [8192*f+spread[0]*r+spread[1]*u for f,r,u in zip(forward,right,up)]


def main():
    ap=argparse.ArgumentParser()
    for name in ('evaluation','spec','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    args=ap.parse_args();assert not args.out.exists();pins={}
    def pin(path,expected=None):
        path=pathlib.Path(path);digest=sha(path);assert expected is None or digest==expected
        pins[str(path)]=digest;return path
    for name in ('native_projectile_window.py','report_combat_machinegun_hits.py','report_combat_machinegun_aim.py','export_combat_aim_feedback.py'):
        pin(pathlib.Path(__file__).with_name(name))
    root=args.evaluation.resolve();seal=read(pin(root/'complete.json'));assert seal['state']=='complete'
    spec=read(pin(args.spec,seal['spec_sha256']));pool=read(pin(root/'pool/report.json',seal['pool_sha256']))
    assert pool['state']=='complete' and pool['source_unchanged']
    results={}
    for index,variant in enumerate(spec['variants']):
        if variant['mode']!='learned':continue
        execution=read(pin(root/('execution-'+variant['label']+'.json')))
        assert execution['state']=='complete' and execution['pool_sha256']==seal['pool_sha256'] and execution['selected_plan_index']==index
        bindings={c['job']:c for c in execution['cases']}
        counts=collections.Counter();height_fractions=[];quantization_errors=[];max_direction_error=0.
        vertical_strata=collections.defaultdict(collections.Counter)
        for job in pool['jobs']:
            if job['plan_index']!=index:continue
            binding=bindings[job['job']]
            capture=read(pin(pathlib.Path(job['root'])/'report.json',binding['capture_sha256']))
            assert capture['capture_complete'] and capture['provenance_valid'] and len(capture['results'])==1
            log=pin(pathlib.Path(capture['results'][0]['root'])/'server.log',binding['server_log_sha256'])
            steps=pin(root/('processing-'+variant['label'])/f"case-{job['job']:04d}/native/replay-0/steps.jsonl")
            windows={}
            with steps.open(encoding='utf-8-sig') as stream:
                for step in map(json.loads,stream):
                    n=step.get('native_step');o=step['observation'];dispatch=step.get('server_execution',{})
                    if n and step['owner']=='provider' and o['identity']['life']==1 and o['health']>0 and o['identity']['frame']>=binding['network_release_frame'] and dispatch.get('matched') and dispatch.get('window_exclusive') and not dispatch.get('recovery_commands'):
                        assert n['spawncount']==binding['release_generation']
                        key=(n['spawncount'],n['begin_frame'],n['actor'],n['sequence']);assert key not in windows;windows[key]=step
            with log.open(encoding='utf-8-sig') as stream:shots=scan(stream,set(windows))
            with log.open(encoding='utf-8-sig') as stream:truth=poses(stream,shots)
            for shot in shots:
                if not shot['eligible']:continue
                counts['eligible_shots']+=1;step=windows[shot['window']];observed=measure(shot,step)
                if observed['status']!='measured':counts[observed['status']]+=1;continue
                f=shot['fields'];pose=truth.get((shot['generation'],int(f['shot'])),{}).get(observed['entity'])
                if pose is None:counts['selected_target_without_living_pose']+=1;continue
                if int(f['muzzle_blocked']) or int(f['water']):counts['nonstraight_water_or_blocked_muzzle']+=1;continue
                counts['measured']+=1
                lower=[a+b for a,b in zip(pose['origin'],pose['mins'])];upper=[a+b for a,b in zip(pose['origin'],pose['maxs'])]
                start=vector(f['start']);aim=vector(f['aim']);impact=vector(f['impact']);spread=vector(f['spread'],2)
                actual=[a-b for a,b in zip(impact,start)];reconstructed=spread_ray(aim,spread)
                if math.dist(start,impact)>2:
                    error=angular(actual,reconstructed);max_direction_error=max(max_direction_error,error)
                    assert error<.1,'Native spread ray reconstruction mismatch'
                    counts['reconstruction_verified']+=1
                else:counts['reconstruction_short_segment']+=1
                effective=ballistic_basis(aim)[0];quantization_errors.append(angular(aim,effective))
                before_quantization=intersects(start,aim,lower,upper)
                nominal_hit=intersects(start,effective,lower,upper)
                counts['prequantization_ray_intersects']+=before_quantization
                counts['prequantization_hit_effective_miss']+=before_quantization and not nominal_hit
                counts['prequantization_miss_effective_hit']+=not before_quantization and nominal_hit
                spread_hit=intersects(start,reconstructed,lower,upper,maximum=1.)
                segment_hit=intersects(start,actual,lower,upper,maximum=1.)
                contact=int(f['target'])==pose['entity'] and int(f['damageable']) and int(f['health_before'])>0
                if contact:assert segment_hit,'Native selected contact outside shot-time AABB'
                counts['nominal_ray_intersects']+=nominal_hit
                counts['spread_ray_intersects_without_obstruction_check']+=spread_hit
                counts['actual_segment_intersects']+=segment_hit
                counts['selected_native_contact']+=bool(contact)
                counts['selected_health_damage']+=observed['selected_target_damage']
                counts['nominal_hit_spread_miss']+=nominal_hit and not spread_hit
                counts['nominal_miss_spread_hit']+=not nominal_hit and spread_hit
                counts['spread_hit_but_segment_short_or_other_contact']+=spread_hit and not segment_hit
                if contact:height_fractions.append((impact[2]-lower[2])/(upper[2]-lower[2]))
                center=[(a+b)/2 for a,b in zip(lower,upper)]
                above=below=False
                if not nominal_hit:
                    horizontal=sum(x*x for x in effective[:2])
                    t=sum((c-s)*d for c,s,d in zip(center[:2],start[:2],effective[:2]))/horizontal if horizontal>1e-12 else -1
                    if t>0:
                        z=start[2]+t*effective[2]
                        above=z>upper[2];below=z<lower[2]
                        counts['nominal_miss_above_at_center_range']+=above
                        counts['nominal_miss_below_at_center_range']+=below
                o=step['observation'];vertical=step['applied_action']['vertical']
                group=vertical+('_ground' if o['on_ground'] else '_air')
                transition=('crouch_from_standing' if vertical=='crouch' and not o['ducked'] else
                    'release_from_ducked' if vertical=='release' and o['ducked'] else None)
                for key in [group]+([transition] if transition else []):
                    bucket=vertical_strata[key]
                    bucket.update(shots=1,selected_native_contact=int(bool(contact)),nominal_intersects=int(nominal_hit),
                        nominal_above=int(above),nominal_below=int(below),
                        muzzle_minus_observed_origin_z=start[2]-o['position'][2],
                        native_vertical_speed=vector(f['velocity'])[2],observed_vertical_speed=o['velocity'][2])
                # Fixed point choices independent of this shot's RNG. This is
                # bbox-only matched-spread geometry, never a BSP visibility test.
                bottom,top=pose['mins'][2],pose['maxs'][2]
                height=(top+bottom)/2 if top-bottom<16 else min(max(22,bottom+8),top-8)
                upper_point=list(pose['origin']);upper_point[2]+=height
                head_ray=spread_ray([a-b for a,b in zip(upper_point,start)],spread)
                center_ray=spread_ray([a-b for a,b in zip(center,start)],spread)
                head_ok=intersects(start,head_ray,lower,upper,maximum=1.)
                center_ok=intersects(start,center_ray,lower,upper,maximum=1.)
                counts['ideal_upper_point_bbox_only']+=head_ok
                counts['ideal_center_point_bbox_only']+=center_ok
                counts['ideal_center_only_bbox']+=center_ok and not head_ok
                counts['ideal_upper_only_bbox']+=head_ok and not center_ok
        assert counts['measured'] and counts['reconstruction_verified']
        results[variant['label']]=dict(counts=dict(counts),
            mean_contact_height_fraction=sum(height_fractions)/len(height_fractions) if height_fractions else None,
            mean_native_angle_quantization_degrees=sum(quantization_errors)/len(quantization_errors),
            max_native_angle_quantization_degrees=max(quantization_errors),
            vertical_strata={key:dict(counts=dict(value),mean_muzzle_minus_observed_origin_z=value['muzzle_minus_observed_origin_z']/value['shots'],
                mean_native_vertical_speed=value['native_vertical_speed']/value['shots'],
                mean_observed_vertical_speed=value['observed_vertical_speed']/value['shots']) for key,value in vertical_strata.items()},
            spread_reconstruction_max_error_degrees=max_direction_error)
    for path,digest in pins.items():assert sha(path)==digest
    save(args.out,dict(state='complete',version='native_hitbox_aim_audit_v1',variants=results,
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),training_performed=False,promotion=False,
        scope='Actual native shot-time selected living target AABB and MG ray, including fire_lead integer-degree vectoangles truncation. Native contact validates finite-segment geometry; reconstruction checked against logged impact. Fixed upper-vs-center ideal point comparisons reuse native spread samples without cancelling them, ignore BSP and other occluders, and assume perfect fresh recoil compensation. Not live hit probability, an optimal teacher, or train data. No nearest-target substitution or rules comparison without explicit target identity.'))
    print(json.dumps(results))


if __name__=='__main__':main()

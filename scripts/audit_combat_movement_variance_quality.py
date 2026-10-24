"""Native movement and MG impact audit for the fixed 24-condition variance confirmation."""
import argparse
import collections
import json
import math
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from report_combat_machinegun_hits import scan
from report_registered_machinegun_outcomes import outcome
from native_combat_release import release_boundary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();root=a.evaluation.resolve();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path);h=pins[key] if key in pins else sha(path)
        assert expected is None or h==expected,str(path);pins[key]=h;return path
    pin(__file__);closed=read(pin(root/'complete.json'));assert closed['native_cases']==72 and closed['cuda_cases']==48
    # The complete receipt binds the caller-supplied frozen spec; fixture proof
    # binds labels and plan indices without guessing an external filename.
    fixture=read(pin(root/'fixture-proof.json'));assert fixture['spec_sha256']==closed['spec_sha256']
    pool=read(pin(root/'pool/report.json',closed['pool_sha256']));assert pool['source_unchanged']
    labels={i:Path(p['path']).stem.removesuffix('-plan') for i,p in enumerate(fixture['plans'])}
    assert set(labels.values())=={'source','noise-only','rules'}
    variants={};cases=[]
    for label in ('source','noise-only'):
        meta=read(pin(root/f'processing-{label}/report.json'));assert meta['device']=='cuda' and not meta['training_performed']
        totals=collections.Counter();sum_squared=0.;angle_count=0
        for job in sorted((j for j in pool['jobs'] if labels[j['plan_index']]==label),key=lambda j:j['job']):
            folder=root/f'processing-{label}'/f"case-{job['job']:04d}";cm=read(pin(folder/'cuda/report.json'))
            proof=read(pin(folder/'cuda/cuda-verification.json',cm['cuda_verification_sha256']));assert proof['state']=='passed' and proof['device']=='cuda'
            capture=read(pin(Path(job['root'])/'report.json'));r=capture['results'][0]
            assert capture['capture_complete'] and capture['provenance_valid'] and r['dispatch_valid']
            manifest=read(pin(Path(job['root'])/'manifest.json'));log=pin(Path(r['root'])/'server.log')
            with log.open(encoding='utf-8-sig') as stream:_,release,_=release_boundary(stream,manifest,job['seed'])
            sp=next((folder/'native').glob('replay-*/steps.jsonl'))
            all_steps=list(map(json.loads,pin(sp,cm['source_sha256'][str(sp.resolve())]).open(encoding='utf-8-sig')))
            ep=sp.with_name('server_outcomes.jsonl');effects={e['step']:e for e in map(json.loads,pin(ep,cm['source_sha256'][str(ep.resolve())]).open(encoding='utf-8-sig'))}
            stop=r['goal_stop']['kill_frame'] if r['goal_stop'] else r['first_life']['end_frame']
            steps=[s for s in all_steps if s['owner']=='provider' and s.get('sample') and s.get('native_step') and s['observation']['identity']['life']==1 and s['observation']['health']>0 and release<=s['observation']['identity']['frame']<stop]
            windows={tuple(s['native_step'][k] for k in ('spawncount','begin_frame','actor','sequence')):s for s in steps if (s.get('server_execution') or {}).get('matched') and (s.get('server_execution') or {}).get('window_exclusive') and not (s.get('server_execution') or {}).get('recovery_commands')}
            assert len(windows)==len(steps)
            with log.open(encoding='utf-8-sig') as stream:shots=scan(stream,set(windows))
            byframe=collections.defaultdict(list)
            for shot in shots:
                if shot['eligible']:byframe[windows[shot['window']]['observation']['identity']['frame']].append(shot)
            counts=collections.Counter(cases=1,alive_goals=int(bool(r['goal_stop'] and not r['death_stop'] and r['goal_stop']['health']>0)),deaths=int(bool(r['death_stop'])))
            counts['received']=r['first_life']['damage']['received_health_damage'];health={};bbox={};sq=0.;n_angles=0
            included={s['index'] for s in steps}
            for s in all_steps:
                o=s['observation']
                if o['identity']['life']!=1:continue
                for e in o['enemies']:bbox[e['id']]=(e['class'],e.get('observed_solid',0))
                if s['index'] in included:
                    f=o['identity']['frame'];command=s['applied_action'];memory=sorted(o.get('remembered_threats') or [],key=lambda t:(t['age_frames'],t['id']))
                    active=bool(not o['enemies'] and memory and memory[0]['age_frames']<=10)
                    counts['provider_frames']+=1;counts['memory_active_frames']+=active
                    request=s['action'];rn=math.hypot(request['forward'],request['side']);an=math.hypot(command['forward'],command['side'])
                    visible_ground=bool(o['enemies'] and o['on_ground']);speed=math.hypot(*o['velocity'][:2])
                    counts['grounded_visible_frames']+=visible_ground
                    counts['standing_grounded_visible_frames']+=bool(visible_ground and speed<40)
                    counts['large_requested_movement_frames']+=rn>=.25
                    counts['large_applied_movement_frames']+=an>=.25
                    counts['request_reduced_near_zero_frames']+=bool(rn>=.25 and an<.1)
                    nxt=s.get('next_observation')
                    if nxt:
                        delta=math.hypot(*(b-a for a,b in zip(o['position'][:2],nxt['position'][:2])))
                        counts['large_applied_near_zero_travel_frames']+=bool(an>=.25 and delta<1)

                    counts['large_yaw_frames']+=abs(command['yaw_delta_degrees'])>=30
                    counts['attack_without_target_frames']+=bool(command['attack'] and not s['action'].get('target_entity',0))
                    counts['memory_known_dead_reference_frames']+=bool(active and health.get((memory[0]['id'],memory[0]['class']),1)<=0)
                    if active:
                        counts['active_large_yaw_frames']+=abs(command['yaw_delta_degrees'])>=30
                        t=memory[0];solid=bbox.get(t['id'],('',0))[1];bottom=-((solid>>5)&31)*8;top=((solid>>10)&63)*8-32
                        if solid not in (0,31) and solid&31 and top>bottom:
                            height=(top+bottom)/2 if top-bottom<16 else min(max(22,bottom+8),top-8)
                            x,y,z=t['last_observed_relative'];z+=height-(-2 if o['ducked'] else 22)
                            if math.hypot(x,y)>1:
                                desired=(-math.degrees(math.atan2(z,math.hypot(x,y))),math.degrees(math.atan2(y,x)))
                                actual=[o['view_angles'][i]*360/65536+command[k] for i,k in enumerate(('pitch_delta_degrees','yaw_delta_degrees'))]
                                errors=[(a-b+180)%360-180 for a,b in zip(actual,desired)];sq+=sum(e*e for e in errors)/2;n_angles+=1
                    for shot in byframe[f]:
                        kind=outcome(shot);counts['mg_shots']+=1;counts['mg_'+kind]+=1
                        if active:counts['active_mg_shots']+=1;counts['active_mg_'+kind]+=1
                effect=effects.get(s['index'])
                if effect and effect['available']:
                    for e in effect['events']:
                        if e['target_class'].startswith('monster_'):health[e['target'],e['target_class']]=e['health_after']
            cases.append(dict(label=label,seed=job['seed'],map=r['generated_fixture']['map'],counts=dict(counts),active_direction_rmse=math.sqrt(sq/n_angles) if n_angles else None))
            totals.update(counts);sum_squared+=sq;angle_count+=n_angles
        variants[label]=dict(counts=dict(totals),active_direction_rmse=math.sqrt(sum_squared/angle_count) if angle_count else None,
            mg_live_damage_fraction=totals['mg_live_monster_damage']/totals['mg_shots'] if totals['mg_shots'] else None)
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True);report=dict(state='complete',version='combat_movement_variance_native_quality_audit_v1',variants=variants,cases=cases,inputs=pins,
        neural_execution=False,new_games=False,training_performed=False,promotion=False,
        scope='Native impacts and actual applied gaze, not offline mean predictions. Unequal closed-loop states/frame counts are descriptive. Past native HP only audits dead reference labels, never policy input. Not proof of optimal direction or general superiority.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(variants)


if __name__=='__main__':main()

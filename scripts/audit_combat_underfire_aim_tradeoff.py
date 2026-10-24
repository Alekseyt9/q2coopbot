"""Describe shot/target/motion mediation in already sealed movement comparisons."""
import argparse
import collections
import json
import math
from pathlib import Path
import statistics
from evaluate_registered_combat_models import read,save,sha
from report_combat_machinegun_hits import scan
from report_registered_machinegun_outcomes import outcome
from report_combat_machinegun_aim import measure
from export_combat_aim_feedback import poses,label as aim_label


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();root=a.collection.resolve();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path);h=pins[key] if key in pins else sha(path)
        assert expected is None or expected==h,str(path);pins[key]=h;return path
    pin(__file__)
    for name in ('report_combat_machinegun_hits.py','report_registered_machinegun_outcomes.py','report_combat_machinegun_aim.py','export_combat_aim_feedback.py','native_projectile_window.py'):
        pin(Path(__file__).with_name(name))
    seal=read(pin(root/'complete.json'));assert seal['version']=='combat_native_continuation_pairs_v1'
    spec=read(pin(root/'spec.json',seal['spec_sha256']));p=read(pin(spec['protocol'],spec['protocol_sha256']))
    assert p['study']=='combat_underfire_constrained_confirmation_v1'
    ds=read(pin(root/'dataset/complete.json'));report=read(pin(root/'dataset/report.json',ds['report_sha256']))
    assert report['gate_passed']
    pool=read(pin(root/'pool/report.json',seal['pool_sha256']));assert pool['source_unchanged']
    jobs={j['plan_index']:j for j in pool['jobs']};cases=[];shot_rows=[];frame_rows=[]
    for unit in spec['contexts']:
        c=unit['context'];frame=c['frame'];proof_seal=read(pin(root/'proofs'/unit['unit_id']/'complete.json'))
        proof=read(pin(root/'proofs'/unit['unit_id']/'report.json',proof_seal['report_sha256']));assert proof['gate_passed']
        for variant in unit['variants']:
            if variant['label'] in ('ordinary_reference','ordinary_no_fork'):continue
            i=variant['plan_index'];folder=root/f'case-{i:02d}';meta=read(pin(folder/'cuda/report.json'))
            assert meta['numerical_verification']=='cuda_verified_v1'
            sp=next((folder/'native').glob('replay-*/steps.jsonl'))
            steps=list(map(json.loads,pin(sp,meta['source_sha256'][str(sp.resolve())]).open(encoding='utf-8-sig')))
            relevant=[s for s in steps if s['owner']=='provider' and s.get('native_step') and s['observation']['identity']['life']==1 and frame<=s['observation']['identity']['frame']<frame+10]
            assert len(relevant)==10
            windows={tuple(s['native_step'][k] for k in ('spawncount','begin_frame','actor','sequence')):s for s in relevant}
            ep=sp.with_name('server_outcomes.jsonl');effects={e['step']:e for e in map(json.loads,pin(ep,meta['source_sha256'][str(ep.resolve())]).open(encoding='utf-8-sig'))}
            capture=read(pin(Path(jobs[i]['root'])/'report.json'));log=pin(Path(capture['results'][0]['root'])/'server.log')
            with log.open(encoding='utf-8-sig') as stream:shots=scan(stream,set(windows))
            with log.open(encoding='utf-8-sig') as stream:shot_poses=poses(stream,shots)
            summary=collections.Counter();weapons=collections.Counter();dealt_mod=collections.Counter();native_errors=[];observed_errors=[];previous=None
            common=dict(map=c['instance']['map'],world_seed=c['seed'],stream_seed=proof['fork']['seed'],label=variant['label'],case_index=i)
            for s in relevant:
                o=s['observation'];command=s['applied_action'];target=s['action'].get('target_entity',0)
                enemy=next((e for e in o['enemies'] if e['id']==target and e.get('observed_track',0)==s['action'].get('target_track',0)),None)
                summary['attack_decisions']+=bool(command['attack']);summary['target_unavailable_frames']+=enemy is None
                summary['target_blocked_frames']+=bool(enemy is not None and not enemy.get('clear_shot'))
                if previous is not None:summary['target_switches']+=previous!=target
                previous=target;weapons[o['weapon']]+=1
                next_o=s.get('next_observation');distance=math.dist(o['position'],next_o['position']) if next_o else None
                commanded=math.hypot(command['forward'],command['side'])>.2
                stalled=commanded and distance is not None and distance<1
                summary['commanded_motion_stalled_frames']+=stalled
                effect=effects[s['index']];assert effect['available']
                for event in effect['events']:
                    if event['attacker']==o['identity']['actor'] and event['target_class'].startswith('monster_'):
                        dealt_mod[str(event['mod'])]+=min(max(event['health_before'],0),event['take'])
                frame_rows.append(dict(**common,frame=o['identity']['frame'],target=target,target_class=enemy['class'] if enemy else None,
                    clear_shot=enemy.get('clear_shot') if enemy else None,attack=command['attack'],vertical=command['vertical'],
                    position=o['position'],observed_velocity=o['velocity'],displacement=distance,commanded_motion_stalled=stalled,
                    dealt=effect['monster_health_damage'],received=effect['received_health_damage']))
            for shot in shots:
                if not shot['eligible']:continue
                s=windows[shot['window']];result=outcome(shot);measurement=measure(shot,s)
                summary['mg_shots']+=1;summary['mg_'+result]+=1;summary['muzzle_blocked_shots']+=int(shot['fields']['muzzle_blocked'])
                item=dict(**common,frame=s['observation']['identity']['frame'],outcome=result,measurement=measurement)
                if measurement['status']=='measured':
                    observed_errors.append(measurement['post_recoil_degrees'])
                    pose=shot_poses.get((shot['generation'],int(shot['fields']['shot'])),{}).get(measurement['entity'])
                    if pose:
                        native=aim_label(shot,pose,s);native_errors.append(native['native_bodypoint_error_degrees'])
                        item['native_bodypoint_error_degrees']=native['native_bodypoint_error_degrees']
                shot_rows.append(item)
            cases.append(dict(**common,counts=dict(summary),observed_weapons=dict(weapons),dealt_by_mod=dict(dealt_mod),
                native_bodypoint_error_mean=statistics.mean(native_errors) if native_errors else None,
                observed_bodypoint_error_mean=statistics.mean(observed_errors) if observed_errors else None,
                dealt=sum(effects[s['index']]['monster_health_damage'] for s in relevant),
                received=sum(effects[s['index']]['received_health_damage'] for s in relevant)))
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True);save(a.out/'frames.json',frame_rows);save(a.out/'shots.json',shot_rows)
    report=dict(state='complete',version='combat_underfire_aim_tradeoff_audit_v1',cases=cases,inputs=pins,
        frames_sha256=sha(a.out/'frames.json'),shots_sha256=sha(a.out/'shots.json'),neural_execution=False,new_games=False,
        training_performed=False,promotion=False,
        scope='Posthoc mediation description within accepted H10movement comparisons. Shot-time native target poses are offline labels only. Misses, stochastic spread and geometry are not collapsed into a causal aim-error proof; no model selection or fitting.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(cases=len(cases),ware2=[c for c in cases if c['map']=='ware2']))


if __name__=='__main__':main()

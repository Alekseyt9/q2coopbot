"""Scalar admission/outcomes for the frozen32case ready-weapon/rules TRAIN screen."""
import argparse
import json
import math
from pathlib import Path
import re
from evaluate_registered_combat_models import read, save, sha
from combat_equipped_weapon_features import equipped_weapon_index
from combat_native_shot_timing import physical_shots


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--screen',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();root=a.screen.resolve();assert not a.out.exists()
    seal,spec,pool=[read(root/p) for p in ('complete.json','spec.json','pool/report.json')]
    assert seal['native_cases']==32 and seal['cuda_cases']==16 and len(spec['variants'])==32
    assert sha(root/'spec.json')==seal['spec_sha256'] and sha(root/'pool/report.json')==seal['pool_sha256']
    assert sha(root/'cuda-batch/report.json')==seal['cuda_receipt_sha256'] and pool['source_unchanged']
    jobs={j['plan_index']:j for j in pool['jobs']};loaded=[]
    for v in spec['variants']:
        i=v['plan_index'];capture=Path(jobs[i]['root']);report=read(capture/'report.json');mf=read(capture/'manifest.json');r=report['results'][0]
        assert report['capture_complete'] and report['provenance_valid'] and r['capture_valid'] and r['dispatch_valid'] and r['seed_confirmed']
        assert r['seed']==v['seed'] and r['generated_fixture']==v['instance'] and r['generated_start']['confirmed']
        assert mf['observation_timing_contract']=='frozen_world_v1' and mf['loadout']==v['label'] and mf['provider']==v['mode']
        assert r['harness_accepted'] and r['worker_exit_code']==0 and r['harness_reason']=='generated_native_capture'
        if v['mode']=='learned':
            directory=root/f'case-{i:02d}';meta=read(directory/'cuda/report.json');proof=read(directory/'cuda/cuda-verification.json')
            assert proof['state']=='passed' and proof['device']=='cuda'
            assert sha(directory/'cuda/cuda-verification.json')==meta['cuda_verification_sha256']
            assert meta['model_sha256']==spec['model_sha256']==sha(v['model'])
            path=next((directory/'native').glob('replay-*/steps.jsonl'))
            assert sha(path)==meta['source_sha256'][str(path.resolve())]
        else:
            path=Path(r['root'])/'dataset/steps.jsonl'
            assert r['dataset']['command_proof']['accepted'] and r['dataset']['observed_reset_confirmed']
        with path.open(encoding='utf-8-sig') as stream:steps=[json.loads(line) for line in stream]
        life=r['first_life'];goal=r['goal_stop'];end=goal['kill_frame'] if goal else life['end_frame']
        active=[s for s in steps if 98<=s['observation']['identity']['frame']<end and s['observation']['identity']['life']==1 and s['observation']['health']>0]
        assert active
        expected='provider' if v['mode']=='learned' else 'rules'
        ownership=all(s['owner']==expected for s in active)
        dispatch=all((s.get('server_execution') or {}).get('matched') and (s.get('server_execution') or {}).get('window_exclusive')
            and (s.get('server_execution') or {}).get('recovery_commands')==0 and s.get('native_step') for s in active)
        o=active[0]['observation'];weapon,idle,slot=('Shotgun',19,2) if v['label']=='shotgun' else ('Blaster',9,1)
        log=(Path(r['root'])/'server.log').read_text(encoding='utf-8-sig')
        ready=len(re.findall(r'^g_test_weapon_start game_frame=100 actor=1 weapon='+weapon+r' gunframe_before=\d+ gunframe_after='+str(idle)+r'$',log,re.M))==1
        stock=sorted((x['name'],x['count']) for x in o['inventory'])==[('Blaster',1),('Shells',20),('Shotgun',1)]
        initial=o['identity']['frame']==98 and equipped_weapon_index(o)==slot and 0<=o['inventory_age_frames']<=2
        start=sorted((e['class'],e['health'],e['solid'],tuple(e['position'])) for e in r['generated_start']['entities'])
        initial_nonweapon={k:o.get(k) for k in ('position','velocity','view_angles','kick_angles_degrees','health','armor','on_ground','ducked')}
        damage=life['damage'];attacks=stationary=unclear=0;yaw_degrees=0.;previous=None
        for s in active:
            obs,cmd=s['observation'],s['sent_command'];attack=bool(cmd['Buttons']&1);attacks+=attack
            vel=obs.get('velocity');stationary+=bool(attack and vel and len(vel)==3 and math.hypot(vel[0],vel[1])<=20)
            unclear+=bool(attack and not any(e.get('clear_shot') for e in obs['enemies']))
            if previous is not None:yaw_degrees+=abs((cmd['Yaw']-previous+32768)%65536-32768)*360/65536
            previous=cmd['Yaw']
        timing=physical_shots(steps,log,98,end)
        metrics=dict(goal=bool(goal),death=bool(r['death_stop']),end_frame=life['end_frame'],
            kills=sum(x['monster_kills'] for x in damage['by_mod']),dealt=sum(x['monster_health_damage'] for x in damage['by_mod']),
            received=damage['received_health_damage'],active_frames=len(active),attack_command_frames=attacks,
            stationary_attack_frames=stationary,attack_without_observed_clear_enemy_frames=unclear,
            cumulative_command_yaw_degrees=yaw_degrees,first_physical_shot=timing[0] if timing else None)
        loaded.append(dict(v=v,start=start,nonweapon=initial_nonweapon,ownership=ownership,dispatch=dispatch,ready=ready,stock=stock,initial=initial,metrics=metrics))
    results=[]
    for d in loaded:
        v=d['v'];peers=[x for x in loaded if x['v']['context_index']==v['context_index']]
        matched=all(x['start']==d['start'] and x['nonweapon']==d['nonweapon'] for x in peers)
        admitted=matched and all(x['ownership'] and x['dispatch'] and x['ready'] and x['stock'] and x['initial'] for x in peers)
        results.append(dict(map=v['map'],episode=v['episode'],seed=v['seed'],label=v['label'],mode=v['mode'],admitted=bool(admitted),
            matched_native_start=matched,ownership_exact=d['ownership'],dispatch_exact=bool(d['dispatch']),native_ready=d['ready'],
            inventory_exact=d['stock'],initial_equipment=d['initial'],**d['metrics']))
    totals=[];system_totals=[]
    for label in ('blaster','shotgun'):
        for mode in ('learned','rules'):
            rr=[x for x in results if x['label']==label and x['mode']==mode and x['admitted']]
            totals.append(dict(label=label,mode=mode,conditions=len(rr),wins=sum(x['goal'] for x in rr),
                deaths=sum(x['death'] for x in rr),kills=sum(x['kills'] for x in rr),dealt=sum(x['dealt'] for x in rr),received=sum(x['received'] for x in rr)))
            # Ownership failures are product failures, not a reason to remove
            # difficult worlds from the primary quality denominator.
            all_cases=[x for x in results if x['label']==label and x['mode']==mode]
            system_totals.append(dict(label=label,mode=mode,conditions=len(all_cases),wins=sum(x['goal'] for x in all_cases),
                deaths=sum(x['death'] for x in all_cases),timeouts=sum(not x['goal'] and not x['death'] for x in all_cases),
                ownership_failures=sum(not x['ownership_exact'] for x in all_cases),
                kills=sum(x['kills'] for x in all_cases),dealt=sum(x['dealt'] for x in all_cases),received=sum(x['received'] for x in all_cases)))
    a.out.mkdir(parents=True)
    result=dict(state='complete',version='combat_ready_weapon_rules_train_screen_acceptance_v2',
        admitted=sum(x['admitted'] for x in results),excluded=sum(not x['admitted'] for x in results),results=results,
        technically_valid_native_cases=32,system_totals=system_totals,pure_controller_subset_totals=totals,
        native_cases=32,cuda_cases=16,independent_worlds=8,training_performed=False,promotion=False,
        scope='Primary system_totals include all32technicallyvalid captures, including learned-to-rules ownership failures. Strict28case subset is conditional, not the quality denominator. Exploratory TRAIN only; no fitted or held-out superiority claim. Stationary/occluded-fire and yaw are diagnostics, not proof of waste.')
    save(a.out/'report.json',result);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(admitted=result['admitted'],excluded=result['excluded'],system_totals=system_totals))


if __name__=='__main__':main()

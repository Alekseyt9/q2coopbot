"""Separate lost visibility from missing target intent using accepted offline traces."""
import argparse
import collections
import json
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','aim-audit','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path);h=pins[key] if key in pins else sha(path)
        assert expected is None or expected==h,str(path);pins[key]=h;return path
    pin(__file__);closed=read(pin(a.aim_audit/'complete.json'))
    parent=read(pin(a.aim_audit/'report.json',closed['report_sha256']))
    assert parent['version']=='combat_underfire_aim_tradeoff_audit_v1'
    for path,h in parent['inputs'].items():pin(path,h)
    frames=read(pin(a.aim_audit/'frames.json',parent['frames_sha256']))
    shots=read(pin(a.aim_audit/'shots.json',parent['shots_sha256']))
    cases=[];timeline=[]
    for case in parent['cases']:
        i=case['case_index'];folder=a.collection/f'case-{i:02d}';meta=read(pin(folder/'cuda/report.json'))
        sp=next((folder/'native').glob('replay-*/steps.jsonl'))
        steps=list(map(json.loads,pin(sp,meta['source_sha256'][str(sp.resolve())]).open(encoding='utf-8-sig')))
        ep=sp.with_name('server_outcomes.jsonl');effects={e['step']:e for e in map(json.loads,pin(ep,meta['source_sha256'][str(ep.resolve())]).open(encoding='utf-8-sig'))}
        chosen={r['frame'] for r in frames if r['case_index']==i};assert len(chosen)==10
        health={};last_target=None;counts=collections.Counter();rows=[]
        for s in steps:
            o=s['observation'];f=o['identity']['frame']
            if o['identity']['life']!=1 or f>max(chosen):continue
            target=s['action'].get('target_entity',0)
            if target:last_target=target
            if f in chosen:
                assert effects[s['index']]['available']
                visible=bool(o['enemies']);attack=s['applied_action']['attack'];memory=o.get('remembered_threats') or []
                remembered=next((t for t in memory if t['id']==last_target),None)
                counts['no_visible_enemy_frames']+=not visible
                counts['no_target_with_visible_enemy_frames']+=bool(visible and not target)
                counts['attack_without_visible_enemy_frames']+=bool(attack and not visible)
                counts['attack_without_target_with_visible_enemy_frames']+=bool(attack and visible and not target)
                counts['lost_visibility_with_fresh_memory_frames']+=bool(not visible and remembered and remembered['age_frames']<=5)
                if not visible or not target:
                    item=dict(case_index=i,map=case['map'],stream_seed=case['stream_seed'],label=case['label'],frame=f,
                        observed_enemies=len(o['enemies']),declared_target=target,attack=attack,
                        yaw_delta_degrees=s['applied_action']['yaw_delta_degrees'],pitch_delta_degrees=s['applied_action']['pitch_delta_degrees'],
                        remembered_last_target=remembered,last_recorded_native_target_health=health.get(last_target),
                        mg_outcomes=[q['outcome'] for q in shots if q['case_index']==i and q['frame']==f])
                    rows.append(item);timeline.append(item)
            effect=effects.get(s['index'])
            if effect and effect['available']:
                for e in effect['events']:
                    if e['target_class'].startswith('monster_'):health[e['target']]=e['health_after']
        cases.append(dict(**{key:case[key] for key in ('map','world_seed','stream_seed','label','case_index')},counts=dict(counts),
            lost_target_frames=len(rows),last_target_last_recorded_native_health=health.get(last_target)))
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True);save(a.out/'timeline.json',timeline)
    report=dict(state='complete',version='combat_underfire_target_loss_audit_v1',cases=cases,inputs=pins,timeline_sha256=sha(a.out/'timeline.json'),
        neural_execution=False,new_games=False,training_performed=False,promotion=False,
        scope='Posthoc observation/intent/memory and recorded native-health description. Native health is an offline diagnostic, never an input or runtime target. No inferred visibility geometry or causal attribution from angular error alone.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(cases=len(cases),ware2=[c for c in cases if c['map']=='ware2']))


if __name__=='__main__':main()

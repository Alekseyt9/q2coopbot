"""Matched fresh-seed TRAIN contact-loss development cohort, all cases retained."""
import argparse
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from audit_combat_contact_loss_probe import load_steps


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--screen',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--processing-root',type=Path,help='Explicit offline processing receipt with rejected export admission')
    a=ap.parse_args();assert not a.out.exists();root=a.screen.resolve()
    spec,pool=read(root/'spec.json'),read(root/'pool/report.json')
    assert spec['version']=='combat_memory_route_candidate_native_v1' and spec['fixed_health_100'] and spec['fresh_seed_offsets']
    assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==spec['native_budget']
    processing=a.processing_root.resolve() if a.processing_root else root
    batch=read(processing/'cuda-batch/report.json');assert batch['state']=='complete'
    learned_ids={v['index'] for v in spec['variants'] if v['label']!='rules'}
    admission=None;extra_pins={}
    if a.processing_root:
        path=processing/'export-admission.json';admission=read(path)
        assert admission['version']=='combat_native_export_admission_v1' and admission['pool_report_sha256']==sha(root/'pool/report.json')
        assert {r['index'] for r in admission['cases']}==learned_ids
        accepted_ids={r['index'] for r in admission['cases'] if r['state']=='accepted_native_export'}
        assert all(r['state'] in ('accepted_native_export','rejected_before_cuda') for r in admission['cases'])
        for p,h in admission['inputs'].items():assert sha(p)==h
        extra_pins[str(path)]=sha(path)
    else:accepted_ids=learned_ids
    assert len(batch['tasks'])==len(accepted_ids) and len(learned_ids)==spec['cuda_budget']
    jobs={j['plan_index']:j for j in pool['jobs']};results=[];sites=set();site_pins={}
    for v in spec['variants']:
        source_plan=Path(v['context']['source_plan'])
        site=read(source_plan)['tasks'][0]['episode']['generator']['site']
        assert site['map']==v['context']['map']
        sites.add((site['map'],site['entity_index']))
        site_pins[str(source_plan.resolve())]=sha(source_plan)
        i=v['index'];capture=Path(jobs[i]['root']);rpt=read(capture/'report.json');r=rpt['results'][0]
        assert rpt['capture_complete'] and rpt['provenance_valid'] and r['capture_valid'] and r['dispatch_valid'] and r['seed_confirmed']
        assert r['generated_fixture']==v['context']['instance'] and r['generated_fixture']['health']==100
        d=root/f'case-{i:02d}';active=0;fallback=None
        verified=i in accepted_ids
        if v['label']!='rules' and verified:
            meta=read(d/'cuda/report.json');proof=read(d/'cuda/cuda-verification.json')
            assert meta['model_sha256']==v['model_sha256']==sha(v['model']) and proof['state']=='passed' and proof['device']=='cuda'
            assert sha(d/'cuda/cuda-verification.json')==meta['cuda_verification_sha256']
            if meta['feature_version']=='combat_features_v13_memory_static_routes':
                for row in load_steps(d/'cuda/rollout.jsonl'):
                    x=row['features']
                    active+=x[0]>0 and x[1489]==1 and not any(x[73+12*j] for j in range(8)) and any(x[881+30*j]==1 and x[1490+8*j]==1 and x[1491+8*j]==1 for j in range(8))
        if v['label']!='rules' and not verified:active=None
        life=r['first_life'];damage=life['damage'];end=r['goal_stop']['kill_frame'] if r['goal_stop'] else life['end_frame']
        if v['label']!='rules':
            steps=load_steps(next((d/'native').glob('replay-*/steps.jsonl')))
            owned=[s for s in steps if s['observation']['identity']['life']==1 and s['observation']['health']>0 and s['observation']['identity']['frame']<end]
            fallback=sum(s['owner']!='provider' for s in owned)
        results.append(dict(map=v['context']['map'],seed=v['context']['seed'],loadout=v['context']['loadout'],
            controller={'zero':'candidate','source':'source','original':'original_source','rules':'rules'}[v['label']],goal=bool(r['goal_stop']),death=bool(r['death_stop']),
            autonomous_goal=bool(r['goal_stop']) and fallback in (None,0),
            kills=sum(m['monster_kills'] for m in damage['by_mod']),dealt=sum(m['monster_health_damage'] for m in damage['by_mod']),
            received=damage['received_health_damage'],fallback_frames=fallback,route_active_rows=active,
            inference_verified=verified if v['label']!='rules' else None))
    totals=[]
    for controller in ('source','candidate','original_source','rules'):
        rr=[r for r in results if r['controller']==controller]
        if not rr:continue
        assert len(rr)==spec['independent_worlds']
        totals.append(dict(controller=controller,battles=len(rr),wins=sum(r['autonomous_goal'] for r in rr),native_goals=sum(r['goal'] for r in rr),deaths=sum(r['death'] for r in rr),
            kills=sum(r['kills'] for r in rr),dealt=sum(r['dealt'] for r in rr),received=sum(r['received'] for r in rr),
            ownership_failures=sum(bool(r['fallback_frames']) for r in rr) if controller!='rules' else None,route_active_rows=sum(r['route_active_rows'] or 0 for r in rr),
            route_active_battles=sum(bool(r['route_active_rows']) for r in rr)))
    assert len({r['seed'] for r in results})==spec['independent_worlds']
    save(a.out,dict(state='passed',results=results,totals=totals,promotion=False,
        control_role=spec.get('control_role','original_source'),metric_version='autonomous_goal_without_handoff_v1',
        independent_worlds=spec['independent_worlds'],map_sites=len(sites),
        source_only_geometry_selection=spec.get('selection_kind')!='known_regression_diagnostic',
        selection_kind=spec.get('selection_kind','source_selected_development'),
        accepted_cuda_cases=len(accepted_ids),requested_cuda_cases=len(learned_ids),numerical_verification_complete=accepted_ids==learned_ids,
        rejected_export_cases=sorted(learned_ids-accepted_ids),route_active_measurement_complete=accepted_ids==learned_ids,
        scope=('Known failure-selected TRAIN mechanism diagnostic; all declared cases count, not an independent population-quality estimate.'
               if spec.get('selection_kind')=='known_regression_diagnostic' else
               'Geometry-selected TRAIN diagnosis with fixed100HP. No selection by outcome; all cases count, no held-out superiority claim.'),
        inputs={**{str((root/p).resolve()):sha(root/p) for p in ('spec.json','pool/report.json')},
                str(processing/'cuda-batch/report.json'):sha(processing/'cuda-batch/report.json'),**site_pins,**extra_pins}))
    print(totals,flush=True)


if __name__=='__main__':main()

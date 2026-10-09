"""Paired native gameplay results; no model inference or training in this report."""
import argparse, collections, hashlib, json, pathlib
from rebind_combat_architecture_evaluation import read, sha, save

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=pathlib.Path,required=True)
    ap.add_argument('--member-proof',type=pathlib.Path); args=ap.parse_args()
    root=args.root.resolve(); protocol=read(root/'protocol.json'); pool_path=args.member_proof or root/'pool/report.json';pool=read(pool_path)
    if args.member_proof:
        assert pool['version']=='verified_evaluation_members_v1' and pool['protocol_sha256']==sha(root/'protocol.json')
    assert pool['state']=='complete' and pool['source_unchanged'],'Incomplete cohort'
    assert len(pool['jobs'])==protocol['total_episodes'] and not any(j['error'] for j in pool['jobs'])
    variants=[]; allrows=[]; paired={}; source_fingerprints=set()
    for entry in protocol['evaluations']:
        plan=read(entry['plan']); assert sha(entry['plan'])==entry['plan_sha256']
        label=entry['model']+'-'+entry['label']; rows=[]
        for i,task in enumerate(plan['tasks']):
            mode=task['modes'][0]; group=pathlib.Path(entry['root'])/f'case-{i}-{mode}'
            if not args.member_proof:
                aggregate=read(group/'report.json'); manifest=read(group/'manifest.json')
                assert aggregate['capture_complete'] and aggregate['provenance_valid']
                assert sha(group/'q2combat-export.exe')==manifest['exporter_sha256'].lower()
                source_fingerprints.add(manifest['source_fingerprint'])
            for seed in task['seeds']:
                member=group/f's-{seed}'; report=read(member/'report.json'); result=report['results'][0]
                if args.member_proof:
                    receipt=pool['members'][str(member)]
                    assert sha(member/'report.json')==receipt['report_sha256'] and sha(member/'manifest.json')==receipt['manifest_sha256']
                    source_fingerprints.add(read(member/'manifest.json')['source_fingerprint'])
                assert report['capture_complete'] and report['provenance_valid'] and len(report['results'])==1
                assert result['seed']==seed and all(result[k] for k in ('capture_valid','dispatch_valid','seed_confirmed','frame_budget_valid'))
                if task.get('instances'):
                    expected=task['instances'][task['seeds'].index(seed)]
                    assert result['generated_fixture']==expected and result['generated_start']['confirmed']
                life=result['first_life']; damage=life['damage']; goal=result.get('goal_stop')
                win=bool(goal and goal['reason']=='combat_goal_complete')
                row={'variant':label,'episode':task['episode']['id'],'seed':seed,'win':win,
                     'death':life['end_reason']=='first_observed_death',
                     'kills':sum(x.get('monster_kills',0) for x in damage.get('by_mod',[])),
                     'outgoing_damage':sum(x.get('monster_health_damage',0) for x in damage.get('by_mod',[])),
                     'received_damage':damage['received_health_damage'],'frames':result['actual_game_frames'],
                     'reward':result['dataset'].get('reward_sum'), 'report_sha256':sha(member/'report.json'),
                     'capture_guard_interventions':result['guard_interventions'],
                     'capture_provider_controlled_frames':result['provider_controlled_frames'],
                     'capture_frame_gaps':result['frame_gaps'],
                     'selection_p95_us':result['selection_p95_us']}
                rows.append(row); allrows.append(row)
        assert len(rows)==protocol['episodes_per_model']
        wins=sum(x['win'] for x in rows); deaths=sum(x['death'] for x in rows)
        per_family={}
        for family in protocol['families']:
            selected=[x for x in rows if x['episode']==family]
            per_family[family]={'wins':sum(x['win'] for x in selected),'episodes':len(selected)}
        latency=[x['selection_p95_us'] for x in rows if x['selection_p95_us'] is not None]
        variants.append({'variant':label,'episodes':len(rows),'wins':wins,'win_rate':wins/len(rows),'deaths':deaths,
                         'mean_capture_guard_interventions':sum(x['capture_guard_interventions'] for x in rows)/len(rows),
                         'capture_provider_controlled_frames':sum(x['capture_provider_controlled_frames'] for x in rows),
                         'capture_frame_gaps':sum(x['capture_frame_gaps'] for x in rows),
                         'mean_selection_p95_us':sum(latency)/len(latency) if latency else None,
                         'mean_kills':sum(x['kills'] for x in rows)/len(rows),
                         'mean_outgoing_damage':sum(x['outgoing_damage'] for x in rows)/len(rows),
                         'mean_received_damage':sum(x['received_damage'] for x in rows)/len(rows),'families':per_family})
        paired[label]={(x['episode'],x['seed']):x['win'] for x in rows}
    assert len(source_fingerprints)==1,'Mixed harness cohort'
    comparisons=[]
    models=sorted({e['model'] for e in protocol['evaluations'] if e['label'] in ('before','after')})
    for model in models:
        before=paired[model+'-before']; after=paired[model+'-after']; assert before.keys()==after.keys()
        comparisons.append({'model':model,'gained_wins':sum(after[k] and not before[k] for k in before),
                            'lost_wins':sum(before[k] and not after[k] for k in before),
                            'win_delta':sum(after.values())-sum(before.values())})
    reference_comparisons=[]
    if protocol.get('comparison_reference'):
        reference_label=protocol['comparison_reference'];reference=paired[reference_label]
        for label,results in paired.items():
            if label==reference_label:continue
            assert reference.keys()==results.keys(),'Reference conditions differ'
            reference_comparisons.append(dict(variant=label,reference=reference_label,
                gained_wins=sum(results[k] and not reference[k] for k in reference),
                lost_wins=sum(reference[k] and not results[k] for k in reference),
                win_delta=sum(results.values())-sum(reference.values())))
    report={'state':'complete','episodes':len(allrows),'variants':variants,'paired_changes':comparisons,
            'reference_comparisons':reference_comparisons,
            'protocol_sha256':sha(root/'protocol.json'),'pool_sha256':sha(pool_path),'member_proof':str(pool_path),
            'scope':protocol.get('scope','Native paired validation on reused validation seeds; not untouched final-test superiority. Victory requires verified goal-stop.'),
            'comparison_stage':protocol.get('comparison_stage','cuda_ppo'),
            'source_fingerprint':next(iter(source_fingerprints)),
            'execution_diagnostics_scope':'Guard events/provider frames/frame gaps are full native capture totals, not unique first-life collision counts. Selection latency is measured under pool load; synchronous lockstep throughput is separate from realtime behavior.'}
    save(root/'quality-report.json',report); save(root/'quality-episodes.json',allrows)
    lines=['# Парное сравнение архитектур, 2026-10-09','',report['scope'],'',
           '| Вариант | Победы | Смерти | Урон монстрам, средний | Полученный урон, средний |',
           '| --- | ---: | ---: | ---: | ---: |']
    for v in variants: lines.append(f"| {v['variant']} | {v['wins']}/{v['episodes']} | {v['deaths']} | {v['mean_outgoing_damage']:.1f} | {v['mean_received_damage']:.1f} |")
    lines+=['','| Модель | Изменение побед | Новые победы | Потерянные победы |','| --- | ---: | ---: | ---: |']
    for c in comparisons: lines.append(f"| {c['model']} | {c['win_delta']:+d} | {c['gained_wins']} | {c['lost_wins']} |")
    if reference_comparisons:
        lines+=['',f"Парные изменения относительно {protocol['comparison_reference']}:",'',
                '| Вариант | Изменение побед | Новые победы | Потерянные победы |','| --- | ---: | ---: | ---: |']
        for c in reference_comparisons:
            lines.append(f"| {c['variant']} | {c['win_delta']:+d} | {c['gained_wins']} | {c['lost_wins']} |")
    lines+=['',report['execution_diagnostics_scope'],'',
            '| Вариант | Guard events, среднее / capture | Provider frames, всего | Frame gaps, всего | Mean selection p95, ms |',
            '| --- | ---: | ---: | ---: | ---: |']
    for v in variants:
        latency='—' if v['mean_selection_p95_us'] is None else f"{v['mean_selection_p95_us']/1000:.2f}"
        lines.append(f"| {v['variant']} | {v['mean_capture_guard_interventions']:.1f} | {v['capture_provider_controlled_frames']} | {v['capture_frame_gaps']} | {latency} |")
    (root/'quality-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(root/'progress.json',{'stage':'quality_report_complete','episodes':len(allrows)})
    print(json.dumps({'state':'complete','episodes':len(allrows),'report':str(root/'quality-report.md')}))

if __name__=='__main__': main()

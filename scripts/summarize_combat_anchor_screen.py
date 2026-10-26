"""Summarize all TRAIN mechanism pairs, keeping post-hoc admission explicit."""
import argparse
from pathlib import Path
from process_combat_architecture_pool import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args();root=args.root.resolve();assert not (root/'summary.json').exists()
    complete=read(root/'complete.json');assert complete['state']=='complete' and complete['original_protocol_passed'] is False
    comparison=read(root/'comparison.json');assert sha(root/'comparison.json')==complete['comparison_sha256']
    assert comparison['state']=='passed' and comparison['posthoc_scope_correction']
    spec,pool=read(root/'spec.json'),read(root/'pool/report.json')
    effects=read(root/'branch-effects.json');continuation=read(root/'continuation-first-life.json')
    assert continuation['first_life_only'] and continuation['includes_rules']
    assert effects['source_complete_sha256']==continuation['source_complete_sha256']==sha(root/'complete.json')
    metrics={(r['context_id'],r['mode']):r for r in comparison['metrics']}
    shots={(r['context_id'],r['mode']):r for r in effects['records']}
    timelines={(r['context_id'],r['mode']):r['timeline'] for r in continuation['cases']}
    jobs={j['plan_index']:j for j in pool['jobs']};bounds={};pins={}
    for a in spec['assignments']:
        path=Path(jobs[a['plan_index']]['root'])/'report.json';r=read(path)['results'][0]
        bounds[(a['context_id'],a['mode'])]=dict(end=r['first_life']['end_frame'],terminal=bool(r.get('goal_stop') or r.get('death_stop')))
        pins[str(path)]=sha(path)
    pairs=[]
    totals={m:dict(goals=0,deaths=0,kills=0,dealt=0,received=0,root_mg_damage=0) for m in ('baseline','observed_mg')}
    for c in spec['contexts']['contexts']:
        cid=c['context_id'];arms={}
        for mode in ('baseline','observed_mg'):
            k=(cid,mode);m=metrics[k];shot=shots[k];bound=bounds[k];root_frame=c['branch_frame']
            # At an observed terminal state there is no further task action.
            # Preserve incomplete live horizons; mark declared task-stop separately.
            rows=[r for r in timelines[k] if r['frame']<bound['end'] or not bound['terminal'] and r['frame']==bound['end']]
            horizons={}
            for count in (1,3,5,10,20):
                rr=rows[:count];terminal=bound['terminal'] and root_frame+count>bound['end']
                horizons[str(count)]=dict(requested_rows=count,observed_rows=len(rr),complete=len(rr)==count,
                    task_terminal_before_horizon=terminal,known_task_return=len(rr)==count or terminal,
                    provider_rows=sum(t['owner']=='provider' for t in rr),rules_rows=sum(t['owner']=='rules' for t in rr),
                    learned_only=all(t['owner']=='provider' for t in rr),
                    dealt=sum(t['native_damage'].get('dealt',0) for t in rr),received=sum(t['native_damage'].get('received',0) for t in rr),
                    kills=sum(t['native_damage'].get('kills',0) for t in rr),mg_bullets=sum(t['native_mg_bullets'] for t in rr),
                    mg_damage=sum(t['native_mg_live_damage'] for t in rr),target_changes=sum(a['target']!=b['target'] for a,b in zip(rr,rr[1:])),
                    movement_suppressed=sum(max(abs(x) for x in t['requested_movement'][:2])>=.25 and t['sent_movement']==[0,0,0] for t in rr))
            arms[mode]=dict(goal=m['goal'],death=m['death'],kills=m['kills'],dealt=m['dealt'],received=m['received'],
                root_mg_damage=shot['live_monster_health_damage'],root_selected_bbox_intersection=shot['selected_native_bbox_intersection'],horizons=horizons,
                continuation_owners={owner:sum(t['owner']==owner for t in rows) for owner in ('provider','rules')})
            t=totals[mode]
            for target,source in (('goals','goal'),('deaths','death'),('kills','kills'),('dealt','dealt'),('received','received'),('root_mg_damage','root_mg_damage')):
                t[target]+=arms[mode][source]
        pairs.append(dict(context_id=cid,map=c['map'],source_update=c['update'],seed=c['seed'],branch_frame=c['branch_frame'],arms=arms))
    changes=dict(root_damage_better=0,root_damage_same=0,root_damage_worse=0,whole_dealt_better=0,whole_dealt_same=0,whole_dealt_worse=0)
    for p in pairs:
        b,a=(p['arms'][m] for m in ('baseline','observed_mg'))
        for prefix,key in (('root_damage','root_mg_damage'),('whole_dealt','dealt')):
            changes[prefix+('_better' if a[key]>b[key] else '_same' if a[key]==b[key] else '_worse')]+=1
    for p,h in pins.items():assert sha(p)==h
    save(root/'summary.json',dict(state='complete',unique_train_contexts=len(pairs),native_cases=24,cuda_cases=24,
        original_protocol_passed=False,posthoc_scope_correction=True,totals=totals,changes=changes,pairs=pairs,pins=pins,
        complete_sha256=sha(root/'complete.json'),training_performed=False,promotion=False,quality_confirmation=False,
        scope='All8 outcome-blind selected TRAIN states included. Different frozen source per context, so sums are descriptive paired mechanism evidence, not current-model quality or an architecture ranking. V1/V2 rejections preserved. Current within-state active first-life records compared exactly; debug enemy snapshot order only is canonicalized. Historical replay mismatch remains explicit. Horizons exclude post-task commands and distinguish declared terminal return from unobserved live future.'))
    lines=['# Observed anchor: other-map TRAIN mechanism screen','',
        'Post-hoc first-life analysis; original V1 protocol remains rejected. No fitting or promotion.','',
        '| Map | Immediate MG damage | Whole dealt HP | Kills | Goal | Received HP |',
        '|---|---:|---:|---:|---|---:|']
    for p in pairs:
        b,a=(p['arms'][m] for m in ('baseline','observed_mg'))
        arrow=lambda k:f"{b[k]} -> {a[k]}"
        lines.append('| '+p['map']+' | '+' | '.join(arrow(k) for k in ('root_mg_damage','dealt','kills','goal','received'))+' |')
    lines+=['','These are reused TRAIN states with different source models; no independent quality claim.',
        'Current-pair observations/history/source events/native commands are exact before the root. Identity is exact throughout active first life.',
        'V1 full-file identity differed only in post-goal shutdown tails; old historical entity IDs differ on two maps and remain unnormalized.',
        'All24 captures were retained and replayed on CUDA. No diagnostic data enters ordinary PPO.']
    lines+=['Whole-fight effects can include rules after handoff; per-horizon and continuation owner counts are explicit. These sums are not autonomous learned-policy scores.']
    (root/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(dict(totals=totals,changes=changes))


if __name__=='__main__':main()

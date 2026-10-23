"""TRAIN-only observed physics/native effects after an admitted movement intervention; no NN."""
import argparse
import json
import math
from pathlib import Path
import statistics

from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','protocol','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();root=a.collection.resolve();assert not a.out.exists();pins={}

    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or expected==h,str(path)
        pins[str(path)]=h;return path

    protocol=read(pin(a.protocol));assert protocol['version']=='combat_movement_short_horizon_audit_v1'
    ds=read(pin(root/'dataset/report.json',protocol['dataset_report_sha256']))
    assert ds['gate_passed']
    records=[r for r in read(pin(root/'dataset/records.json',protocol['records_sha256']))['records'] if r['split']=='train']
    assert len(records)==ds['counts']['train_contexts']
    closed=read(pin(root/'complete.json'))
    spec=read(pin(root/'spec.json',closed['spec_sha256']))
    assert closed['version']=='combat_native_movement_pairs_v1'
    by_context={item['context']['context_id']:item for item in spec['contexts']}
    horizons=protocol['horizons_frames'];assert horizons==[1,3,5,10]
    comparisons=[];cases=[]

    def utility(row):
        return 4*float(row['goal'] and not row['death'])+.25*row['kills']-row['received']/100.-float(row['death'])

    def vector_distance(x,y):
        return math.sqrt(sum((left-right)**2 for left,right in zip(x,y)))

    for record in records:
        context=record['context'];frame=context['frame'];item=by_context[context['context_id']]
        proof=read(pin(record['proof'],record['proof_sha256']))
        assert proof['baseline_full_exact'] and proof['excluded']==0
        labels={v['label']:v for v in record['labels']};loaded={}
        for v in item['variants']:
            assert v['label'] in labels
            directory=root/f"case-{v['plan_index']:02d}"
            meta_path=directory/'cuda/report.json'
            meta=read(pin(meta_path,proof['inputs'][str(meta_path.resolve())]))
            steps_path=next((directory/'native').glob('replay-*/steps.jsonl'))
            steps=list(map(json.loads,pin(steps_path,meta['source_sha256'][str(steps_path.resolve())]).open(encoding='utf-8-sig')))
            effects_path=steps_path.with_name('server_outcomes.jsonl')
            effects=list(map(json.loads,pin(effects_path,meta['source_sha256'][str(effects_path.resolve())]).open(encoding='utf-8-sig')))
            effects={r['step']:r for r in effects};assert len(effects)==len(set(effects))
            active={s['observation']['identity']['frame']:s for s in steps if s['owner']=='provider' and s.get('native_step') and s['observation']['identity']['life']==1}
            first=active[frame];o=first['observation']
            parasites=[e for e in o['enemies'] if e['class']=='monster_parasite']
            nearest=min(parasites,key=lambda e:e['distance']);parasite_id=nearest['id']
            snapshots={}
            for s in steps:
                for key in ('observation','next_observation'):
                    q=s.get(key)
                    if q and q['identity']['life']==1:
                        f=q['identity']['frame']
                        if f>=frame and f not in snapshots:snapshots[f]=q
            measurements={}
            for h in horizons:
                selected=[active.get(f) for f in range(frame,frame+h)]
                full=all(selected) and frame+h in snapshots
                if not full:
                    measurements[h]=dict(full_window=False,missing_native_frames=[frame+i for i,s in enumerate(selected) if s is None],snapshot_missing=frame+h not in snapshots)
                    continue
                chunks=[]
                for s in selected:
                    ns=s['native_step'];dispatch=s['server_execution']
                    assert dispatch['matched'] and dispatch['window_exclusive'] and dispatch['recovery_commands']==0
                    assert ns['begin_frame']==s['observation']['identity']['frame'] and ns['end_frame']==ns['begin_frame']+1
                    effect=effects[s['index']];assert effect['available']
                    chunks.append(effect)
                q=snapshots[frame+h]
                after=next((e for e in q['enemies'] if e['id']==parasite_id),None)
                measurements[h]=dict(full_window=True,position=q['position'],velocity=q['velocity'],health=q['health'],
                    displacement=vector_distance(q['position'],o['position']),
                    received=sum(r['received_health_damage'] for r in chunks),
                    dealt=sum(r['monster_health_damage'] for r in chunks),kills=sum(r['monster_kills'] for r in chunks),
                    deaths=sum(r['deaths'] for r in chunks),parasite_visible=after is not None,
                    parasite_distance=after['distance'] if after else None,parasite_distance_before=nearest['distance'])
            loaded[v['label']]=dict(first_command=first['sent_command'],initial_position=o['position'],measurements=measurements)
        zero=loaded['control'];zlabel=labels['control']
        cases.append(dict(context=context,measurements=loaded,labels=labels))
        for label,values in loaded.items():
            if label=='control':continue
            assert values['initial_position']==zero['initial_position']
            for h in horizons:
                left,right=values['measurements'][h],zero['measurements'][h]
                pair=dict(context_id=context['context_id'],seed=record['seed'],map=record['map'],label=label,horizon=h,
                    command_changed=values['first_command']!=zero['first_command'],full_window=left['full_window'] and right['full_window'],
                    final_utility_delta=utility(labels[label])-utility(zlabel),useful=labels[label]['useful'])
                if pair['full_window']:
                    pair.update(position_difference=vector_distance(left['position'],right['position']),
                        velocity_difference=vector_distance(left['velocity'],right['velocity']),
                        received_delta=left['received']-right['received'],dealt_delta=left['dealt']-right['dealt'],
                        kill_delta=left['kills']-right['kills'],death_delta=left['deaths']-right['deaths'],
                        parasite_distance_delta=left['parasite_distance']-right['parasite_distance'] if left['parasite_visible'] and right['parasite_visible'] else None)
                comparisons.append(pair)
    summary=[]
    for h in horizons:
        all_pairs=[v for v in comparisons if v['horizon']==h];good=[v for v in all_pairs if v['full_window']]
        large=[v for v in good if abs(v['final_utility_delta'])>=.5]
        tied=lambda v:v['received_delta']==v['dealt_delta']==v['kill_delta']==v['death_delta']==0
        distances=[v for v in good if v['parasite_distance_delta'] is not None and abs(v['parasite_distance_delta'])>=5 and abs(v['final_utility_delta'])>=.05]
        summary.append(dict(horizon=h,total_pairs=len(all_pairs),full_pairs=len(good),censored_pairs=len(all_pairs)-len(good),
            changed_commands=sum(v['command_changed'] for v in all_pairs),positions_differ_by_at_least_1=sum(v['position_difference']>=1 for v in good),
            median_position_difference=statistics.median(v['position_difference'] for v in good) if good else None,
            early_effects_tied=sum(tied(v) for v in good),large_final_change=len(large),large_final_change_early_effects_tied=sum(tied(v) for v in large),
            received_improved=sum(v['received_delta']<0 for v in good),received_worsened=sum(v['received_delta']>0 for v in good),
            paired_distance_changes=len(distances),distance_and_final_utility_same_sign=sum(v['parasite_distance_delta']*v['final_utility_delta']>0 for v in distances)))
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True)
    save(a.out/'contexts.json',cases);save(a.out/'comparisons.json',comparisons)
    report=dict(state='complete',version='combat_movement_short_horizon_audit_v1',split='train_only',contexts=len(records),
        alternatives=len([v for v in comparisons if v['horizon']==1]),summary=summary,inputs=pins,
        contexts_sha256=sha(a.out/'contexts.json'),comparisons_sha256=sha(a.out/'comparisons.json'),script_sha256=sha(__file__),
        neural_execution=False,training_performed=False,new_games=False,promotion=False,
        scope='Descriptive native conditional consequences only. Censoring and missing visibility explicit; late differences with tied early effects do not establish random label noise or an adequate/inadequate representation.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(contexts=report['contexts'],alternatives=report['alternatives'],summary=summary))


if __name__=='__main__':main()

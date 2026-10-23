"""Compare accepted one/five-frame interventions under the same four RNG streams."""
import argparse
import copy
import json
import statistics
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from audit_combat_joint_aim_grid import physical_equal_with_entity_bijection


def utility(result):
    return 4*float(result['goal'] and not result['death'])+.25*result['kills']-result['received']/100.-float(result['death'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collection',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();root=a.collection.resolve();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path);h=pins[key] if key in pins else sha(path)
        assert expected is None or expected==h,str(path);pins[key]=h;return path
    seal=read(pin(root/'complete.json'));assert seal['version']=='combat_native_continuation_pairs_v1'
    spec=read(pin(root/'spec.json',seal['spec_sha256']));p=read(pin(spec['protocol'],spec['protocol_sha256']))
    assert p['study']=='combat_movement_window_comparison_v1'
    seeds=p['continuation_seeds'];order=p['movement_labels'];assert len(seeds)==len(set(seeds))==4
    ds=Path(p['dataset']);ds_seal=read(pin(ds/'complete.json'))
    records=read(pin(ds/'records.json',ds_seal['records_sha256']))['records']
    train=[r for r in records if r['split']=='train']
    selected=[min([r for r in train if r['map']==mp],key=lambda r:r['seed']) for mp in sorted({r['map'] for r in train})]
    originals={r['context']['context_id']:r for r in selected};assert len(originals)==4
    groups={1:{},5:{}};excluded=[]
    collections=[(1,Path(item['collection']),Path(item['proofs'])) for item in p['previous_collections']]+[(5,root,root/'proofs')]
    for duration,collection,proofs in collections:
        closed=read(pin(collection/'complete.json'));specification=read(pin(collection/'spec.json',closed['spec_sha256']))
        for unit in specification['contexts']:
            prep=read(pin(Path(unit['prepared'])/'report.json',unit['prepared_report_sha256']))
            seed=prep['fork']['seed']
            if seed not in seeds:continue
            cid=unit['context']['context_id'];assert cid in originals and unit['context']==originals[cid]['context']
            assert prep['source_weights_sha256']==p['source_weights_sha256']
            if duration==5:assert prep['movement_duration_frames']==5
            folder=proofs/unit['unit_id'];ps=read(pin(folder/'complete.json'))
            proof=read(pin(folder/'report.json',ps['report_sha256']))
            for path,h in proof['inputs'].items():pin(path,h)
            zero=proof['zero_fork_full_exact'] or (proof['version']=='combat_native_continuation_probe_acceptance_v3_native_tail' and proof.get('zero_fork_task_exact'))
            admitted=proof['gate_passed'] and proof['nofork_full_exact'] and zero and proof['fork_prefix_exact']
            labels={v['label']:v for v in proof['results']};assert set(order)<=labels.keys()
            if duration==5:
                admitted &= all(labels[name]['movement_window'] and labels[name]['movement_window']['duration_frames']==5 for name in order)
            key=(cid,seed);assert key not in groups[duration]
            if not admitted:
                excluded.append(dict(context_id=cid,stream_seed=seed,duration=duration,reason='Native/CUDA/control admission failed'))
                continue
            groups[duration][key]=dict(labels={name:labels[name] for name in order},collection=collection,unit=unit)

    def control_capture(item):
        collection,unit=item['collection'],item['unit']
        control=next(v for v in unit['variants'] if v['label']=='control');index=control['plan_index']
        folder=collection/f'case-{index:02d}';meta=read(pin(folder/'cuda/report.json'))
        assert meta['observation_timing_contract']=='frozen_world_v1'
        step_path=next((folder/'native').glob('replay-*/steps.jsonl'))
        steps=list(map(json.loads,pin(step_path,meta['source_sha256'][str(step_path.resolve())]).open(encoding='utf-8-sig')))
        pool=read(pin(collection/'pool/report.json'));job=next(j for j in pool['jobs'] if j['plan_index']==index)
        capture=read(pin(Path(job['root'])/'report.json'));result=capture['results'][0]
        rows=list(map(json.loads,pin(folder/'cuda/rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig')))
        controlled=[s for s in steps if s['owner']=='provider' and s.get('sample') and s.get('native_step') and s['observation']['identity']['life']==1 and s['observation']['health']>0]
        return dict(steps=controlled,rows={r['frame']:r for r in rows},result=result)

    def same_zero_task(short,long):
        a,b=control_capture(short),control_capture(long)
        if a['result']['generated_fixture']!=b['result']['generated_fixture']:return False
        ga,gb=a['result']['goal_stop'],b['result']['goal_stop']
        if bool(ga)!=bool(gb) or bool(a['result']['death_stop'])!=bool(b['result']['death_stop']):return False
        end=None
        if ga:
            signature=lambda g:(g['kill_frame'],sorted(g['classes']),sorted((k['frame'],k['target_class']) for k in g['kills']))
            if signature(ga)!=signature(gb):return False
            end=ga['kill_frame']
        fields=('sampling_seed','latent','attack','vertical','weapon','target','aim_mode','memory','aim_candidate','continuation_probe')
        def same_steps(left,right):
            forward={};reverse={}
            if len(left)!=len(right):return False
            for x,y in zip(left,right):
                frame=x['observation']['identity']['frame']
                if frame!=y['observation']['identity']['frame']:return False
                if not physical_equal_with_entity_bijection(x['observation'],y['observation'],forward,reverse):return False
                if any(x['sample'].get(k,0)!=y['sample'].get(k,0) for k in fields) or x['sent_command']!=y['sent_command']:return False
                if (frame in a['rows'])!=(frame in b['rows']):return False
                if frame in a['rows'] and a['rows'][frame]['features']!=b['rows'][frame]['features']:return False
                nx,ny=x.get('next_observation'),y.get('next_observation')
                if (nx is None)!=(ny is None):return False
                if nx is not None and not physical_equal_with_entity_bijection(nx,ny,forward,reverse):return False
            return True
        outcomes_equal=all(short['labels']['control'][k]==long['labels']['control'][k] for k in ('goal','death','kills','received'))
        # Full control identity already proves the contract. The special
        # completion-row condition applies only to truncated goal-tail matching.
        if same_steps(a['steps'],b['steps']):return outcomes_equal
        if end is None:return False
        if not all(any(s['observation']['identity']['frame']==end and s.get('next_observation') for s in d['steps']) for d in (a,b)):return False
        steps=lambda d:[s for s in d['steps'] if s['observation']['identity']['frame']<=end]
        return outcomes_equal and same_steps(steps(a),steps(b))

    paired={};zero_checks=[]
    for cid in originals:
        for seed in seeds:
            key=(cid,seed)
            if key not in groups[1] or key not in groups[5]:continue
            zero_exact=same_zero_task(groups[1][key],groups[5][key])
            zero_checks.append(dict(context_id=cid,stream_seed=seed,observed_zero_task_exact=zero_exact))
            if not zero_exact:
                excluded.append(dict(context_id=cid,stream_seed=seed,reason='One/five zero controls differ on observed task'));continue
            paired[key]={d:groups[d][key]['labels'] for d in (1,5)}
    contexts=[]
    def stats(values):return dict(mean=statistics.mean(values),sample_std=statistics.stdev(values),minimum=min(values),maximum=max(values))
    for cid,original in originals.items():
        available=[seed for seed in seeds if (cid,seed) in paired]
        row=dict(context_id=cid,map=original['map'],seed=original['seed'],stream_seeds=available,complete=available==seeds)
        if available==seeds:
            durations={};outcomes={}
            for d in (1,5):
                deltas={label:[utility(paired[cid,seed][d][label])-utility(paired[cid,seed][d]['control']) for seed in seeds] for label in order}
                first={label:statistics.mean(v[:2]) for label,v in deltas.items()};second={label:statistics.mean(v[2:]) for label,v in deltas.items()}
                top=[label for label in order if max(first.values())-first[label]<.05];choice=top[0]
                durations[str(d)]=dict(utility_deltas=deltas,statistics={label:stats(v) for label,v in deltas.items()},
                    first_two_means=first,new_two_means=second,first_half_choice=choice,new_half_gain_of_first_choice=second[choice])
                outcomes[str(d)]={label:[{k:paired[cid,seed][d][label][k] for k in ('goal','death','kills','received')} for seed in seeds] for label in order}
            same_action={label:[utility(paired[cid,seed][5][label])-utility(paired[cid,seed][1][label]) for seed in seeds] for label in order}
            chosen=durations['5']['first_half_choice']
            row.update(durations=durations,outcomes=outcomes,long_minus_short=same_action,
                long_minus_short_statistics={label:stats(v) for label,v in same_action.items()},
                mean_long_minus_short_for_long_first_choice=statistics.mean(same_action[chosen]))
        contexts.append(row)
    complete=[c for c in contexts if c['complete']];q=p['long_action_signal_screen']
    confirmed=sum(c['durations']['5']['first_half_choice']!='control' and c['durations']['5']['new_half_gain_of_first_choice']>=.25 for c in complete)
    worst=min((c['durations']['5']['new_half_gain_of_first_choice'] for c in complete),default=None)
    better_short=sum(c['durations']['5']['first_half_choice']!='control' and c['mean_long_minus_short_for_long_first_choice']>=.25 for c in complete)
    checks=dict(coverage=len(complete)==4,
        repeated_noncontrol_gain=confirmed>=q['minimum_contexts_repeated_noncontrol_gain_at_least0.25'],
        downside=worst is not None and worst>=q['minimum_worst_newhalf_gain'],
        same_action_better_than_short=better_short>=q['minimum_contexts_same_action_long_minus_short_mean_at_least0.25'])
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True);save(a.out/'contexts.json',contexts);save(a.out/'zero-controls.json',zero_checks)
    technical=len(paired)==16 and not excluded
    report=dict(state='complete',version='combat_movement_window_comparison_report_v1',gate_passed=technical,
        long_action_signal_screen_passed=technical and all(checks.values()),signal_checks=checks,
        summary=dict(complete_contexts=len(complete),paired_context_streams=len(paired),expected_context_streams=16,
            repeated_noncontrol_gain_contexts=confirmed,worst_newhalf_gain=worst,same_action_better_than_short_contexts=better_short),
        contexts_sha256=sha(a.out/'contexts.json'),zero_controls_sha256=sha(a.out/'zero-controls.json'),excluded=excluded,inputs=pins,
        training_performed=False,neural_execution=False,fit_admission=False,ordinary_training_eligible=False,promotion=False,
        scope='Descriptive TRAIN-only action-duration screen in four fixed worlds conditional on fixed root draws. Future randomness paired. Prior short captures reused unchanged. Technical admission and exploratory signal are separate, no independent policy quality or true-Q claim.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(summary=report['summary'],technical_gate=technical,signal_gate=report['long_action_signal_screen_passed'],excluded=len(excluded)))


if __name__=='__main__':main()

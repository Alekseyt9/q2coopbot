"""Merge immutable previous streams and four new streams; test four-sample mean stability."""
import argparse
import statistics
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collection',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();root=a.collection.resolve();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path);h=pins[key] if key in pins else sha(path)
        assert expected is None or expected==h,str(path);pins[key]=h;return path
    closed=read(pin(root/'complete.json'));assert closed['version']=='combat_native_continuation_pairs_v1'
    spec=read(pin(root/'spec.json',closed['spec_sha256']));p=read(pin(spec['protocol'],spec['protocol_sha256']))
    assert p['study']=='combat_movement_mc_budget8_v1'
    seeds=p['previous_continuation_seeds']+p['continuation_seeds'];assert len(seeds)==len(set(seeds))==7
    old_report_path=pin(p['previous_report']);old_report=read(old_report_path)
    os_=read(pin(old_report_path.with_name('complete.json')));assert sha(old_report_path)==os_['report_sha256'] and old_report['gate_passed']
    source=Path(p['dataset']);ss=read(pin(source/'complete.json'))
    ds=read(pin(source/'report.json',ss['report_sha256']));assert ds['gate_passed'] and ds['source_weights_sha256']==p['source_weights_sha256']
    records=[r for r in read(pin(source/'records.json',ss['records_sha256']))['records'] if r['split']=='train']
    selected=[min([r for r in records if r['map']==mp],key=lambda r:r['seed']) for mp in sorted({r['map'] for r in records})]
    originals={r['context']['context_id']:r for r in selected};assert len(originals)==4
    groups={cid:{0:{v['label']:v for v in r['labels']}} for cid,r in originals.items()};excluded=[];units=0
    for collection,proof_root in [(Path(p['previous_collection']),Path(p['previous_proofs'])),(root,root/'proofs')]:
        seal=read(pin(collection/'complete.json'));specification=read(pin(collection/'spec.json',seal['spec_sha256']))
        for unit in specification['contexts']:
            cid=unit['context']['context_id'];assert cid in originals and unit['context']==originals[cid]['context']
            folder=proof_root/unit['unit_id'];ps=read(pin(folder/'complete.json'));proof=read(pin(folder/'report.json',ps['report_sha256']))
            for path,h in proof['inputs'].items():pin(path,h)
            prepared=read(pin(Path(unit['prepared'])/'report.json',unit['prepared_report_sha256']))
            assert prepared['source_weights_sha256']==p['source_weights_sha256']
            replicate=seeds.index(proof['fork']['seed'])+1;assert replicate not in groups[cid]
            if not proof['gate_passed']:
                excluded.append(dict(unit_id=unit['unit_id'],proof=str(folder/'report.json'),reason='Control/prefix/draw acceptance failed'));continue
            zero=proof['zero_fork_full_exact'] or (proof['version']=='combat_native_continuation_probe_acceptance_v3_native_tail' and proof['zero_fork_task_exact'])
            assert proof['nofork_full_exact'] and zero and proof['fork_prefix_exact']
            assert all(v['admitted'] for v in proof['results'])
            groups[cid][replicate]={v['label']:v for v in proof['results']};units+=1
    def utility(r):return 4*float(r['goal'] and not r['death'])+.25*r['kills']-r['received']/100.-float(r['death'])
    order=['control','left','right','forward','backward','near_stationary']
    contexts=[];preferences=[]
    for cid,streams in groups.items():
        labels=[label for label in order if label in streams[0]]
        assert all(set(v)==set(labels) for v in streams.values())
        complete=set(streams)==set(range(8));deltas={}
        for label in labels:
            d=[utility(streams[k][label])-utility(streams[k]['control']) for k in sorted(streams)];deltas[label]=d
            if label!='control':preferences.append(dict(context_id=cid,map=originals[cid]['map'],label=label,replicates=sorted(streams),utility_deltas=d,
                observed_mean_delta=statistics.mean(d),sample_standard_deviation=statistics.stdev(d) if len(d)>1 else None,
                minimum_delta=min(d),maximum_delta=max(d),complete=complete,sign_changes=any(v>=.05 for v in d) and any(v<=-.05 for v in d)))
        row=dict(context_id=cid,seed=originals[cid]['seed'],map=originals[cid]['map'],complete=complete,replicates=sorted(streams),utility_deltas=deltas)
        if complete:
            first={label:statistics.mean(deltas[label][:4]) for label in labels};second={label:statistics.mean(deltas[label][4:]) for label in labels}
            top_first=[label for label in labels if max(first.values())-first[label]<.05]
            top_second=[label for label in labels if max(second.values())-second[label]<.05]
            chosen=next(label for label in order if label in top_first)
            row.update(first_four_mean=first,new_four_mean=second,all_eight_mean={label:statistics.mean(deltas[label]) for label in labels},
                first_four_top=top_first,new_four_top=top_second,top_sets_overlap=bool(set(top_first)&set(top_second)),
                first_half_chosen=chosen,new_half_gain_of_first_choice=second[chosen])
        contexts.append(row)
    complete=[r for r in contexts if r['complete']];q=p['budget4_diagnostic']
    overlap=sum(r['top_sets_overlap'] for r in complete)
    noncontrol=sum(r['first_half_chosen']!='control' and r['new_half_gain_of_first_choice']>=.25 for r in complete)
    worst=min((r['new_half_gain_of_first_choice'] for r in complete),default=None)
    checks=dict(coverage=len(complete)==4,rank_overlap=overlap>=q['minimum_contexts_with_overlapping_top_sets'],
        noncontrol_signal=noncontrol>=q['minimum_noncontrol_firsthalf_choices_with_newhalf_gain_at_least0.25'],
        downside=worst is not None and worst>=q['minimum_worst_newhalf_gain_of_firsthalf_choice'])
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True);save(a.out/'contexts.json',contexts);save(a.out/'preferences.json',preferences)
    summary=dict(contexts=4,complete_contexts=len(complete),accepted_fork_units=units,expected_fork_units=28,
        continuations_per_context=8,top_sets_overlap=overlap,noncontrol_firsthalf_choices_repeated_gain=noncontrol,
        worst_newhalf_gain_of_first_choice=worst,complete_action_preferences=sum(r['complete'] for r in preferences),
        sign_unstable=sum(r['complete'] and r['sign_changes'] for r in preferences))
    report=dict(state='complete',version='combat_continuation_budget8_report_v1',gate_passed=not excluded and units==28,
        budget4_supported=all(checks.values()),budget4_checks=checks,summary=summary,contexts_sha256=sha(a.out/'contexts.json'),preferences_sha256=sha(a.out/'preferences.json'),
        excluded=excluded,inputs=pins,training_performed=False,neural_execution=False,fit_admission=False,ordinary_training_eligible=False,promotion=False,
        intervention_frame_draws_fixed=True,root_action_distribution_integrated=False,
        scope='TRAIN-only split-half stability conditional on fixed original intervention-frame draws and actual sampled actions. Future policy randomness varied; root Gaussian draws were not integrated. Original streams reused unchanged; eight-sample means not independently validated. No fitted winner, superiority or true-Q claim.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(summary=summary,budget4_supported=report['budget4_supported'],checks=checks,excluded=len(excluded)))


if __name__=='__main__':main()

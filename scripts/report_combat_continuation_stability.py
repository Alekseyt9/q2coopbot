"""Observed preference stability across forked future policy streams; no NN or fitting."""
import argparse
import itertools
from pathlib import Path
import statistics
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collection',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--proofs-root',type=Path,help='Separate supplemental proofs; original capture report remains immutable')
    a=ap.parse_args();root=a.collection.resolve();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or expected==h,str(path)
        pins[str(path)]=h;return path
    closed=read(pin(root/'complete.json'));assert closed['version']=='combat_native_continuation_pairs_v1'
    spec=read(pin(root/'spec.json',closed['spec_sha256']));p=read(pin(spec['protocol'],spec['protocol_sha256']))
    original_gate=None
    if a.proofs_root:
        original_report=read(pin(root/'dataset/report.json'));original_gate=original_report['gate_passed']
    assert p['version']=='combat_movement_continuation_stability_v1'
    original=Path(p['dataset']);oc=read(pin(original/'complete.json'))
    od=read(pin(original/'report.json',oc['report_sha256']));assert od['gate_passed']
    records=[r for r in read(pin(original/'records.json',oc['records_sha256']))['records'] if r['split']=='train']
    selected=[min([r for r in records if r['map']==mp],key=lambda r:r['seed']) for mp in sorted({r['map'] for r in records})]
    originals={r['context']['context_id']:r for r in selected};assert len(originals)==4
    groups={cid:{} for cid in originals};excluded=[];admitted=0

    def utility(r):return 4*float(r['goal'] and not r['death'])+.25*r['kills']-r['received']/100.-float(r['death'])
    def useful(r,b):return bool((r['goal'] and not r['death'] and not b['goal']) or
        (r['goal'] and b['goal'] and not r['death'] and not b['death'] and r['kills']>=b['kills'] and b['received']-r['received']>=5))
    for cid,r in originals.items():groups[cid][0]={v['label']:v for v in r['labels']}
    for unit in spec['contexts']:
        cid=unit['context']['context_id'];assert cid in originals and unit['context']==originals[cid]['context']
        folder=(a.proofs_root if a.proofs_root else root/'proofs')/unit['unit_id'];pc=read(pin(folder/'complete.json'))
        proof=read(pin(folder/'report.json',pc['report_sha256']));assert proof['context']==unit['context']
        for path,h in proof['inputs'].items():pin(path,h)
        replicate=p['continuation_seeds'].index(proof['fork']['seed'])+1
        assert replicate not in groups[cid]
        if not proof['gate_passed']:
            excluded.append(dict(unit_id=unit['unit_id'],proof=str(folder/'report.json'),reason='Native prefix/control/draw acceptance failed'))
            continue
        zero_valid=proof['zero_fork_full_exact']
        if a.proofs_root and not zero_valid:
            assert proof['version']=='combat_native_continuation_probe_acceptance_v3_native_tail'
            zero_valid=proof['zero_fork_task_exact'] and proof['goal_tail_supplement_applied']
            assert proof['goal_tail_details']['post_goal_native_effects_empty']
        assert proof['nofork_full_exact'] and zero_valid and proof['fork_prefix_exact']
        admitted+=proof['admitted'];groups[cid][replicate]={v['label']:v for v in proof['results']}
    contexts=[];preferences=[];rank_same=rank_total=ties_changed=0
    for cid,streams in groups.items():
        original_row=originals[cid];labels=set(streams[0])
        assert all(set(v)==labels for v in streams.values())
        rows=[]
        for replicate,actions in sorted(streams.items()):
            values={label:utility(v) for label,v in actions.items()};best=max(values.values())
            rows.append(dict(replicate=replicate,fork_seed=None if replicate==0 else p['continuation_seeds'][replicate-1],
                control=actions['control'],utility=values,best_labels=[label for label,v in values.items() if best-v<.05]))
        for label in sorted(labels-{'control'}):
            deltas=[utility(streams[k][label])-utility(streams[k]['control']) for k in sorted(streams)]
            flags=[useful(streams[k][label],streams[k]['control']) for k in sorted(streams)]
            preferences.append(dict(context_id=cid,seed=original_row['seed'],map=original_row['map'],label=label,
                replicates=sorted(streams),utility_deltas=deltas,observed_mean_delta=statistics.mean(deltas),
                minimum_delta=min(deltas),maximum_delta=max(deltas),useful=flags,useful_count=sum(flags),
                complete=len(deltas)==4,sign_changes=any(v>=.05 for v in deltas) and any(v<=-.05 for v in deltas)))
        local_same=local_total=local_ties=0
        for a_stream,b_stream in itertools.combinations(sorted(streams),2):
            for left,right in itertools.combinations(sorted(labels),2):
                da=utility(streams[a_stream][left])-utility(streams[a_stream][right]);db=utility(streams[b_stream][left])-utility(streams[b_stream][right])
                if abs(da)>=.05 and abs(db)>=.05:
                    local_total+=1;local_same+=da*db>0
                elif (abs(da)>=.05)!=(abs(db)>=.05):local_ties+=1
        rank_same+=local_same;rank_total+=local_total;ties_changed+=local_ties
        contexts.append(dict(context_id=cid,seed=original_row['seed'],map=original_row['map'],streams=rows,
            ranking_non_tied_comparisons=local_total,ranking_same_sign=local_same,ties_changed=local_ties))
    complete=[r for r in preferences if r['complete']]
    summary=dict(contexts=len(contexts),expected_fork_units=12,accepted_fork_units=sum(len(v)-1 for v in groups.values()),
        original_plus_fork_streams=4,complete_action_preferences=len(complete),sign_unstable=sum(r['sign_changes'] for r in complete),
        ever_useful=sum(r['useful_count']>0 for r in complete),always_useful=sum(r['useful_count']==4 for r in complete),
        originally_useful=sum(r['useful'][0] for r in complete),originally_useful_lost_in_any_fork=sum(r['useful'][0] and not all(r['useful'][1:]) for r in complete),
        ranking_non_tied_comparisons=rank_total,ranking_same_sign=rank_same,ranking_agreement=rank_same/rank_total if rank_total else None,ties_changed=ties_changed)
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True);save(a.out/'contexts.json',contexts);save(a.out/'preferences.json',preferences)
    report=dict(state='complete',version='combat_continuation_stability_report_v1',gate_passed=not excluded and summary['accepted_fork_units']==12,
        summary=summary,excluded=excluded,admitted_variants=admitted,inputs=pins,contexts_sha256=sha(a.out/'contexts.json'),preferences_sha256=sha(a.out/'preferences.json'),
        neural_execution=False,training_performed=False,new_games=False,fit_admission=False,ordinary_training_eligible=False,promotion=False,
        scope='Technical acceptance and descriptive policy-randomness stability in4 fixed TRAIN worlds. Shared prefixes, no environment RNG reset; not independent map seeds or proof of expected Q. Ranking counts are dependent, not inferential sample sizes.')
    if a.proofs_root:
        report.update(version='combat_continuation_stability_report_v2_native_task_tail',original_full_capture_gate_passed=original_gate,
            supplemental_proofs=str(a.proofs_root.resolve()),
            scope='Separate supplemental task-equivalence report. Original full-capture failure preserved; one empty post-goal native tail omission explicitly accepted. Observed policy-randomness stability, not independent environment seeds or proof of expected Q. No fitting admission.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(gate_passed=report['gate_passed'],summary=summary,excluded=len(excluded)))


if __name__=='__main__':main()

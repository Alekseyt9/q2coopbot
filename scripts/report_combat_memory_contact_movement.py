"""Full-task and contact recovery signal on an excluded future stream; no NN."""
import argparse
import collections
import json
import statistics
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();root=a.collection.resolve();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or h==expected,str(path);pins[str(path)]=h;return path
    pin(__file__);seal=read(pin(root/'complete.json'));assert seal['native_cases']==seal['cuda_cases']==76
    spec=read(pin(root/'spec.json',seal['spec_sha256']));p=read(pin(spec['protocol'],spec['protocol_sha256']));assert p['study']=='combat_memory_contact_movement_pilot_v1'
    erratum=read(pin(Path('docs/combat_memory_contact_movement_pilot_protocol_erratum_20261024.json')))
    assert erratum['inherits_sha256']==sha(spec['protocol']) and not erratum['gates_changed'] and not erratum['cases_changed']
    pool=read(pin(root/'pool/report.json',seal['pool_sha256']));jobs={j['plan_index']:j for j in pool['jobs']}
    groups={};strict=0;records=[]
    for unit in spec['contexts']:
        c=unit['context'];assert c['split']=='train' and c['role']=='auxiliary_train' and c['observed_enemies']==0
        folder=root/'proofs'/unit['unit_id'];ps=read(pin(folder/'complete.json'));proof=read(pin(folder/'report.json',ps['report_sha256']))
        assert proof['context']==c
        stream=proof['fork']['seed'];assert stream in p['continuation_seeds'];strict+=int(proof['gate_passed'])
        group=groups.setdefault(c['context_id'],dict(context=c,streams={}));assert stream not in group['streams'];rows={}
        for accepted in proof['results']:
            label=accepted['label'];assert label in p['movement_labels']
            item={k:accepted[k] for k in ('label','admitted','goal','death','kills','received')}
            item['alive_goal']=bool(item['goal'] and not item['death']);item['terminal_known']=bool(item['goal'] or item['death'])
            item['utility']=4*item['alive_goal']+.25*item['kills']-item['received']/100-int(item['death'])
            variant=next(v for v in unit['variants'] if v['label']==label);case=root/f"case-{variant['plan_index']:02d}"
            meta=read(pin(case/'cuda/report.json'));sp=next((case/'native').glob('replay-*/steps.jsonl'));pin(sp,meta['source_sha256'][str(sp.resolve())])
            capture=read(pin(Path(jobs[variant['plan_index']]['root'])/'report.json'))['results'][0]
            stop=capture['goal_stop']['kill_frame'] if capture['goal_stop'] else capture['first_life']['end_frame']
            steps=[s for s in map(json.loads,sp.open(encoding='utf-8-sig')) if s['owner']=='provider' and s.get('sample') and s.get('native_step') and s['observation']['identity']['life']==1 and s['observation']['health']>0 and c['frame']<=s['observation']['identity']['frame']<stop]
            reacquired=next((s['observation']['identity']['frame'] for s in steps if s['observation']['enemies']),None)
            item['reacquisition_delay_frames']=reacquired-c['frame'] if reacquired is not None else None
            item['reacquisition_censored']=reacquired is None
            rows[label]=item;records.append(dict(context_id=c['context_id'],seed=c['seed'],stream_seed=stream,**item))
        assert set(rows)==set(p['movement_labels']);group['streams'][stream]=rows
    assert len(groups)==4 and len(spec['contexts'])==12 and len(records)==60
    selected=[];gained=lost=positive=complete=0;gains=[]
    mean=lambda rows,key:statistics.mean(r[key] for r in rows)
    for cid,g in groups.items():
        streams=g['streams'];assert set(streams)==set(p['continuation_seeds'])
        candidates=[]
        for order,label in enumerate(p['movement_labels']):
            rows=[streams[seed][label] for seed in p['continuation_seeds'][:2]]
            if all(r['admitted'] and r['terminal_known'] for r in rows):candidates.append((mean(rows,'alive_goal'),mean(rows,'kills'),mean(rows,'utility'),-order,label))
        choice=max(candidates)[-1] if candidates else 'control';hold=streams[p['continuation_seeds'][2]];before,after=hold['control'],hold[choice]
        known=all(r['admitted'] and r['terminal_known'] for r in (before,after));complete+=int(known)
        delta=after['utility']-before['utility'];gains.append(delta);gained+=int(after['alive_goal'] and not before['alive_goal']);lost+=int(before['alive_goal'] and not after['alive_goal']);positive+=int(known and delta>=.05)
        selected.append(dict(context_id=cid,map=g['context']['instance']['map'],selected=choice,selection_terminal_candidates=len(candidates),third_stream= p['continuation_seeds'][2],terminal_known_pair=known,before=before,after=after,observed_utility_delta=delta))
    gate=p['signal'];supported=bool(strict>=gate['minimum_strict_units'] and complete>=gate['minimum_complete_pairs'] and gained-lost>=gate['minimum_net_gained_alive_goals'] and lost<=gate['maximum_lost_alive_goals'] and positive>=gate['minimum_worlds_positive_utility'] and statistics.mean(gains)>=gate['minimum_mean_utility_gain'])
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True);save(a.out/'records.json',dict(records=records))
    summary=dict(strict_units=strict,terminal_known_pairs=complete,gained_alive_goals=gained,lost_alive_goals=lost,positive_worlds=positive,mean_observed_utility_gain=statistics.mean(gains))
    save(a.out/'report.json',dict(state='complete',version='combat_memory_contact_movement_pilot_report_v1',gate_passed=supported,signal_supported=supported,summary=summary,selected=selected,records_sha256=sha(a.out/'records.json'),inputs=pins,
        training_performed=False,neural_execution=False,promotion=False,ordinary_training_eligible=False,
        scope='Four TRAIN roots with conditional future repeats. Selection uses first2 streams only; third excluded. Unfinished results explicitly censored and cannot pass complete-pair gate. Reacquisition timing descriptive; this is not trained-policy quality or general superiority.'))
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json'),records_sha256=sha(a.out/'records.json')));print(dict(signal_supported=supported,**summary))

if __name__=='__main__':main()

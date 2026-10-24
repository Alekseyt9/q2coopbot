"""Audit local observed-damage movement effects without inventing terminal labels."""
import argparse
import json
import math
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();root=a.collection.resolve();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path);h=pins[key] if key in pins else sha(path)
        assert expected is None or h==expected,str(path);pins[key]=h;return path
    seal=read(pin(root/'complete.json'));assert seal['version']=='combat_native_movement_pairs_v1'
    spec=read(pin(root/'spec.json',seal['spec_sha256']));p=read(pin(spec['protocol'],spec['protocol_sha256']))
    assert p['version']=='combat_movement_underfire_pairs_v1' and seal['native_cases']==seal['cuda_cases']==36
    fields=('monster_health_damage','received_health_damage','self_health_damage','teammate_health_damage','monster_kills','deaths')
    records=[];sentinels=[];comparisons=[];duration_comparisons=[];context_summaries=[]
    for unit in spec['contexts']:
        c=unit['context'];frame=c['frame'];folder=root/'proofs'/unit['unit_id']
        ps=read(pin(folder/'complete.json'));proof=read(pin(folder/'report.json',ps['report_sha256']))
        assert proof['context']==c
        for path,h in proof['inputs'].items():pin(path,h)
        accepted={r['label']:r for r in proof['results']};local={};roots={}
        for variant in unit['variants']:
            label=variant['label'];acceptance=accepted[label]
            technical=all(acceptance[k] for k in ('baseline_full_exact','prefix_exact','intervention_state_exact',
                'gaussian_and_fire_draws_exact','unmodified_categories_exact','window_common_draws_exact'))
            folder=root/f"case-{variant['plan_index']:02d}";meta=read(pin(folder/'cuda/report.json'))
            assert meta['numerical_verification']=='cuda_verified_v1' and meta['observation_timing_contract']=='frozen_world_v1'
            step_path=next((folder/'native').glob('replay-*/steps.jsonl'))
            steps=list(map(json.loads,pin(step_path,meta['source_sha256'][str(step_path.resolve())]).open(encoding='utf-8-sig')))
            ep=step_path.with_name('server_outcomes.jsonl');effect_list=list(map(json.loads,pin(ep,meta['source_sha256'][str(ep.resolve())]).open(encoding='utf-8-sig')))
            effects={e['step']:e for e in effect_list};assert len(effects)==len(effect_list)
            active={s['observation']['identity']['frame']:s for s in steps if s['owner']=='provider' and s.get('native_step') and s['observation']['identity']['life']==1}
            snapshots={}
            for s in steps:
                for key in ('observation','next_observation'):
                    o=s.get(key)
                    if o and o['identity']['life']==1:snapshots.setdefault(o['identity']['frame'],o)
            first=active[frame];sample=first['sample'];roots[label]=dict(command=first['sent_command'],sample={k:sample.get(k,0) for k in
                ('sampling_seed','latent','attack','vertical','weapon','target','aim_mode','memory','aim_candidate')})
            measurements={}
            for h in p['horizons_frames']:
                chunks=[];unknown=[]
                for f in range(frame,frame+h):
                    s=active.get(f)
                    if s is None:unknown.append(dict(frame=f,reason='missing_provider_native_window'));continue
                    dispatch=s.get('server_execution') or {};ns=s['native_step']
                    if not(dispatch.get('matched') and dispatch.get('window_exclusive') and dispatch.get('recovery_commands')==0 and ns['begin_frame']==f and ns['end_frame']==f+1):
                        unknown.append(dict(frame=f,reason='unaccepted_native_window'));continue
                    e=effects.get(s['index'])
                    if e is None or not e['available']:unknown.append(dict(frame=f,reason='unavailable_native_effect'));continue
                    assert all(type(e[k]) in (int,float) and math.isfinite(e[k]) for k in fields)
                    chunks.append(e)
                endpoint=snapshots.get(frame+h)
                if endpoint is None:unknown.append(dict(frame=frame+h,reason='missing_same_life_endpoint'))
                totals={k:sum(e[k] for e in chunks) for k in fields}
                measured=dict(known=technical and not unknown,technical_admitted=technical,unknown=unknown,known_native_steps=len(chunks),observed_totals=totals)
                if measured['known']:measured.update(effects=totals,end_health=endpoint['health'],end_position=endpoint['position'])
                measurements[str(h)]=measured
            item=dict(context_id=c['context_id'],map=c['instance']['map'],world_seed=c['seed'],frame=frame,label=label,
                duration=variant['duration'],measurements=measurements,technical_admitted=technical,full_episode_admitted=acceptance['admitted'],
                goal=acceptance['goal'],death=acceptance['death'],kills=acceptance['kills'],dealt=acceptance['dealt'],received=acceptance['received'])
            records.append(item);local[label]=item
        assert len(local)==9
        for direction in ('left','right','forward','backward'):
            short,long=direction+'_one',direction+'_five'
            root_exact=roots[short]==roots[long]
            h1=local[short]['measurements']['1'];h5=local[long]['measurements']['1']
            known=h1['known'] and h5['known'];exact=h1['effects']==h5['effects'] if known else None
            assert root_exact and (not known or exact),(c['context_id'],direction)
            sentinels.append(dict(context_id=c['context_id'],direction=direction,root_action_exact=root_exact,h1_known=known,h1_exact=exact))
        for h in p['horizons_frames']:
            control=local['control']['measurements'][str(h)]
            for label,item in local.items():
                if label=='control':continue
                current=item['measurements'][str(h)];known=current['known'] and control['known']
                row=dict(context_id=c['context_id'],map=c['instance']['map'],label=label,horizon=h,known=known)
                if known:row['alternative_minus_control']={k:current['effects'][k]-control['effects'][k] for k in fields}
                comparisons.append(row)
            for direction in ('left','right','forward','backward'):
                short=local[direction+'_one']['measurements'][str(h)]
                long=local[direction+'_five']['measurements'][str(h)]
                known=short['known'] and long['known']
                row=dict(context_id=c['context_id'],map=c['instance']['map'],direction=direction,horizon=h,known=known)
                if known:row['long_minus_short']={k:long['effects'][k]-short['effects'][k] for k in fields}
                duration_comparisons.append(row)
        context_summaries.append(dict(context=c,baseline_full_exact=proof['baseline_full_exact'],
            full_episode_admitted=proof['admitted'],full_episode_excluded=proof['excluded'],variants=[{k:v for k,v in item.items() if k!='measurements'} for item in local.values()]))
    summary=[]
    for h in p['horizons_frames']:
        pairs=[r for r in comparisons if r['horizon']==h];known=[r for r in pairs if r['known']]
        summary.append(dict(horizon=h,known_pairs=len(known),censored_pairs=len(pairs)-len(known),
            incoming_lower=sum(r['alternative_minus_control']['received_health_damage']<0 for r in known),
            incoming_higher=sum(r['alternative_minus_control']['received_health_damage']>0 for r in known),
            incoming_equal=sum(r['alternative_minus_control']['received_health_damage']==0 for r in known),
            outgoing_changed=sum(r['alternative_minus_control']['monster_health_damage']!=0 for r in known)))
    assert len(records)==36 and len(sentinels)==16
    technical=all(r['technical_admitted'] for r in records) and all(s['h1_known'] and s['h1_exact'] for s in sentinels)
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True)
    for name,value in (('records',records),('comparisons',comparisons),('duration-comparisons',duration_comparisons),('sentinels',sentinels)):save(a.out/(name+'.json'),value)
    report=dict(state='complete',version='combat_movement_underfire_report_v1',gate_passed=technical,
        gate_scope='Technical root/common-draw/H1 admission only; never a predictor or runtime action gate.',contexts=context_summaries,
        root_records=36,root_pair_sentinels=16,summary=summary,inputs=pins,
        artifacts={name:sha(a.out/(name+'.json')) for name in ('records','comparisons','duration-comparisons','sentinels')},
        training_performed=False,ordinary_training_eligible=False,runtime_action_admitted=False,promotion=False,
        scope='Four audit worlds with one shared original continuation each; dependent alternatives, no expected-Q, fit or policy improvement claim. Full-episode exclusions retained; unknown horizons not filled.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(technical_gate=technical,records=36,summary=summary))


if __name__=='__main__':main()

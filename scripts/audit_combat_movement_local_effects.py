"""Known local native effects, explicit censoring and root-action inputs; no NN."""
import argparse
import json
import math
import statistics
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from audit_combat_joint_aim_grid import physical_equal_with_entity_bijection


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path);h=pins[key] if key in pins else sha(path)
        assert expected is None or expected==h,str(path);pins[key]=h;return path
    p=read(pin(a.protocol));assert p['version']=='combat_movement_local_effects_v1'
    root=Path(p['source_comparison']);report_path=pin(p['comparison_report'])
    rc=read(pin(report_path.with_name('complete.json')));assert rc['report_sha256']==sha(report_path)
    comparison=read(report_path);assert not comparison['fit_admission']
    seal=read(pin(root/'complete.json'));spec=read(pin(root/'spec.json',seal['spec_sha256']))
    parent=read(pin(spec['protocol'],spec['protocol_sha256']));seeds=parent['continuation_seeds'];labels=parent['movement_labels']
    assert len(seeds)==4 and labels==['control','left','right','forward','backward']
    horizons=p['horizons_frames'];fields=p['labels'];assert horizons==[1,3,5,10,20]
    datasets=[(1,Path(v['collection']),Path(v['proofs'])) for v in parent['previous_collections']]+[(5,root,root/'proofs')]
    loaded={};records=[]
    selected_sample=('sampling_seed','latent','attack','vertical','weapon','target','aim_mode','memory','aim_candidate','continuation_probe')
    for duration,collection,proofroot in datasets:
        closed=read(pin(collection/'complete.json'));cs=read(pin(collection/'spec.json',closed['spec_sha256']))
        pool=read(pin(collection/'pool/report.json',closed['pool_sha256']));jobs={j['plan_index']:j for j in pool['jobs']}
        for unit in cs['contexts']:
            prep=read(pin(Path(unit['prepared'])/'report.json',unit['prepared_report_sha256']));seed=prep['fork']['seed']
            if seed not in seeds:continue
            cid=unit['context']['context_id'];frame=unit['context']['frame']
            ps=read(pin(proofroot/unit['unit_id']/'complete.json'));proof=read(pin(proofroot/unit['unit_id']/'report.json',ps['report_sha256']))
            for path,h in proof['inputs'].items():pin(path,h)
            assert proof['nofork_full_exact'] and proof['fork_prefix_exact']
            assert proof['zero_fork_full_exact'] or (proof['version']=='combat_native_continuation_probe_acceptance_v3_native_tail' and proof.get('zero_fork_task_exact'))
            results={r['label']:r for r in proof['results']}
            for variant in unit['variants']:
                name=variant['label']
                if name not in labels:continue
                acceptance=results[name]
                assert all(acceptance[k] for k in ('prefix_exact','intervention_state_exact','intervention_draws_exact','categories_exact','shared_future_draws_exact'))
                folder=collection/f"case-{variant['plan_index']:02d}";meta=read(pin(folder/'cuda/report.json'))
                assert meta['numerical_verification']=='cuda_verified_v1' and meta['observation_timing_contract']=='frozen_world_v1'
                path=next((folder/'native').glob('replay-*/steps.jsonl'))
                steps=list(map(json.loads,pin(path,meta['source_sha256'][str(path.resolve())]).open(encoding='utf-8-sig')))
                ep=path.with_name('server_outcomes.jsonl')
                effects_list=list(map(json.loads,pin(ep,meta['source_sha256'][str(ep.resolve())]).open(encoding='utf-8-sig')))
                effects={e['step']:e for e in effects_list};assert len(effects)==len(effects_list)
                rows=list(map(json.loads,pin(folder/'cuda/rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig')))
                row_by_frame={r['frame']:r for r in rows}
                active={s['observation']['identity']['frame']:s for s in steps if s['owner']=='provider' and s.get('native_step') and s['observation']['identity']['life']==1}
                assert len(active)==sum(s['owner']=='provider' and bool(s.get('native_step')) and s['observation']['identity']['life']==1 for s in steps)
                first=active[frame];at=row_by_frame[frame];snapshots={}
                for s in steps:
                    for key in ('observation','next_observation'):
                        o=s.get(key)
                        if o and o['identity']['life']==1:snapshots.setdefault(o['identity']['frame'],o)
                outcomes=read(pin(Path(jobs[variant['plan_index']]['root'])/'report.json'))['results'][0]
                root_input=dict(features=at['features'],source_latent=first['sample']['aim_candidate']['source_latent'],
                    final_latent=first['sample']['latent'],attack=first['sample']['attack'],
                    categories={k:first['sample'].get(k,0) for k in ('target','aim_mode','vertical','weapon')},
                    movement_shift=variant['mean_shift'],duration_frames=duration)
                measurements={}
                for h in horizons:
                    chunks=[];unknown=[]
                    for f in range(frame,frame+h):
                        s=active.get(f)
                        if s is None:unknown.append(dict(frame=f,reason='missing_provider_native_window'));continue
                        dispatch=s.get('server_execution') or {};ns=s['native_step']
                        if not(dispatch.get('matched') and dispatch.get('window_exclusive') and dispatch.get('recovery_commands')==0 and ns['begin_frame']==f and ns['end_frame']==f+1):
                            unknown.append(dict(frame=f,reason='unaccepted_native_dispatch'));continue
                        effect=effects.get(s['index'])
                        if effect is None or not effect['available']:
                            unknown.append(dict(frame=f,reason='unavailable_native_effect'));continue
                        assert all(type(effect[k]) in (int,float) and math.isfinite(effect[k]) for k in fields)
                        chunks.append(effect)
                    end=snapshots.get(frame+h)
                    if end is None:unknown.append(dict(frame=frame+h,reason='missing_same_life_end_observation'))
                    totals={k:sum(e[k] for e in chunks) for k in fields}
                    m=dict(known=not unknown,known_native_steps=len(chunks),unknown=unknown,observed_totals=totals)
                    if not unknown:
                        m.update(effects=totals,end_position=end['position'],end_velocity=end['velocity'],end_health=end['health'],
                            visible_enemies=len(end['enemies']),remembered_threats=len(end.get('remembered_threats') or []))
                    measurements[str(h)]=m
                record=dict(context_id=cid,map=unit['context']['instance']['map'],world_seed=unit['context']['seed'],stream_seed=seed,
                    label=name,duration=duration,frame=frame,root_input=root_input,measurements=measurements,
                    full_episode_admitted=acceptance['admitted'],goal=bool(outcomes['goal_stop']),death=bool(outcomes['death_stop']),
                    case=str(folder.resolve()),native_capture=str(outcomes['root']))
                records.append(record);key=(cid,seed,duration,name);assert key not in loaded
                loaded[key]=dict(record=record,active=active,rows=row_by_frame,snapshots=snapshots)
    assert len(records)==160
    def equal_local_controls(a,b,h):
        frame=a['record']['frame'];forward={};reverse={}
        for f in range(frame,frame+h):
            x,y=a['active'].get(f),b['active'].get(f)
            if x is None or y is None:return False
            if not physical_equal_with_entity_bijection(x['observation'],y['observation'],forward,reverse):return False
            if any(x['sample'].get(k,0)!=y['sample'].get(k,0) for k in selected_sample):return False
            if x['sent_command']!=y['sent_command'] or a['rows'][f]['features']!=b['rows'][f]['features']:return False
            xx,yy=x.get('next_observation'),y.get('next_observation')
            if xx is None or yy is None or not physical_equal_with_entity_bijection(xx,yy,forward,reverse):return False
        return True
    comparisons=[];zero_checks=[];sentinels=[]
    contexts=sorted({r['context_id'] for r in records})
    for cid in contexts:
        for seed in seeds:
            zeros={d:loaded[cid,seed,d,'control'] for d in (1,5)}
            for h in horizons:
                known=all(zeros[d]['record']['measurements'][str(h)]['known'] for d in (1,5))
                exact=equal_local_controls(zeros[1],zeros[5],h) if known else None
                if known:assert exact,(cid,seed,h,'Local zero controls differ')
                zero_checks.append(dict(context_id=cid,stream_seed=seed,horizon=h,known=known,exact=exact))
            for name in labels[1:]:
                short,long=(loaded[cid,seed,d,name] for d in (1,5))
                x,y=short['record'],long['record'];ix,iy=dict(x['root_input']),dict(y['root_input'])
                ix.pop('duration_frames');iy.pop('duration_frames')
                assert ix==iy and short['active'][x['frame']]['sent_command']==long['active'][y['frame']]['sent_command']
                assert x['measurements']['1']['effects']==y['measurements']['1']['effects']
                sentinels.append(dict(context_id=cid,stream_seed=seed,label=name,root_action_and_h1_exact=True))
                for h in horizons:
                    a1,a5=(r['measurements'][str(h)] for r in (x,y))
                    known=a1['known'] and a5['known'] and all(zeros[d]['record']['measurements'][str(h)]['known'] for d in (1,5))
                    item=dict(context_id=cid,map=x['map'],stream_seed=seed,label=name,horizon=h,known=known)
                    if known:item['long_minus_short']={k:a5['effects'][k]-a1['effects'][k] for k in fields}
                    comparisons.append(item)
    for cid in contexts:
        for d in (1,5):
            for name in labels:
                values=[loaded[cid,seed,d,name]['record'] for seed in seeds]
                assert all(v['root_input']==values[0]['root_input'] and v['measurements']['1']['effects']==values[0]['measurements']['1']['effects'] for v in values)
    summary=[];stability=[]
    for h in horizons:
        pairs=[r for r in comparisons if r['horizon']==h];good=[r for r in pairs if r['known']]
        components={}
        for k in fields:
            counts=dict(nonzero_pairs=sum(v['long_minus_short'][k]!=0 for v in good),complete_preferences=0,sign_unstable=0,nonzero_mean=0,mean_at_least_sample_std=0)
            for cid in contexts:
                for name in labels[1:]:
                    series=[v['long_minus_short'][k] for v in good if v['context_id']==cid and v['label']==name]
                    if len(series)!=4:continue
                    mean,std=statistics.mean(series),statistics.stdev(series)
                    unstable=min(series)<0<max(series)
                    counts['complete_preferences']+=1;counts['sign_unstable']+=unstable
                    counts['nonzero_mean']+=mean!=0;counts['mean_at_least_sample_std']+=mean!=0 and abs(mean)>=std
                    stability.append(dict(horizon=h,context_id=cid,label=name,component=k,stream_seeds=seeds,
                        deltas=series,mean=mean,sample_std=std,sign_unstable=unstable))
            components[k]=counts
        summary.append(dict(horizon=h,pairs=len(pairs),known_pairs=len(good),censored_pairs=len(pairs)-len(good),components=components))
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True)
    for name,obj in [('records',records),('comparisons',comparisons),('zero-controls',zero_checks),('sentinels',sentinels),('stability',stability)]:save(a.out/(name+'.json'),obj)
    report=dict(state='complete',version='combat_movement_local_effects_report_v1',protocol_sha256=sha(a.protocol),
        root_records=len(records),contexts=len(contexts),root_pair_sentinels=len(sentinels),root_and_h1_sanity_passed=True,
        local_zero_checks=len(zero_checks),known_local_zero_checks=sum(v['known'] for v in zero_checks),
        full_episode_exclusions_retained=sum(not r['full_episode_admitted'] for r in records),summary=summary,
        artifacts={name:sha(a.out/(name+'.json')) for name in ('records','comparisons','zero-controls','sentinels','stability')},inputs=pins,
        training_performed=False,neural_execution=False,new_games=False,fit_admission=False,ordinary_training_eligible=False,promotion=False,
        scope='Local recorded effects conditional on actual root action and declared duration in four explored TRAIN worlds. Later handoff never supplies final labels. Unknown windows remain masked. No policy/expected-Q/generalization claim.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(json.dumps({k:report[k] for k in ('root_records','root_pair_sentinels','known_local_zero_checks','full_episode_exclusions_retained','summary')}))


if __name__=='__main__':main()

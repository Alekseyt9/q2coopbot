"""Native prefix, no-fork replay and forked zero-control acceptance; no NN calls."""
import argparse
import json
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from audit_combat_joint_aim_grid import physical_equal_with_entity_bijection


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('probe','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--context-index',type=int,help='Select one continuation unit from a shared sealed native pool')
    ap.add_argument('--goal-tail-protocol',type=Path,help='Explicit separate task-equivalence supplement; never overwrite full-capture proofs')
    a=ap.parse_args();root=a.probe.resolve();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or expected==h,str(path)
        pins[str(path)]=h;return path
    seal=read(pin(root/'complete.json'));assert seal['version'] in ('combat_native_continuation_probe_v1','combat_native_continuation_pairs_v1')
    multi=seal['version']=='combat_native_continuation_pairs_v1';assert multi==(a.context_index is not None)
    whole=read(pin(root/'spec.json',seal['spec_sha256']))
    spec=whole['contexts'][a.context_index] if multi else whole;context=spec['context'];frame=context['frame']
    prepared=read(pin(Path(spec['prepared'])/'report.json',spec['prepared_report_sha256']))
    assert context==prepared['context'] and context['split']=='train'
    fork=prepared['fork'];assert fork['fork_frame']==frame+1
    if a.goal_tail_protocol:
        tp=read(pin(a.goal_tail_protocol));assert tp['version'] in ('combat_continuation_goal_tail_supplement_v1','combat_continuation_goal_tail_supplement_v2_raw_windows')
        if tp.get('inherits'):pin(tp['inherits'])
    pool=read(pin(root/'pool/report.json',seal['pool_sha256']));assert pool['source_unchanged']
    batch=read(pin(root/'cuda-batch/report.json',seal['cuda_receipt_sha256']));assert batch['state']=='complete'
    source=read(pin(Path(prepared['source'])/'weights.json',prepared['source_weights_sha256']))
    jobs={j['plan_index']:j for j in pool['jobs']}

    def load(folder,capture):
        meta=read(pin(folder/'cuda/report.json'))
        if folder.parent.resolve()==root:
            t=next(t for t in batch['tasks'] if Path(t['out']).resolve()==(folder/'cuda').resolve())
            assert sha(folder/'cuda/report.json')==t['report_sha256']
        proof=read(pin(folder/'cuda/cuda-verification.json',meta['cuda_verification_sha256']))
        assert proof['state']=='passed' and proof['device']=='cuda' and meta['observation_timing_contract']=='frozen_world_v1'
        path=next((folder/'native').glob('replay-*/steps.jsonl'))
        steps=list(map(json.loads,pin(path,meta['source_sha256'][str(path.resolve())]).open(encoding='utf-8-sig')))
        controlled=[s for s in steps if s['owner']=='provider' and s.get('sample') and s.get('native_step') and s['observation']['identity']['life']==1 and s['observation']['health']>0]
        rows=list(map(json.loads,pin(folder/'cuda/rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig')))
        report=read(pin(capture));assert report['capture_complete'] and report['provenance_valid']
        effect_path=path.with_name('server_outcomes.jsonl')
        effects=list(map(json.loads,pin(effect_path,meta['source_sha256'][str(effect_path.resolve())]).open(encoding='utf-8-sig')))
        assert len(effects)==len({q['step'] for q in effects})
        return dict(meta=meta,proof=proof,steps=controlled,all_steps=steps,rows={r['frame']:r for r in rows},result=report['results'][0],effects={q['step']:q for q in effects})

    loaded={}
    window_evidence={}
    for i,v in enumerate(spec['variants']):
        if multi:i=v['plan_index']
        model=read(pin(v['model'],v['model_sha256']))
        header=model.pop('diagnostic_continuation_probe',None);movement=model.pop('diagnostic_movement_probe',None)
        assert model==source
        if v['fork']:assert header==fork
        else:assert header is None and movement is None
        if movement is not None:assert movement['frame']==frame and movement['mean_shift']==v['mean_shift']
        d=load(root/f'case-{i:02d}',Path(jobs[i]['root'])/'report.json');loaded[v['label']]=d
        if movement and movement.get('duration_frames'):
            duration=movement['duration_frames']
            observed=[s for s in d['steps'] if frame<=s['observation']['identity']['frame']<frame+duration]
            observed_frames=[s['observation']['identity']['frame'] for s in observed]
            last=max(s['observation']['identity']['frame'] for s in d['steps'])
            assert observed_frames==list(range(frame,min(frame+duration,last+1))), 'Skipped movement-window frame'
            for s in observed:
                receipt=s['sample'].get('movement_probe')
                assert receipt and receipt['frame']==s['observation']['identity']['frame'] and receipt['version']==movement['version'] and receipt['mean_shift']==movement['mean_shift']
            window_evidence[v['label']]=dict(duration_frames=duration,observed_receipts=len(observed))
        assert d['result']['generated_fixture']==context['instance']
        if v['fork']:
            assert d['meta']['version']=='combat_diagnostic_continuation_rollout_v1' and d['meta']['diagnostic_only'] and not d['meta']['ppo_training_ready']
            assert d['proof']['diagnostic_continuation']['observed'] and d['proof']['diagnostic_continuation']['fire_verified']
    if multi and 'ordinary_no_fork' not in loaded:
        nf=next(v for unit in whole['contexts'] if unit['context']['context_id']==context['context_id']
            for v in unit['variants'] if v['label']=='ordinary_no_fork')
        i=nf['plan_index'];model=read(pin(nf['model'],nf['model_sha256']));assert model==source
        loaded['ordinary_no_fork']=load(root/f'case-{i:02d}',Path(jobs[i]['root'])/'report.json')
    baseline=Path(prepared['baseline']);bs=read(pin(baseline/'complete.json'))
    bp=read(pin(baseline/'pool/report.json',bs['pool_sha256']))
    original_job=next(j for j in bp['jobs'] if j['job']==context['job'])
    original=load(Path(context['directory']),Path(original_job['root'])/'report.json')

    def same_step(a,b,da,db,forward,reverse,allow_missing_after=None):
        f=a['observation']['identity']['frame']
        if f!=b['observation']['identity']['frame']:return False
        if not physical_equal_with_entity_bijection(a['observation'],b['observation'],forward,reverse):return False
        fields=('sampling_seed','latent','attack','vertical','weapon','target','aim_mode','memory','aim_candidate','continuation_probe')
        if any(a['sample'].get(k,0)!=b['sample'].get(k,0) for k in fields):return False
        if a['sent_command']!=b['sent_command']:return False
        if (f in da['rows'])!=(f in db['rows']):return False
        if f in da['rows'] and da['rows'][f]['features']!=db['rows'][f]['features']:return False
        aa,bb=a.get('next_observation'),b.get('next_observation')
        if (aa is None)!=(bb is None):return allow_missing_after is not None and f>allow_missing_after
        return aa is None or physical_equal_with_entity_bijection(aa,bb,forward,reverse)

    def same_prefix(a,b,until=None):
        left=[s for s in a['steps'] if until is None or s['observation']['identity']['frame']<=until]
        right=[s for s in b['steps'] if until is None or s['observation']['identity']['frame']<=until]
        forward={};reverse={}
        return len(left)==len(right) and all(same_step(x,y,a,b,forward,reverse) for x,y in zip(left,right))

    control=loaded['control'];reference=loaded['ordinary_reference'];nofork=loaded['ordinary_no_fork']
    nofork_exact=same_prefix(nofork,original);zero_exact=same_prefix(control,reference)
    fork_prefix_exact=same_prefix(reference,original,frame)
    tail_accepted=False;tail_details=None
    if a.goal_tail_protocol and not zero_exact:
        ga,gb=control['result']['goal_stop'],reference['result']['goal_stop']
        def goal_signature(g):
            return (g['kill_frame'],sorted(g['classes']),sorted((q['frame'],q['target_class']) for q in g['kills']))
        goal_equal=bool(ga and gb and ga['health']>0 and gb['health']>0 and
            not control['result']['death_stop'] and not reference['result']['death_stop'] and goal_signature(ga)==goal_signature(gb))
        if goal_equal:
            end=ga['kill_frame'];left,right=control['steps'],reference['steps'];forward={};reverse={}
            common=min(len(left),len(right))
            prefix_equal=all(same_step(x,y,control,reference,forward,reverse,end) for x,y in zip(left,right))
            completion_present=all(any(s['observation']['identity']['frame']==end and s.get('next_observation') for s in d['steps']) for d in (control,reference))
            extras=left[common:]+right[common:]
            extras_after=all(s['observation']['identity']['frame']>end for s in extras)
            effect_fields=('monster_health_damage','received_health_damage','self_health_damage','teammate_health_damage','monster_kills','deaths')
            empty=True;tail_counts=[];raw_verified_unknown=0
            for d in (control,reference):
                tail=[s for s in d['all_steps'] if s['observation']['identity']['frame']>end and s.get('native_step')]
                tail_counts.append(len(tail))
                raw_windows=None;raw_damage=None
                if tp['version']=='combat_continuation_goal_tail_supplement_v2_raw_windows':
                    from native_projectile_window import records
                    log=pin(Path(d['result']['root'])/'server.log');raw_windows=set();raw_damage=set()
                    with log.open(encoding='utf-8-sig') as stream:
                        for window,line in records(stream):
                            if window is not None and line.startswith('sv_test_step version=1 phase=begin '):raw_windows.add(window)
                            if window is not None and line.startswith('sv_test_damage '):raw_damage.add(window)
                for s in tail:
                    effect=d['effects'].get(s['index'])
                    if effect is not None and effect['available']:
                        if any(effect[k]!=0 for k in effect_fields):empty=False
                    elif effect is not None and effect.get('reason')=='no_next_observation' and raw_windows is not None:
                        ns=s['native_step'];window=tuple(ns[k] for k in ('spawncount','begin_frame','actor','sequence'))
                        if window not in raw_windows or window in raw_damage:empty=False
                        else:raw_verified_unknown+=1
                    else:empty=False
            tail_accepted=bool(prefix_equal and completion_present and extras_after and empty)
            tail_details=dict(goal_frame=end,common_recorded_steps=common,extra_steps=len(extras),common_observed_fields_exact=prefix_equal,
                completion_window_present=completion_present,extra_steps_after_goal=extras_after,post_goal_native_effects_empty=empty,
                tail_steps=tail_counts,unknown_effects_independently_verified_in_raw_windows=raw_verified_unknown)
    zero_task_exact=zero_exact or tail_accepted
    at=lambda d:next(s for s in d['steps'] if s['observation']['identity']['frame']==frame)
    base_at=at(control);base_draw=base_at['sample']['movement_probe']
    results=[]
    for label,d in loaded.items():
        if label in ('ordinary_reference','ordinary_no_fork'):continue
        prefix=same_prefix(d,control,frame-1);current=at(d);forward={};reverse={}
        state=physical_equal_with_entity_bijection(current['observation'],base_at['observation'],forward,reverse) and d['rows'][frame]['features']==control['rows'][frame]['features']
        draw=current['sample']['movement_probe']
        intervention_draws=(draw['gaussian_noise']==base_draw['gaussian_noise'] and draw['fire_uniform']==base_draw['fire_uniform'])
        categories=all(current['sample'].get(k,0)==base_at['sample'].get(k,0) for k in ('target','aim_mode','vertical','weapon','memory'))
        rows={s['observation']['identity']['frame']:s for s in d['steps']};cr={s['observation']['identity']['frame']:s for s in control['steps']}
        shared=sorted(f for f in set(rows)&set(cr) if f>=fork['fork_frame'])
        shared_draws=bool(shared) and all(
            rows[f]['sample']['continuation_probe']['gaussian_noise']==cr[f]['sample']['continuation_probe']['gaussian_noise'] and
            rows[f]['sample']['continuation_probe']['fire_uniform']==cr[f]['sample']['continuation_probe']['fire_uniform'] for f in shared)
        r=d['result'];life=r['first_life'];damage=life['damage'];end=r['goal_stop']['kill_frame'] if r['goal_stop'] else life['end_frame']
        rules=sum(s['owner']=='rules' and frame<=s['observation']['identity']['frame']<end and s['observation']['health']>0 for s in d['all_steps'])
        admitted=all((nofork_exact,zero_task_exact if a.goal_tail_protocol else zero_exact,fork_prefix_exact,prefix,state,intervention_draws,categories,shared_draws,rules==0))
        results.append(dict(label=label,admitted=admitted,prefix_exact=prefix,intervention_state_exact=state,
            intervention_draws_exact=intervention_draws,categories_exact=categories,shared_future_draws_exact=shared_draws,
            shared_future_frames=len(shared),rules_before_task_end=rules,goal=bool(r['goal_stop']),death=bool(r['death_stop']),
            movement_window=window_evidence.get(label),
            kills=sum(v['monster_kills'] for v in damage['by_mod']),received=damage['received_health_damage']))
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True)
    report=dict(state='complete',version='combat_native_continuation_probe_acceptance_v1',context=context,fork=fork,
        unit_id=spec.get('unit_id',context['context_id']),
        nofork_full_exact=nofork_exact,zero_fork_full_exact=zero_exact,fork_prefix_exact=fork_prefix_exact,
        admitted=sum(r['admitted'] for r in results),excluded=sum(not r['admitted'] for r in results),results=results,inputs=pins,
        gate_passed=all(r['admitted'] for r in results),neural_execution=False,new_games=False,training_performed=False,promotion=False,
        scope='Technical common-prefix and forked draw acceptance in one TRAIN context; not label stability or improved policy evidence.')
    if a.goal_tail_protocol:
        report.update(version='combat_native_continuation_probe_acceptance_v2_goal_tail',zero_fork_task_exact=zero_task_exact,
            goal_tail_supplement_applied=tail_accepted,goal_tail_details=tail_details,
            goal_tail_protocol=str(a.goal_tail_protocol.resolve()),goal_tail_protocol_sha256=sha(a.goal_tail_protocol),
            scope='Supplemental exact observed task/control acceptance. Empty post-goal capture-tail omission explicitly allowed; full-capture result retained. No unseen post-recording state identity or policy superiority claimed.')
        if tp['version']=='combat_continuation_goal_tail_supplement_v2_raw_windows':report['version']='combat_native_continuation_probe_acceptance_v3_native_tail'
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(gate_passed=report['gate_passed'],nofork_exact=nofork_exact,zero_exact=zero_exact,fork_prefix_exact=fork_prefix_exact,results=results))


if __name__=='__main__':main()

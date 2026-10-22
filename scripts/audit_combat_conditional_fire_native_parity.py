"""Native source/zero conditional-fire compatibility; compares data, never runs a NN."""
import argparse
import json
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--diagnose-post-goal-tail',action='store_true',help='Diagnose a strictly post-goal capture tail; does not establish full-trace parity')
    parser.add_argument('--episode-boundary',action='store_true',help='Compare the entire semantic episode through the verified earliest goal observation; retain shutdown tails separately')
    args=parser.parse_args();assert not args.out.exists()
    assert not (args.episode_boundary and args.diagnose_post_goal_tail)
    root=args.evaluation.resolve();closed=read(root/'complete.json');assert closed['state']=='complete'
    pool_path=root/'pool/report.json';pool=read(pool_path)
    assert pool['state']=='complete' and pool['source_unchanged'] and sha(pool_path)==closed['pool_sha256']
    plans=[read(item['path']) for item in pool['plans']];assert len(plans)==2
    zero=read(plans[1]['model_path'])
    assert zero['conditional_fire']['version']=='combat_action_conditioned_fire_v1'
    original=read(plans[0]['model_path']); view=dict(zero);view.pop('conditional_fire')
    assert view==original, 'Non-fire model fields changed'
    last=zero['conditional_fire']['layers'][-1]
    assert all(value==0 for row,bias in zip(last['weight'],last['bias']) for value in [*row,bias])
    groups={};pins={str(pool_path):sha(pool_path)}
    for job in pool['jobs']:
        assert not job['error'];index=job['plan_index'];label=['source','fire-zero'][index]
        capture_path=Path(job['root'])/'report.json';capture=read(capture_path);assert capture['capture_complete'] and capture['provenance_valid']
        result=capture['results'][0];assert result['capture_valid']
        trace=Path(result['root'])/'bot.jsonl';sequence=root/('processing-'+label)/('case-%04d'%job['job'])/'native/sequence.jsonl'
        for path in [capture_path,trace,sequence]:pins[str(path)]=sha(path)
        with trace.open(encoding='utf-8-sig') as stream:rows=[json.loads(line) for line in stream]
        with sequence.open(encoding='utf-8-sig') as stream:context=[json.loads(line) for line in stream]
        indexed={r['combat_policy']['observation']['identity']['frame']:r for r in rows}
        assert len(indexed)==len(rows)
        groups.setdefault((job['task_index'],job['seed']),{})[index]=(indexed,context,result)
    cases=[];age_jitter=0;tails=[]
    for (task,seed),pair in sorted(groups.items()):
        assert set(pair)=={0,1};left,lcontext,lresult=pair[0];right,rcontext,rresult=pair[1]
        boundary=None
        if args.episode_boundary and (lresult.get('goal_stop') or rresult.get('goal_stop')):
            lg,rg=lresult.get('goal_stop'),rresult.get('goal_stop')
            assert lg and rg and all(lg[k]==rg[k] for k in ('reason','kill_frame','observed_frame','health','classes'))
            boundary=lg['observed_frame'];assert boundary==lg['kill_frame']+1
            tails.append(dict(seed=seed,episode_end_frame=boundary,goal_kill_frame=lg['kill_frame'],
                source_shutdown_frames=[v['frame'] for v in lcontext if v['frame']>boundary],
                zero_shutdown_frames=[v['frame'] for v in rcontext if v['frame']>boundary]))
            lcontext=[v for v in lcontext if v['frame']<=boundary]
            rcontext=[v for v in rcontext if v['frame']<=boundary]
        assert lcontext and rcontext
        if lcontext!=rcontext:
            assert args.diagnose_post_goal_tail,('canonical context',seed)
            lg,rg=lresult.get('goal_stop'),rresult.get('goal_stop')
            assert lg and rg and all(lg[k]==rg[k] for k in ('reason','kill_frame','health','classes'))
            n=min(len(lcontext),len(rcontext));assert lcontext[:n]==rcontext[:n],('context prefix differs',seed)
            extras=lcontext[n:]+rcontext[n:];boundary=max(lg['observed_frame'],rg['observed_frame'])
            assert extras and all(row['frame']>boundary for row in extras)
            tails.append(dict(seed=seed,goal_kill_frame=lg['kill_frame'],goal_observed_frames=[lg['observed_frame'],rg['observed_frame']],
                extra_context_frames=[row['frame'] for row in extras]))
        # Bind entity IDs by the same actual observed class/track/pose.
        remap={}
        owned={k:v for k,v in left.items() if v['combat_policy']['selection']['owner']=='provider'}
        other_owned={k:v for k,v in right.items() if v['combat_policy']['selection']['owner']=='provider'}
        if boundary is not None:
            owned={k:v for k,v in owned.items() if k<=boundary}
            other_owned={k:v for k,v in other_owned.items() if k<=boundary}
        assert owned and other_owned
        if set(owned)!=set(other_owned):
            assert args.diagnose_post_goal_tail
            lg,rg=lresult.get('goal_stop'),rresult.get('goal_stop')
            assert lg and rg
            boundary=max(lg['observed_frame'],rg['observed_frame'])
            extra_frames=set(owned)^set(other_owned)
            for frame in extra_frames:
                row=owned.get(frame) or other_owned[frame]
                assert frame>boundary and not row['combat_policy']['observation']['enemies']
            common=set(owned)&set(other_owned)
            owned={k:v for k,v in owned.items() if k in common}
            other_owned={k:v for k,v in other_owned.items() if k in common}
        for frame in owned:
            le=owned[frame]['combat_policy']['observation']['enemies'];re=other_owned[frame]['combat_policy']['observation']['enemies']
            assert len(le)==len(re)
            for a,b in zip(le,re):
                for field in ['class','observed_track','relative']:assert a.get(field)==b.get(field),(seed,frame,field)
                assert b['id'] not in remap or remap[b['id']]==a['id'];remap[b['id']]=a['id']
        assert len(remap)==len(set(remap.values()))
        def normalize(value):
            if isinstance(value,list):return [normalize(x) for x in value]
            if not isinstance(value,dict):return value
            result={}
            for key,item in value.items():
                if key=='version' and isinstance(item,str) and item.startswith('ppo:'):continue
                result[key]=remap.get(item,item) if key in ('id','entity','target_entity','target') and isinstance(item,int) else normalize(item)
            return result
        # Normalize each side using its own ID map; source IDs map to themselves.
        original_remap=remap.copy()
        for frame,row in owned.items():
            a=row['combat_policy'];b=other_owned[frame]['combat_policy']
            remap={value:value for value in original_remap.values()}
            expected=normalize(a['observation']);left_sample=normalize(a['selection'].get('sample'))
            remap=original_remap
            actual=normalize(b['observation'])
            # Age is wall-clock freshness metadata, not a feature. Require the
            # same accepted freshness class as policy.ValidateAction (0..300ms).
            # Canonical features/history and every sampled/executed action are
            # still compared exactly; do not mask any geometric/state fields.
            age_left=expected.pop('observation_age_ms');age_right=actual.pop('observation_age_ms')
            assert 0<=age_left<=300 and 0<=age_right<=300
            age_jitter=max(age_jitter,abs(age_left-age_right))
            assert expected==actual,('observation',seed,frame)
            assert left_sample==normalize(b['selection'].get('sample')),('sample',seed,frame)
            for field in ['proposed','applied','proposed_command','applied_command','command_changed']:
                assert a.get(field)==normalize(b.get(field)),(field,seed,frame)
            for field in ['candidate','candidate_command','guarded_command','owner','mode']:
                assert a['selection'].get(field)==normalize(b['selection'].get(field)),(field,seed,frame)
            assert row['sent_command']==other_owned[frame]['sent_command'],('sent command',seed,frame)
        for field in ['minimum_health','damage','incoming_by_mod','incoming_by_attacker_class','outgoing_by_target_class']:
            assert lresult['first_life'][field]==rresult['first_life'][field],('native outcome',seed,field)
        cases.append(dict(seed=seed,task=task,provider_frames=len(owned),canonical_frames=min(len(lcontext),len(rcontext)),
            state='common_prefix_exact' if args.diagnose_post_goal_tail else 'exact'))
    for path,digest in pins.items():assert sha(Path(path))==digest
    save(args.out,dict(state='diagnostic' if args.diagnose_post_goal_tail else 'complete',cases=cases,native_cases=len(pool['jobs']),cuda_cases=closed['cuda_cases'],
         version='native_episode_terminal_parity_v1' if args.episode_boundary else 'native_full_trace_parity_v1',
         strict_full_trace_parity=not (args.diagnose_post_goal_tail or args.episode_boundary),
         semantic_episode_parity=args.episode_boundary,post_goal_tails=tails,
         provider_frames=sum(c['provider_frames'] for c in cases),canonical_features_and_memory_exact=not args.diagnose_post_goal_tail,
         common_prefix_features_memory_samples_and_commands_exact=True,
         observation_age_jitter_max_ms=age_jitter,freshness_class_valid_for_all_frames=True,
         proposed_applied_and_sent_commands_exact=not args.diagnose_post_goal_tail,old_samples_exact_except_declared_model_version=not args.diagnose_post_goal_tail,
         inputs=[dict(path=p,sha256=d) for p,d in pins.items()],script_sha256=sha(__file__),training_performed=False,promotion=False,
         scope=('Diagnostic common prefix only; strictly post-goal empty-enemy tail excluded. Full-trace parity remains unproven. ' if args.diagnose_post_goal_tail else
             'All released frames through the earliest verified terminal observation; asynchronous shutdown tails recorded separately, not compared. ' if args.episode_boundary else
             'All released source/zero provider frames and native canonical histories. ')+
             'Entity identity bijection permitted only by observed class/track/pose; PPO version intentionally differs. No combat improvement claim.'))
    print(args.out)


if __name__=='__main__':main()

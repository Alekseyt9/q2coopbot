"""Scalar prefix/zero/fork admission and full-fight outcomes for contact-loss diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from audit_combat_joint_aim_grid import physical_equal_with_entity_bijection


FIELDS=('sampling_seed','latent','attack','vertical','weapon','target','aim_mode','memory','aim_candidate','log_probability','value')


def load_steps(path):
    with path.open(encoding='utf-8-sig') as stream:return [json.loads(s) for s in stream]


def equivalent(left,right,before=None):
    select=lambda ss:[s for s in ss if s['observation']['identity']['life']==1 and s['observation']['health']>0
        and (before is None or s['observation']['identity']['frame']<before)]
    a,b=select(left),select(right)
    if len(a)!=len(b):return False
    forward,reverse={},{}
    for x,y in zip(a,b):
        if x['owner']!=y['owner'] or x['sent_command']!=y['sent_command']:return False
        if not physical_equal_with_entity_bijection(x['observation'],y['observation'],forward,reverse):return False
        xx,yy=x.get('sample') or {},y.get('sample') or {}
        if any(xx.get(k,0)!=yy.get(k,0) for k in FIELDS):return False
    return True


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--probe',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();root=a.probe.resolve();assert not a.out.exists()
    seal,spec,pool=[read(root/p) for p in ('complete.json','spec.json','pool/report.json')]
    assert seal['native_cases']==seal['cuda_cases']==24 and len(spec['variants'])==24 and pool['source_unchanged']
    assert sha(root/'spec.json')==seal['spec_sha256'] and sha(root/'pool/report.json')==seal['pool_sha256']
    assert sha(root/'cuda-batch/report.json')==seal['cuda_receipt_sha256']
    jobs={j['plan_index']:j for j in pool['jobs']};loaded=[];parents={}
    for v in spec['variants']:
        i=v['index'];case=v['context'];d=root/f'case-{i:02d}';meta=read(d/'cuda/report.json');proof=read(d/'cuda/cuda-verification.json')
        assert proof['state']=='passed' and proof['device']=='cuda' and meta['diagnostic_only'] and not meta['ppo_training_ready']
        # Older continuation reports put the explicit eligibility flag in the
        # verification receipt, with diagnostic_only/ppo_training_ready in the
        # outer metadata. Both must prohibit ordinary training.
        assert proof['ordinary_training_eligible'] is False
        if 'ordinary_training_eligible' in meta:assert meta['ordinary_training_eligible'] is False
        assert proof['diagnostic_continuation_version']=='combat_policy_continuation_rng_fork_v1'
        assert meta['model_sha256']==v['model_sha256']==sha(v['model']) and sha(d/'cuda/cuda-verification.json')==meta['cuda_verification_sha256']
        path=next((d/'native').glob('replay-*/steps.jsonl'));assert sha(path)==meta['source_sha256'][str(path.resolve())]
        steps=load_steps(path);roll=d/'cuda/rollout.jsonl';assert sha(roll)==meta['rollout_sha256'];rows={r['frame']:r for r in load_steps(roll)}
        if v['context_index'] not in parents:parents[v['context_index']]=load_steps(next((Path(case['source_case'])/'native').glob('replay-*/steps.jsonl')))
        r=read(Path(jobs[i]['root'])/'report.json')['results'][0];assert r['capture_valid'] and r['dispatch_valid'] and r['generated_fixture']==case['instance']
        f=case['frame'];at=next((s for s in steps if s['observation']['identity']['frame']==f and s['owner']=='provider'),None)
        assert at is not None
        feature_exact=f in rows and hashlib.sha256(json.dumps(rows[f]['features'],separators=(',',':')).encode()).hexdigest()==case['features_sha256']
        life=r['first_life'];goal=r['goal_stop'];end=goal['kill_frame'] if goal else life['end_frame'];damage=life['damage']
        active=[s for s in steps if f<=s['observation']['identity']['frame']<end and s['observation']['identity']['life']==1 and s['observation']['health']>0]
        window=[s for s in active if s['observation']['identity']['frame']<f+10]
        dispatched=all((s.get('server_execution') or {}).get('matched') and (s.get('server_execution') or {}).get('window_exclusive') and (s.get('server_execution') or {}).get('recovery_commands')==0 for s in window)
        owned_window=bool(window) and all(s['owner']=='provider' and s.get('sample') for s in window)
        remembered=case['observed_remembered_target'];fresh=next(m for m in at['observation']['remembered_threats'] if m['class']==remembered['class'])
        reacquired=next((s['observation']['identity']['frame'] for s in active if any(e['id']==fresh['id'] and e.get('clear_shot') for e in s['observation']['enemies'])),None)
        any_clear=next((s['observation']['identity']['frame'] for s in active if any(e.get('clear_shot') for e in s['observation']['enemies'])),None)
        fork=next((s.get('sample',{}).get('continuation_probe') for s in active if (s.get('sample') or {}).get('continuation_probe',{}).get('reset')),None)
        metrics=dict(goal=bool(goal),death=bool(r['death_stop']),end_frame=life['end_frame'],kills=sum(x['monster_kills'] for x in damage['by_mod']),
            dealt=sum(x['monster_health_damage'] for x in damage['by_mod']),received=damage['received_health_damage'],
            fully_policy_owned=all(s['owner']=='provider' for s in active),fallback_frames=sum(s['owner']!='provider' for s in active),
            reacquired_remembered_target_frame=reacquired,reacquisition_latency=None if reacquired is None else reacquired-f,
            first_any_clear_enemy_frame=any_clear)
        loaded.append(dict(v=v,steps=steps,at=at,feature_exact=feature_exact,owned_window=owned_window,dispatch=dispatched,fork=fork,metrics=metrics))
    results=[]
    for d in loaded:
        v=d['v'];case=v['context'];f=case['frame'];peers=[x for x in loaded if x['v']['context_index']==v['context_index'] and x['v']['replicate']==v['replicate']]
        control=next(x for x in peers if x['v']['label']=='control');zero=next(x for x in peers if x['v']['label']=='zero')
        prefix=equivalent(parents[v['context_index']],d['steps'],before=f)
        same_root=physical_equal_with_entity_bijection(control['at']['observation'],d['at']['observation'],{}, {})
        # Check through the native boundary observation, excluding asynchronous
        # stop-file/quit tails. Ownership metrics exclude the goal frame itself:
        # its observation can already show all enemies defeated.
        zero_exact=control['metrics']==zero['metrics'] and equivalent(control['steps'],zero['steps'],before=control['metrics']['end_frame']+1)
        draws=True
        if v['label']!='control':
            receipt=d['at']['sample']['movement_probe'];reference=zero['at']['sample']['movement_probe']
            draws=all(receipt[k]==reference[k] for k in ('gaussian_noise','fire_uniform'))
        other=next(x for x in loaded if x['v']['context_index']==v['context_index'] and x['v']['label']=='control' and x['v']['replicate']!=v['replicate'])
        distinct=control['fork'] is not None and other['fork'] is not None and control['fork']['seed']!=other['fork']['seed'] and control['fork']['gaussian_noise']!=other['fork']['gaussian_noise']
        admitted=prefix and same_root and zero_exact and d['feature_exact'] and d['owned_window'] and d['dispatch'] and draws and distinct
        results.append(dict(map=case['map'],seed=case['seed'],frame=f,label=v['label'],replicate=v['replicate'],future_policy_seed=v['fork_seed'],
            admitted=bool(admitted),prefix_exact=prefix,root_state_exact=same_root,selected_features_exact=d['feature_exact'],zero_control_exact=zero_exact,
            root_draws_exact=draws,window_owned=d['owned_window'],window_dispatch_exact=d['dispatch'],distinct_future_policy_streams=distinct,
            fork_observed=d['fork'] is not None,**d['metrics']))
    totals=[]
    for label in ('control','zero','toward','away'):
        rr=[x for x in results if x['label']==label]
        totals.append(dict(label=label,conditions=len(rr),wins=sum(x['goal'] for x in rr),kills=sum(x['kills'] for x in rr),dealt=sum(x['dealt'] for x in rr),
            received=sum(x['received'] for x in rr),ownership_failures=sum(not x['fully_policy_owned'] for x in rr),
            target_reacquired=sum(x['reacquired_remembered_target_frame'] is not None for x in rr)))
    a.out.mkdir(parents=True)
    report=dict(version='combat_contact_loss_probe_acceptance_v2_task_boundary',state='complete',admitted=sum(x['admitted'] for x in results),
        excluded=sum(not x['admitted'] for x in results),results=results,system_totals=totals,independent_worlds=3,future_policy_streams_per_world=2,
        native_engine_rng_forked=False,training_performed=False,promotion=False,
        scope='Diagnostic mean-shift effects on3TRAINroots with2futurepolicy streams. Full-fight outcomes include late fallback failures. Excluded causal comparisons cannot supply labels. Motion-conditioned aim/fire active; no pure movement-only command claim or broad superiority.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(admitted=report['admitted'],excluded=report['excluded'],totals=totals))


if __name__=='__main__':main()

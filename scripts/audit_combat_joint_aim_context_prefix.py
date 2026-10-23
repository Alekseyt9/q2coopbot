"""Native prefix and first intervention audit for a selected TRAIN replay."""
import argparse
import copy
import json
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha
from report_combat_machinegun_hits import scan
from report_registered_machinegun_outcomes import outcome


def normalized(v):
    if isinstance(v,dict):return {k:normalized(x) for k,x in v.items() if k not in ('spawncount','connection','observation_age_ms')}
    if isinstance(v,list):return [normalized(x) for x in v]
    return v


def source_latent(sample):
    return (sample.get('aim_candidate') or {}).get('source_latent',sample['latent'])


def category(sample):return (sample.get('aim_candidate') or {}).get('candidate',0)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('probe','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or expected==h,str(p);pins[str(p)]=h;return p
    root=a.probe.resolve();closed=read(pin(root/'complete.json'))
    assert closed['state']=='complete' and closed['native_cases']==closed['cuda_cases'] and not closed['quality_confirmation']
    spec=read(pin(root/'spec.json',closed['spec_sha256']));assert spec['split']=='train'
    assert closed['native_cases']==2*len(spec['selected']) and len(spec['selected']) in (1,2)
    models=[]
    for v in spec['variants']:
        folder=Path(v['update']);seal=read(pin(folder/'complete.json'))
        models.append(read(pin(folder/'weights.json',seal['weights_sha256'])))
    assert models[0]['aim_candidate_head']['predictor']==models[1]['aim_candidate_head']['predictor']
    for m in models:m['aim_candidate_head']['layers']=None
    assert models[0]==models[1],'Unexpected changes outside the stateless head'
    # AgeMS is absent from the base 1147 features, but it DOES enter the
    # separate head timing/predictor input. Earlier timing differences cannot
    # propagate through this feedforward head when chosen actions match.
    # At the first intervention, require identical timing explicitly as well
    # as identical base features, original draws and recurrent actor state.
    pool=read(pin(root/'pool/report.json',closed['pool_sha256']));assert pool['source_unchanged'] and pool['state']=='complete'
    metrics=read(pin(root/'native-metrics.json',closed['native_metrics_sha256']));assert metrics['split']=='train'
    corpus=Path(spec['source_corpus']);old_closed=read(pin(corpus/'complete.json'))
    old_pool=read(pin(corpus/'pool/report.json',old_closed['pool_sha256']));old_jobs={j['job']:j for j in old_pool['jobs']}
    old_processing=read(pin(corpus/'processing-train/report.json'))
    old_by_seed={c['seed']:old_jobs[c['job']] for c in old_processing['cases']}
    def load(directory,label,job,old=False):
        processing=directory/'processing-train' if old else directory/('processing-'+label)
        case=processing/f"case-{job['job']:04d}"
        meta=read(pin(case/'cuda/report.json'));verification=read(pin(case/'cuda/cuda-verification.json'))
        assert verification['state']=='passed' and verification['device']=='cuda'
        path=(case/'native/replay-0/steps.jsonl').resolve();pin(path,meta['source_sha256'][str(path)])
        rollout=pin(case/'cuda/rollout.jsonl',meta['rollout_sha256'])
        features={r['frame']:r['features'] for r in map(json.loads,rollout.open(encoding='utf-8-sig'))}
        capture=read(pin(Path(job['root'])/'report.json'));assert capture['capture_complete'] and capture['provenance_valid']
        log=pin(Path(capture['results'][0]['root'])/'server.log')
        result=[]
        for s in map(json.loads,path.open(encoding='utf-8-sig')):
            d=s.get('server_execution') or {};o=s['observation']
            if (s['owner']=='provider' and s.get('sample') and s.get('native_step') and o['identity']['life']==1 and o['health']>0
                    and d.get('matched') and d.get('window_exclusive') and d.get('recovery_commands')==0):result.append(s)
        assert result and len({s['observation']['identity']['frame'] for s in result})==len(result)
        return result,log,features
    pairs={}
    for job in pool['jobs']:
        assert not job['error'];label=spec['variants'][job['plan_index']]['label']
        pairs.setdefault(job['seed'],{})[label]=load(root,label,job)
    def shot_summary(s,log):
        n=s['native_step'];key=(n['spawncount'],n['begin_frame'],n['actor'],n['sequence'])
        with log.open(encoding='utf-8-sig') as stream:shots=[v for v in scan(stream,{key}) if v['eligible']]
        return dict(native_window=dict(begin=n['begin_frame'],end=n['end_frame'],sequence=n['sequence']),
            shots=[dict(outcome=outcome(v),target_class=v['fields']['target_class'],target=v['fields']['target']) for v in shots])
    results=[]
    for seed,pair in sorted(pairs.items()):
        source,source_log,sf=pair['source'];control,control_log,cf=pair['marginal'];old,_,of=load(corpus,'source',old_by_seed[seed],True)
        old_raw_equal=len(source)==len(old) and all(normalized(s['observation'])==normalized(o['observation']) for s,o in zip(source,old))
        # Historical runs can allocate different raw entity IDs. Require the
        # actual encoded policy inputs and executed decisions to reproduce,
        # and retain raw-observation compatibility as a separate result.
        # This does NOT relax the fresh pair's physical-prefix comparison.
        old_exact=len(source)==len(old) and sf==of and all(
            source_latent(s['sample'])==source_latent(o['sample']) and s['sample']['latent']==o['sample']['latent']
            and category(s['sample'])==category(o['sample']) and s['sent_command']==o['sent_command']
            and all(s['sample'].get(k,0)==o['sample'].get(k,0) for k in ('sampling_seed','attack','target','aim_mode','vertical','weapon')) for s,o in zip(source,old))
        prefix=0;first=None;problem=None;age_differences=[]
        for s,c in zip(source,control):
            frame=s['observation']['identity']['frame']
            if normalized(s['observation'])!=normalized(c['observation']):
                problem=dict(reason='observation differs before a category intervention',frame=s['observation']['identity']['frame'],
                    differing_keys=[k for k in s['observation'] if normalized(s['observation'][k])!=normalized(c['observation'].get(k))]);break
            if frame not in sf or frame not in cf or sf[frame]!=cf[frame]:
                problem=dict(reason='Base policy input features differ or missing before intervention',frame=frame);break
            ss,cs=s['sample'],c['sample']
            if source_latent(ss)!=source_latent(cs) or ss.get('memory')!=cs.get('memory') or any(ss.get(k,0)!=cs.get(k,0) for k in ('sampling_seed','target','aim_mode','vertical','weapon')):
                problem=dict(reason='Non-head draws differ',frame=s['observation']['identity']['frame']);break
            if category(ss)!=category(cs):
                st=(ss.get('aim_candidate') or {}).get('timing');ct=(cs.get('aim_candidate') or {}).get('timing')
                if st!=ct:
                    problem=dict(reason='Head timing differs at intervention',frame=frame,source_timing=st,marginal_timing=ct);break
                first=dict(frame=s['observation']['identity']['frame'],source_category=category(ss),marginal_category=category(cs),
                    source_attack=ss['attack'],marginal_attack=cs['attack'],native_command_changed=s['sent_command']!=c['sent_command'],
                    timing_equal_at_intervention=True,timing=st,base_features_and_actor_memory_equal=True,
                    source_shots=shot_summary(s,source_log),marginal_shots=shot_summary(c,control_log));break
            if s['sent_command']!=c['sent_command'] or ss['latent']!=cs['latent'] or ss['attack']!=cs['attack']:
                problem=dict(reason='Command differs without a category intervention',frame=s['observation']['identity']['frame']);break
            if s['observation']['observation_age_ms']!=c['observation']['observation_age_ms']:
                age_differences.append(dict(frame=frame,source=s['observation']['observation_age_ms'],marginal=c['observation']['observation_age_ms']))
            prefix+=1
        results.append(dict(seed=seed,source_reproduces_prior_train_actions_and_features=old_exact,matching_native_prefix_frames=prefix,
            historical_normalized_raw_observation_equal=old_raw_equal,
            earlier_age_differences_without_action_effect=age_differences,
            first_intervention=first,unexplained_prefix_mismatch=problem,
            controlled_intervention_verified=bool(old_exact and problem is None and first and first['native_command_changed'])))
    report=dict(state='complete',version='combat_joint_aim_context_prefix_v3',results=results,
        all_controlled_interventions_verified=all(r['controlled_intervention_verified'] for r in results),
        native_metrics=metrics['pairs'],inputs=pins,script_sha256=sha(__file__),new_games=False,neural_execution=False,
        training_performed=False,promotion=False,quality_confirmation=False,
        scope='Selected known TRAIN cases only. Fresh-pair base features, recurrent actor memory, original draws and physical command prefix match. Any earlier AgeMS timing differences have no chosen-action effect; the feedforward head/predictor timing MUST match at intervention. Historical source TRAIN actions/features must reproduce; its raw observation/entity-ID equality is reported separately, not silently asserted. Network generation identities normalized. No population quality claim or proof that useful context is absent elsewhere.')
    for p,h in pins.items():assert sha(Path(p))==h,p
    a.out.mkdir(parents=True);save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()

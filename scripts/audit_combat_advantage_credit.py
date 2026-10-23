"""Scalar TRAIN GAE/credit arithmetic only; no neural models or fitting."""
import argparse
import collections
import json
import math
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def mean(values):return math.fsum(values)/len(values) if values else None


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','native_audit','out'):ap.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();digest=sha(p);assert expected is None or digest==expected,str(p)
        pins[str(p)]=digest;return p
    closed=read(pin(a.native_audit/'complete.json'));native=read(pin(a.native_audit/'report.json',closed['report_sha256']))
    assert native['totals']['rows']>0 and not native['unassigned_native_effects']
    assert all(h['confirmed_goal'] for h in native['handoffs']) and native['spacing_telescoping_max_error']<1e-8
    config=read(pin(a.corpus/'training-config.json'));gamma,lam=config['gamma'],config['lambda']
    root=a.corpus/'processing-train';meta=read(pin(root/'merged/report.json'))
    rows=list(map(json.loads,pin(root/'merged/rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig')))
    assert len(rows)==meta['rows']==native['totals']['rows']
    facts={}
    for case in sorted(root.glob('case-*')):
        if not case.is_dir():continue
        cm=read(pin(case/'cuda/report.json'));roll=list(map(json.loads,pin(case/'cuda/rollout.jsonl',cm['rollout_sha256']).open(encoding='utf-8-sig')))
        seed=roll[0]['seed'];files=list((case/'native').glob('replay-*/steps.jsonl'));assert len(files)==1
        sp=files[0].resolve();rp=sp.with_name('rewards.jsonl');pin(sp,cm['source_sha256'][str(sp)]);pin(rp,cm['source_sha256'][str(rp)])
        steps={s['index']:s for s in map(json.loads,sp.open(encoding='utf-8-sig'))}
        for reward in map(json.loads,rp.open(encoding='utf-8-sig')):facts[(seed,reward['step'])]=(steps[reward['step']],reward)
    segments=[]
    for i,r in enumerate(rows):
        p=rows[i-1] if i else None
        consecutive=p is not None and p['seed']==r['seed'] and p['index']+1==r['index'] and p['next_frame']==r['frame'] and not p['terminal'] and not p['truncated']
        if not consecutive:segments.append([])
        segments[-1].append(i)
    assert len(segments)>=len(native['cases'])
    delta=[r['reward']+gamma*(0 if r['terminal'] else r['next_value'])-r['sample']['value'] for r in rows]
    advantages=[0.]*len(rows);closed_form=[0.]*len(rows);mc=[0.]*len(rows)
    for segment in segments:
        running=0.
        for i in reversed(segment):running=delta[i]+gamma*lam*running;advantages[i]=running
        for offset,i in enumerate(segment):
            closed_form[i]=math.fsum((gamma*lam)**lag*delta[j] for lag,j in enumerate(segment[offset:]))
        running=rows[segment[-1]]['next_value'] if not rows[segment[-1]]['terminal'] else 0.
        for i in reversed(segment):running=rows[i]['reward']+gamma*running;mc[i]=running
    error=max(abs(x-y) for x,y in zip(advantages,closed_form));assert error<1e-8
    center=mean(advantages);spread=math.sqrt(mean([(x-center)**2 for x in advantages]));normalized=[(x-center)/(spread+1e-8) for x in advantages]
    groups=collections.defaultdict(list);per_case=[];components=collections.defaultdict(list)
    names=set().union(*(facts[(r['seed'],r['index'])][1]['components'] for r in rows))
    for name in sorted(names|{'value_td'}):
        values=[gamma*(0 if r['terminal'] else r['next_value'])-r['sample']['value'] if name=='value_td' else facts[(r['seed'],r['index'])][1]['components'].get(name,0.) for r in rows]
        output=[0.]*len(rows)
        for segment in segments:
            running=0.
            for i in reversed(segment):running=values[i]+gamma*lam*running;output[i]=running
        components[name]=output
    assert max(abs(math.fsum(components[n][i] for n in components)-advantages[i]) for i in range(len(rows)))<1e-8
    for i,r in enumerate(rows):
        s,reward=facts[(r['seed'],r['index'])];c=reward['components'];annotation=r['sample'].get('aim_candidate')
        keys=['all','candidate_eligible' if annotation else 'candidate_ineligible',
            'observed_moving_speed_gt10' if math.hypot(*s['observation']['velocity'][:2])>10 else 'observed_stationary_speed_le10']
        if c['monster_kill']>0:keys.append('kill_effect_window')
        if c['death']<0:keys.append('death_effect_window')
        if annotation:keys.append('candidate_nonzero' if annotation['candidate'] else 'candidate_zero')
        if r['bootstrap_zero'] and r['truncated']:keys.append('completed_goal_handoff')
        for key in keys:groups[key].append(i)
    summary={key:dict(rows=len(ids),raw_advantage_mean=mean([advantages[i] for i in ids]),normalized_advantage_mean=mean([normalized[i] for i in ids]),
        normalized_positive_fraction=mean([float(normalized[i]>0) for i in ids]),mc_return_mean=mean([mc[i] for i in ids]),
        component_gae_mean={name:mean([v[i] for i in ids]) for name,v in components.items()}) for key,ids in groups.items()}
    for segment in segments:
        r=rows[segment[0]];ids=segment
        per_case.append(dict(seed=r['seed'],rows=len(ids),first_mc_return=mc[ids[0]],mean_normalized_advantage=mean([normalized[i] for i in ids]),
            zero_end_bootstrap=rows[ids[-1]]['bootstrap_zero'],end_terminal=rows[ids[-1]]['terminal']))
    for path,digest in pins.items():assert sha(Path(path))==digest
    a.out.mkdir(parents=True)
    report=dict(state='complete',version='combat_advantage_credit_arithmetic_v1',neural_execution=False,training_performed=False,new_games=False,
        rows=len(rows),segments=len(segments),gamma=gamma,lambda_=lam,gae_closed_form_max_error=error,
        gae_mean=center,gae_std=spread,groups=summary,cases=per_case,
        zero_end_bootstrap_segments=sum(c['zero_end_bootstrap'] for c in per_case),retained_horizon_bootstraps=sum(not c['zero_end_bootstrap'] for c in per_case),
        gamma_lambda_future_td_weights={str(lag):(gamma*lam)**lag for lag in (1,10,30,50,100)},
        inputs=pins,script_sha256=sha(__file__),
        scope='Existing native scalar values and bootstraps only. Exact GAE/closed-form and linear component decomposition; no NN replay or gradient/update. Group signs do not establish causal credit, unbiased baselines or optimal reward.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(rows=len(rows),segments=len(segments),groups={k:{n:v for n,v in g.items() if n!='component_gae_mean'} for k,g in summary.items()}))


if __name__=='__main__':main()

"""Audit common-state native outcomes; fixed TRAIN grid is not quality evidence."""
import argparse
import collections
import json
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha
from audit_combat_joint_aim_context_prefix import normalized, source_latent, category
from report_combat_machinegun_hits import scan
from report_registered_machinegun_outcomes import outcome


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('probe','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();root=a.probe.resolve();pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    closed=read(pin(root/'complete.json'));assert closed['native_cases']==closed['cuda_cases']==25
    assert not closed['training_performed'] and not closed['quality_confirmation']
    spec=read(pin(root/'spec.json',closed['spec_sha256']));assert spec['split']=='train' and spec['frame']==146
    prepared=Path(spec['prepared']);prep=read(pin(prepared/'report.json',spec['prepared_report_sha256']))
    pool=read(pin(root/'pool/report.json',closed['pool_sha256']));assert pool['state']=='complete' and pool['source_unchanged']
    for receipt in closed['cuda_receipts']:
        r=read(pin(receipt['report'],receipt['report_sha256']));assert r['state']=='complete' and r['device']=='cuda'
        for t in r['tasks']:pin(Path(t['out'])/'report.json',t['report_sha256'])
    source=read(pin(Path(prep['source'])/'weights.json',prep['source_weights_sha256']))
    for v in spec['variants']:
        m=read(pin(v['model'],v['model_sha256']));m['aim_candidate_head']['layers']=None
        s=json.loads(json.dumps(source));s['aim_candidate_head']['layers']=None;assert m==s
    def load(directory,job):
        meta=read(pin(directory/'cuda/report.json'));verification=read(pin(directory/'cuda/cuda-verification.json'))
        assert verification['state']=='passed' and verification['device']=='cuda'
        steps=pin(directory/'native/replay-0/steps.jsonl');assert meta['source_sha256'][str(steps)]==sha(steps)
        rollout=pin(directory/'cuda/rollout.jsonl',meta['rollout_sha256'])
        features={r['frame']:r['features'] for r in map(json.loads,rollout.open(encoding='utf-8-sig'))}
        rows=[]
        for s in map(json.loads,steps.open(encoding='utf-8-sig')):
            o=s['observation'];d=s.get('server_execution') or {}
            if (s['owner']=='provider' and s.get('sample') and s.get('native_step') and o['identity']['life']==1
                    and o['health']>0 and d.get('matched') and d.get('window_exclusive') and d.get('recovery_commands')==0):rows.append(s)
        frames=[r['observation']['identity']['frame'] for r in rows];assert len(frames)==len(set(frames)) and 146 in frames
        capture=read(pin(Path(job['root'])/'report.json'));assert capture['capture_complete'] and capture['provenance_valid']
        result=capture['results'][0];assert result['capture_valid'] and result['dispatch_valid'] and result['seed_confirmed']
        assert result['seed']==spec['seed'] and result['generated_fixture']['split']=='train'
        log=pin(Path(result['root'])/'server.log');return rows,features,log,result
    jobs={j['plan_index']:j for j in pool['jobs']};assert set(jobs)==set(range(25))
    loaded={k:load(root/f'case-{k:02d}',j) for k,j in jobs.items()}
    baseline,bf,_,br=loaded[0];reference=Path(spec['reference']);refclosed=read(pin(reference/'complete.json'))
    refpool=read(pin(reference/'pool/report.json',refclosed['pool_sha256']));refjob=next(j for j in refpool['jobs'] if j['plan_index']==1)
    historical,hf,_,_=load(reference/'processing-marginal'/f"case-{refjob['job']:04d}",refjob)
    prefix=lambda rows:[r for r in rows if r['observation']['identity']['frame']<=146]
    baseline=prefix(baseline);historical=prefix(historical)
    history_exact=len(baseline)==len(historical) and all(
        bf[s['observation']['identity']['frame']]==hf[h['observation']['identity']['frame']]
        and s['sent_command']==h['sent_command'] and source_latent(s['sample'])==source_latent(h['sample'])
        and s['sample']['latent']==h['sample']['latent'] and category(s['sample'])==category(h['sample'])
        and s['sample'].get('memory')==h['sample'].get('memory')
        and all(s['sample'].get(n,0)==h['sample'].get(n,0) for n in ('sampling_seed','attack','target','aim_mode','vertical','weapon'))
        for s,h in zip(baseline,historical))
    results=[]
    for k,(rows,features,log,result) in sorted(loaded.items()):
        current=prefix(rows);issues=[];matched=0
        if len(current)!=len(baseline):issues.append('prefix length differs')
        if result['generated_fixture']!=br['generated_fixture']:issues.append('fixture differs')
        for s,b in zip(current,baseline):
            frame=s['observation']['identity']['frame'];ss=s['sample'];bs=b['sample']
            if normalized(s['observation'])!=normalized(b['observation']):issues.append(f'physical observation differs at {frame}');break
            if features[frame]!=bf[frame]:issues.append(f'encoded input differs at {frame}');break
            if source_latent(ss)!=source_latent(bs) or ss.get('memory')!=bs.get('memory') or any(
                    ss.get(n,0)!=bs.get(n,0) for n in ('sampling_seed','target','aim_mode','vertical','weapon')):
                issues.append(f'non-head draws/state differ at {frame}');break
            if frame==146:
                if category(ss)!=k:issues.append(f'category is {category(ss)} rather than {k}')
                if (ss.get('aim_candidate') or {}).get('timing')!=(bs.get('aim_candidate') or {}).get('timing'):
                    issues.append('head timing differs at intervention')
            elif s['sent_command']!=b['sent_command'] or ss['latent']!=bs['latent'] or ss['attack']!=bs['attack'] or category(ss)!=0:
                issues.append(f'executed prefix differs at {frame}');break
            else:matched+=1
        target=next(s for s in current if s['observation']['identity']['frame']==146);n=target['native_step']
        key=(n['spawncount'],n['begin_frame'],n['actor'],n['sequence'])
        with log.open(encoding='utf-8-sig') as stream:shots=[q for q in scan(stream,{key}) if q['eligible']]
        target_index=target['sample']['target'];enemies=target['observation']['enemies']
        selected=enemies[target_index-1] if 1<=target_index<=len(enemies) else None
        shot_records=[dict(outcome=outcome(q),fields=q['fields'],damage=q['damage'],
            hits_selected_observed_enemy=bool(selected and int(q['fields']['target'])==selected['id'])) for q in shots]
        damage=result['first_life']['damage']
        results.append(dict(category=k,offset_yaw_degrees=spec['variants'][k]['offset_yaw_degrees'],
            offset_pitch_degrees=spec['variants'][k]['offset_pitch_degrees'],common_state_verified=history_exact and not issues,
            matching_prefix_frames=matched,issues=issues,attack=target['sample']['attack'],shots=shot_records,
            selected_target=target_index,selected_enemy=selected,
            battle=dict(goal=result['goal_stop'],deaths=int(result['death_stop']),
                kills=sum(v['monster_kills'] for v in damage['by_mod']),received=damage['received_health_damage'])))
    accepted=[r for r in results if r['common_state_verified']]
    counts=collections.Counter(q['outcome'] for r in accepted for q in r['shots'])
    report=dict(state='complete',version='combat_joint_aim_grid_native_audit_v1',baseline_reproduces_prior_actions_features=history_exact,
        all_common_states_verified=len(accepted)==25,accepted_categories=len(accepted),results=results,outcomes=dict(counts),
        inputs=pins,script_sha256=sha(__file__),training_performed=False,quality_confirmation=False,promotion=False,
        scope='One selected TRAIN world state and fixed seeded spread. Native outcomes, not predicted labels or population accuracy. Later battle outcomes are descriptive, not one-step causal labels.')
    for p,h in pins.items():assert sha(Path(p))==h,p
    a.out.mkdir(parents=True);save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(json.dumps(dict(accepted=len(accepted),historical_baseline_exact=history_exact,outcomes=dict(counts),
        cases=[dict(category=r['category'],verified=r['common_state_verified'],issues=r['issues'],shots=[q['outcome'] for q in r['shots']]) for r in results]),indent=2))


if __name__=='__main__':main()

"""Separate policy-task failures from recording faults; descriptive discovery only."""
import argparse
import collections
import json
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('collection','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();root=a.collection.resolve();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or h==expected,str(path);pins[str(path)]=h;return path
    pin(__file__);closed=read(pin(root/'complete.json'));assert closed['native_cases']==closed['cuda_cases']==76
    spec=read(pin(root/'spec.json',closed['spec_sha256']));pool=read(pin(root/'pool/report.json',closed['pool_sha256']));assert pool['source_unchanged']
    jobs={j['plan_index']:j for j in pool['jobs']};records=[];totals=collections.Counter()
    for unit in spec['contexts']:
        ps=root/'proofs'/unit['unit_id'];seal=read(pin(ps/'complete.json'));proof=read(pin(ps/'report.json',seal['report_sha256']))
        c=unit['context'];accepted={r['label']:r for r in proof['results']}
        controls=bool(proof['nofork_full_exact'] and proof.get('zero_fork_task_exact',proof['zero_fork_full_exact']) and proof['fork_prefix_exact'])
        for variant in unit['variants']:
            label=variant['label']
            if label not in accepted:continue
            original=accepted[label];job=jobs[variant['plan_index']]
            capture=read(pin(Path(job['root'])/'report.json'));assert capture['capture_complete'] and capture['provenance_valid']
            r=capture['results'][0];assert r['dispatch_valid'];goal=r['goal_stop'];death=r['death_stop']
            stop=goal['kill_frame'] if goal else r['first_life']['end_frame']
            folder=root/f"case-{variant['plan_index']:02d}";meta=read(pin(folder/'cuda/report.json'))
            v=read(pin(folder/'cuda/cuda-verification.json',meta['cuda_verification_sha256']));assert v['state']=='passed' and v['device']=='cuda'
            sp=next((folder/'native').glob('replay-*/steps.jsonl'));pin(sp,meta['source_sha256'][str(sp.resolve())])
            ep=sp.with_name('server_outcomes.jsonl');pin(ep,meta['source_sha256'][str(ep.resolve())])
            effects={e['step']:e for e in map(json.loads,ep.open(encoding='utf-8-sig'))}
            first_rules=None;goal_owner_unknown=False;owners=collections.Counter()
            with sp.open(encoding='utf-8-sig') as stream:
                for step in map(json.loads,stream):
                    o=step['observation'];f=o['identity']['frame']
                    if o['identity']['life']!=1 or o['health']<=0 or f<c['frame']:continue
                    if f>=stop:
                        if goal and f==stop and step['owner']=='rules':
                            effect=effects.get(step['index'])
                            if effect and effect['available'] and effect['monster_kills']>0:first_rules=f if first_rules is None else first_rules
                            elif effect is None or not effect['available']:goal_owner_unknown=True
                        continue
                    owners[step['owner']]+=1
                    if step['owner']=='rules' and first_rules is None:first_rules=f
            alive_goal=bool(goal and goal['health']>0 and not death)
            outcome='handoff_before_task_end' if first_rules is not None else 'goal_ownership_unconfirmed' if alive_goal and goal_owner_unknown else 'policy_alive_goal' if alive_goal else 'policy_death' if death else 'budget_without_goal'
            prefix=controls and all(original[k] for k in ('prefix_exact','intervention_state_exact','intervention_draws_exact','categories_exact','shared_future_draws_exact'))
            record=dict(context_id=c['context_id'],map=c['instance']['map'],world_seed=c['seed'],stream_seed=proof['fork']['seed'],label=label,
                outcome=outcome,original_admitted=original['admitted'],control_and_prefix_exact=prefix,first_rules_frame=first_rules,
                observed_post_root_frames=dict(owners),native_alive_goal=alive_goal,death_observed=bool(death),
                full_task_terminal_known=bool(goal or death),goal_boundary_ownership_unknown=goal_owner_unknown,policy_self_completion=outcome=='policy_alive_goal')
            records.append(record);totals[outcome]+=1
            if prefix:totals['prefix_exact_'+outcome]+=1
            if not original['admitted']:totals['original_excluded_'+outcome]+=1
    assert len(records)==60
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True);save(a.out/'records.json',dict(records=records))
    save(a.out/'report.json',dict(state='complete',version='combat_contact_outcome_ledger_v1',counts=dict(totals),records_sha256=sha(a.out/'records.json'),inputs=pins,
        discovery_only=True,original_gates_unchanged=True,new_action_selection=False,training_performed=False,ordinary_training_eligible=False,promotion=False,
        scope='Descriptive finite-budget policy self-completion ledger, not a replacement admission gate or confirmation. Handoff is an observed policy-task failure, not necessarily corrupt transport. Budget_without_goal means not completed in the observed budget, not invented eventual death or completion. Rules outcomes after handoff are never credited to the learned policy.'))
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json'),records_sha256=sha(a.out/'records.json')));print(dict(totals))

if __name__=='__main__':main()

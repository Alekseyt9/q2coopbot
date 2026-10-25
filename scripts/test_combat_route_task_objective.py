"""Real recorded task boundaries plus negative metadata counterexamples; no NN."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from combat_route_task_objective import collect_evidence,overlay,qualifies,task_advantages,CONTINUITY,AUTONOMOUS
from ppo_combat import advantages
from evaluate_registered_combat_models import read,save


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args();assert not a.out.exists()
    root=a.capture.resolve();spec=read(root/'spec.json');rows=[]
    for v in spec['variants']:
        if v['label']=='zero':
            rows.extend(json.loads(s) for s in (root/f'case-{v["index"]:02d}/cuda/rollout.jsonl').open())
    digest=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
    initial=digest(rows);evidence,pins=collect_evidence(root,rows)
    control,unchanged=overlay(rows,evidence,CONTINUITY)
    assert control==rows and not unchanged
    updated,boundaries=overlay(rows,evidence,AUTONOMOUS)
    assert len(boundaries)==2 and digest(rows)==initial
    by_key=lambda r:(r['seed'],r['index'],r['frame'],r['next_frame'])
    chosen={tuple(p['identity']) for p in evidence if any(by_key(r)==tuple(p['identity']) and qualifies(r,p) for r in rows)}
    assert len(chosen)==2
    baseline,_=task_advantages(control,.99,.95,advantages)
    adapted,_=task_advantages(updated,.99,.95,advantages)
    for i,(old,new) in enumerate(zip(rows,updated)):
        assert all(old[k]==new[k] for k in old), 'Native row was relabeled'
        if by_key(old) in chosen:
            assert new['task_terminal'] and new['task_bootstrap_zero']
            assert abs((adapted[i]-baseline[i])+.99*old['next_value'])<1e-6
        if old['seed'] not in {r['seed'] for r in boundaries}:assert adapted[i]==baseline[i]
    key=next(iter(chosen));row=next(r for r in rows if by_key(r)==key);proof=next(p for p in evidence if tuple(p['identity'])==key)
    rejected=0
    for field,value in [('capture_verified',False),('dispatch_verified',False),('same_life',False),
        ('owner_before','rules'),('owner_after','provider'),('fallback','system2_noncombat'),
        ('health_before',0),('health_after',0),('goal_before_or_at_handoff',True),
        ('death_before_or_at_handoff',True),('episode_goal',True),('unfinished_native_fixture',False)]:
        invalid=copy.deepcopy(proof);invalid[field]=value
        assert not qualifies(row,invalid),field
        _,changed=overlay([row],[invalid],AUTONOMOUS);assert not changed;rejected+=1
    for field,value in [('terminal',True),('truncated',False),('bootstrap_zero',True),('next_reset',True),('seed',-1)]:
        invalid=copy.deepcopy(row);invalid[field]=value
        assert not qualifies(invalid,proof),field
        rejected+=1
    assert digest(rows)==initial
    save(a.out,dict(state='passed',rows=len(rows),verified_task_boundaries=len(boundaries),
        metadata_counterexamples_rejected=rejected,continuity_identity=True,native_labels_preserved=True,
        exact_tail_credit_difference=True,inputs=pins,training_performed=False,neural_execution=False))
    print(dict(rows=len(rows),boundaries=len(boundaries),rejected_counterexamples=rejected),flush=True)


if __name__=='__main__':main()

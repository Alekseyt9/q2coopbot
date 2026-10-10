"""Audit remembered threat provenance from own closed client traces only."""
import argparse
import json
import math
import pathlib
from process_combat_architecture_pool import read, save, sha


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pool',type=pathlib.Path,required=True)
    a=ap.parse_args()
    report=read(a.pool/'report.json')
    assert report['accepted']==report['completed']
    cases=[]
    for case in report['results']:
        path=pathlib.Path(case['root'])/'PairLearner.jsonl'
        observed={}
        checked=long_continuation=0
        max_age=0
        for line in path.read_text().splitlines():
            r=json.loads(line)
            c=r.get('combat_policy',{})
            o=c.get('observation')
            if o is None: continue
            identity=o['identity']
            frame=identity['frame']
            key=tuple(identity[k] for k in ('map','connection','spawncount','actor','life'))
            for e in o['enemies']:
                if e.get('clear_shot') is not True: continue
                observed[(key,frame,e['id'])]=(e,[x+y for x,y in zip(o['position'],e['relative'])])
            memory=o.get('remembered_threats',[])
            assert len(memory)<=8 and len({e['id'] for e in memory})==len(memory)
            for e in memory:
                age=e['age_frames']
                assert 0<=age<=200
                original,at=observed[(key,frame-age,e['id'])]
                assert original['class']==e['class'] and original.get('observed_model','')==e['observed_model']
                expected=[x-y for x,y in zip(at,o['position'])]
                assert all(math.isclose(x,y,abs_tol=1e-6) for x,y in zip(expected,e['last_observed_relative']))
                assert original.get('observed_velocity')==e['last_observed_velocity']
                max_age=max(max_age,age)
                checked+=1
            if not o['enemies'] and c.get('selection',{}).get('owner')=='provider' and any(e['age_frames']>30 for e in memory):
                long_continuation+=1
        cases.append(dict(seed=case['seed'],checked_records=checked,max_age_frames=max_age,
            provider_occluded_frames_beyond_old_grace=long_continuation,trace_sha256=sha(path)))
    assert sum(c['checked_records'] for c in cases)>0
    save(a.pool/'threat-memory-audit.json',dict(version='coop_threat_memory_audit_v1',state='passed',
        checked_records=sum(c['checked_records'] for c in cases),
        provider_occluded_frames_beyond_old_grace=sum(c['provider_occluded_frames_beyond_old_grace'] for c in cases),
        cases=cases,scope='Own previously visible positions/velocities only, no extrapolation; explicit age and life binding. No hidden/native positions used. No quality claim.'))


if __name__=='__main__':main()

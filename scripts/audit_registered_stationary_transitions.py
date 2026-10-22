"""Observed next-frame stationary associations with executed movement/guards.

Current observation predates its command; count the NEXT observation instead.
Association is not proof that a guard caused every stationary outcome.
"""
import argparse
import collections
import json
import math
import pathlib
from process_combat_architecture_pool import read,save,sha
from process_registered_combat_pool_cuda import select_learned_jobs
from native_combat_release import release_boundary


def classify(before, after):
    o,n=before['observation'],after['observation'];a,p=before['applied'],before['proposed']
    if any(o['identity'][k]!=n['identity'][k] for k in ('map','connection','spawncount','actor','life')) or n['identity']['frame']!=o['identity']['frame']+1:
        return None
    if n['health']<=0 or not any(e.get('clear_shot') for e in n.get('enemies') or []):return None
    if math.hypot(*n['velocity'][:2])>=10:return None
    requested=math.hypot(p['forward'],p['side'])>.1 or p['vertical']!='release'
    applied=math.hypot(a['forward'],a['side'])>.1 or a['vertical']!='release'
    kind='no_motion_intent' if not requested else ('applied_motion_next_stationary' if applied else 'suppressed_motion_next_stationary')
    reasons=sorted({i['reason'] for i in (before.get('selection') or {}).get('interventions') or [] if i['component'] in ('movement','vertical')})
    return kind,reasons,math.hypot(n['position'][0]-o['position'][0],n['position'][1]-o['position'][1])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pool',type=pathlib.Path,required=True)
    ap.add_argument('--plan-index',type=int,required=True);ap.add_argument('--out',type=pathlib.Path,required=True)
    args=ap.parse_args();assert not args.out.exists();pins={}
    def pin(path):
        path=pathlib.Path(path).resolve();pins[str(path)]=sha(path);return path
    pool=read(pin(args.pool/'report.json'));cases=[]
    for job in select_learned_jobs(pool,args.plan_index):
        root=pathlib.Path(job['root']);capture=read(pin(root/'report.json'));manifest=read(pin(root/'manifest.json'))
        assert capture['capture_complete'] and capture['provenance_valid'] and len(capture['results'])==1
        result=capture['results'][0];assert result['capture_valid'] and result['dispatch_valid'] and result['seed_confirmed']
        worker=pathlib.Path(result['root'])
        with pin(worker/'server.log').open(encoding='utf-8-sig') as f:generation,release,_=release_boundary(f,manifest,job['seed'])
        counts=collections.Counter();reasons=collections.Counter();previous=None;displacement=0.
        with pin(worker/'bot.jsonl').open(encoding='utf-8-sig') as stream:
            for row in map(json.loads,stream):
                c=row.get('combat_policy') or {};o=c.get('observation') or {};i=o.get('identity') or {}
                if (c.get('selection') or {}).get('owner')!='provider' or i.get('spawncount')!=generation or i.get('frame',-1)<release or i.get('life')!=1 or o['health']<=0:
                    previous=None;continue
                if previous is not None:
                    classified=classify(previous,c)
                    if classified:
                        kind,labels,delta=classified;counts[kind]+=1;counts['stationary_clear_next_frames']+=1
                        displacement+=delta
                        reasons.update(kind+'/'+x for x in labels or ['no_recorded_motion_intervention'])
                previous=c
        cases.append(dict(job=job['job'],seed=job['seed'],counts=dict(counts),reasons=dict(reasons),
            stationary_next_displacement_sum=displacement))
    totals=dict(sum((collections.Counter(c['counts']) for c in cases),collections.Counter()))
    reasons=dict(sum((collections.Counter(c['reasons']) for c in cases),collections.Counter()))
    for p,h in pins.items():assert sha(p)==h
    save(args.out,dict(state='complete',version='registered_stationary_transitions_v1',cases=cases,totals=totals,reasons=reasons,
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),training_performed=False,neural_execution=False,
        scope='Adjacent first-life provider transitions after actual native release. Speed<10 with visible enemy in NEXT observation, associated with previous proposed/applied command and its recorded interventions. Current-state immobility is not attributed to a future command. No counterfactual causal proof; endpoint velocity can be low despite displacement.'))
    print(json.dumps(dict(totals=totals,reasons=reasons)))


if __name__=='__main__':main()

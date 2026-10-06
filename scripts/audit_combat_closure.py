"""Audit native transition closure and GAE without changing training inputs."""
import argparse, collections, json, math, pathlib
from ppo_combat import advantages, sha

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def lines(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines()]

def audit(cycle):
    config=read(cycle/'config.json'); result=[]; totals=collections.Counter()
    for iteration in sorted(cycle.glob('iteration-*')):
        data=iteration/'rollout'; meta=read(data/'report.json')
        for path,digest in meta['source_sha256'].items():
            assert sha(pathlib.Path(path))==digest, f'Changed input: {path}'
        rows=lines(data/'rollout.jsonl'); assert sha(data/'rollout.jsonl')==meta['rollout_sha256']
        assert len(rows)==meta['rows']
        lookup={(r['seed'],r['index']):r for r in rows}; assert len(lookup)==len(rows)
        steps={}
        for replay in data.glob('replay-*'):
            ss=lines(replay/'steps.jsonl'); rr=lines(replay/'rewards.jsonl')
            assert len(ss)==len(rr)
            for s,reward in zip(ss,rr):
                seed=int(s['episode'].split('-')[-1]); key=(seed,s['index'])
                assert key not in steps; steps[key]=(s,reward)
        eligible={key for key,(s,r) in steps.items() if s['observation']['identity']['life']==1 and s['owner']=='provider' and s.get('sample') and s.get('next_observation') and r['available'] and r['score'] is not None}
        assert eligible==set(lookup), 'Native eligible transitions missing or extra'
        counts=collections.Counter(); endpoints=[]
        for key,(s,reward) in steps.items():
            if s['observation']['identity']['life']!=1 or s['owner']!='provider':continue
            if key not in lookup:
                counts['excluded:'+s.get('reason','unavailable')]+=1
                continue
            r=lookup[key]; n=s['next_observation']
            assert r['reward']==reward['score'] and r['terminal']==s['terminal'] and r['truncated']==s['truncated']
            assert r['sample']==s['sample'] and r['frame']==s['observation']['identity']['frame'] and r['next_frame']==n['identity']['frame']
            assert all(math.isfinite(r[k]) for k in ('reward','next_value'))
            assert s['terminal']==(n['health']<=0)
            if s['terminal']:
                assert r['next_value']==0 and reward['components']['death']<0
                counts['death_transitions']+=1
            if s['truncated']:
                assert s['reason']=='control_handoff' and r['next_value']==0
                counts['handoff_transitions']+=1
            if reward['components'].get('monster_kill',0)>0:counts['consumed_kill_transitions']+=1
            if r['index']==max(x['index'] for x in rows if x['seed']==r['seed']):
                endpoints.append(dict(seed=r['seed'],frame=r['frame'],next_frame=r['next_frame'],terminal=r['terminal'],truncated=r['truncated'],next_value=r['next_value']))
            counts['consumed_rows']+=1
        actual,returns=advantages(rows,config['gamma'],config['lambda'])
        # Independent forward expansion of each GAE sum; stops at gaps/seeds,
        # terminal/handoff or the end of this frozen batch, but keeps V(next).
        for i,r in enumerate(rows):
            value=0.; factor=1.; j=i
            while True:
                q=rows[j];delta=q['reward']+config['gamma']*(0 if q['terminal'] else q['next_value'])-q['sample']['value']
                value+=factor*delta
                if q['terminal'] or q['truncated'] or j+1==len(rows):break
                following=rows[j+1]
                if following['seed']!=q['seed'] or following['index']!=q['index']+1 or following['frame']!=q['next_frame']:break
                factor*=config['gamma']*config['lambda'];j+=1
            assert math.isclose(value,actual[i],rel_tol=1e-9,abs_tol=1e-9)
            assert math.isclose(returns[i],value+r['sample']['value'],rel_tol=1e-9,abs_tol=1e-9)
        totals.update(counts);result.append(dict(iteration=iteration.name,counts=dict(counts),endpoints=endpoints))
    assert result
    return dict(totals=dict(totals),iterations=result,scope='Exact eligible native transition/reward/sample preservation; death and handoff zero bootstrap; independent GAE expansion including horizon/gaps. Incomplete final command has no next observation and is excluded; last complete transition keeps V(next). No independent critic recomputation or claim of optimal handoff objective.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cycle',type=pathlib.Path,action='append',required=True);ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args()
    report={str(p):audit(p) for p in a.cycle}
    with a.out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({p:r['totals'] for p,r in report.items()},indent=2))

if __name__=='__main__':main()

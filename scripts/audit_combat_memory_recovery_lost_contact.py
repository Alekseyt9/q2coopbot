"""Describe lost contact in one completed matched probe; no neural execution."""
import argparse
import collections
import json
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from native_combat_release import release_boundary

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evaluation',type=Path,required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();root=a.evaluation.resolve();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path)
        assert expected is None or h==expected,str(path)
        pins[str(path)]=h;return path
    closed=read(pin(root/'complete.json'));assert closed['native_cases']==32 and closed['cuda_cases']==24
    pool=read(pin(root/'pool/report.json',closed['pool_sha256']));assert pool['source_unchanged']
    fixture=read(pin(root/'fixture-proof.json'));labels={i:Path(p['path']).stem.removesuffix('-plan') for i,p in enumerate(fixture['plans'])}
    variants={};pin(__file__)
    for pi,label in labels.items():
        if label=='rules':continue
        matches=[j for j in pool['jobs'] if j['plan_index']==pi and j['seed']==a.seed];assert len(matches)==1
        job=matches[0];capture=read(pin(Path(job['root'])/'report.json'));r=capture['results'][0]
        assert capture['capture_complete'] and capture['provenance_valid'] and r['dispatch_valid']
        manifest=read(pin(Path(job['root'])/'manifest.json'));log=pin(Path(r['root'])/'server.log')
        with log.open(encoding='utf-8-sig') as stream:_,release,_=release_boundary(stream,manifest,job['seed'])
        folder=root/f'processing-{label}'/f"case-{job['job']:04d}";meta=read(pin(folder/'cuda/report.json'))
        proof=read(pin(folder/'cuda/cuda-verification.json',meta['cuda_verification_sha256']));assert proof['state']=='passed'
        sp=next((folder/'native').glob('replay-*/steps.jsonl'));pin(sp,meta['source_sha256'][str(sp.resolve())])
        stop=r['goal_stop']['kill_frame'] if r['goal_stop'] else r['first_life']['end_frame']
        steps=[s for s in map(json.loads,sp.open(encoding='utf-8-sig')) if s['owner']=='provider' and s.get('sample') and s['observation']['identity']['life']==1 and s['observation']['health']>0 and release<=s['observation']['identity']['frame']<stop]
        c=collections.Counter();run=longest=0
        for s in steps:
            o=s['observation'];mem=sorted(o.get('remembered_threats') or [],key=lambda t:(t['age_frames'],t['id']))
            visible=bool(o['enemies']);fresh=bool(not visible and mem and mem[0]['age_frames']<=10)
            c['provider_frames']+=1;c['visible_frames']+=visible;c['fresh_memory_frames']+=fresh
            c['stale_memory_frames']+=bool(not visible and mem and not fresh);c['empty_frames']+=bool(not visible and not mem)
            run=0 if visible else run+1;longest=max(run,longest)
            large=abs(s['applied_action']['yaw_delta_degrees'])>=30
            c['large_yaw_frames']+=large;c['large_yaw_fresh_frames']+=bool(large and fresh)
        o=steps[-1]['observation'];mem=sorted(o.get('remembered_threats') or [],key=lambda t:(t['age_frames'],t['id']))
        variants[label]=dict(counts=dict(c),longest_unseen_provider_frames=longest,final_unseen_provider_frames=run,
            last_memory_age_frames=mem[0]['age_frames'] if mem else None,last_visible_enemies=len(o['enemies']),
            alive_goal=bool(r['goal_stop'] and not r['death_stop'] and r['goal_stop']['health']>0),death=bool(r['death_stop']),
            received=r['first_life']['damage']['received_health_damage'])
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True)
    save(a.out/'report.json',dict(state='complete',version='combat_memory_recovery_lost_contact_audit_v1',seed=a.seed,variants=variants,inputs=pins,
        neural_execution=False,new_games=False,training_performed=False,promotion=False,
        scope='First-life provider windows before objective/death only. Descriptive closed-loop diagnosis; unequal trajectories are not proof of a unique causal mechanism. Hidden enemy positions are not used.'))
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')));print(variants)

if __name__=='__main__':main()

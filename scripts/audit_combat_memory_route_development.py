"""Primary full-cohort outcomes; ownership failures stay in the denominator."""
import argparse
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from audit_combat_contact_loss_probe import load_steps


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--screen',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args();assert not a.out.exists()
    root=a.screen.resolve();spec,pool=read(root/'spec.json'),read(root/'pool/report.json')
    assert spec['version']=='combat_memory_route_development_screen_v1' and spec['independent_worlds']==4
    assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==24
    assert read(root/'cuda-batch/report.json')['state']=='complete'
    jobs={j['plan_index']:j for j in pool['jobs']};results=[]
    for v in spec['variants']:
        i=v['plan_index'];capture=Path(jobs[i]['root']);rpt=read(capture/'report.json');r=rpt['results'][0]
        assert rpt['capture_complete'] and rpt['provenance_valid'] and r['capture_valid'] and r['dispatch_valid']
        assert r['seed_confirmed'] and r['seed']==v['seed'] and r['generated_fixture']==v['instance']
        life=r['first_life'];damage=life['damage'];fallback=None;rows=None;route_active=None
        if v['mode']=='learned':
            d=root/f'case-{i:02d}';meta=read(d/'cuda/report.json');proof=read(d/'cuda/cuda-verification.json')
            assert meta['model_sha256']==sha(v['model']) and proof['state']=='passed' and proof['device']=='cuda'
            assert sha(d/'cuda/cuda-verification.json')==meta['cuda_verification_sha256']
            if v['controller']=='candidate':
                rollout=load_steps(d/'cuda/rollout.jsonl')
                route_active=sum(r['features'][0]>0 and r['features'][1489]==1
                    and not any(r['features'][73+12*j] for j in range(8))
                    and any(r['features'][881+30*j]==1 and r['features'][1490+8*j]==1
                            and r['features'][1491+8*j]==1 for j in range(8)) for r in rollout)
            steps=load_steps(next((d/'native').glob('replay-*/steps.jsonl')))
            end=r['goal_stop']['kill_frame'] if r['goal_stop'] else life['end_frame']
            active=[s for s in steps if s['observation']['identity']['life']==1 and s['observation']['health']>0 and s['observation']['identity']['frame']<end]
            fallback=sum(s['owner']!='provider' for s in active);rows=len(active)
        results.append(dict(map=v['map'],seed=v['seed'],loadout=v['label'],controller=v['controller'],
            goal=bool(r['goal_stop']),death=bool(r['death_stop']),end_frame=life['end_frame'],
            kills=sum(m['monster_kills'] for m in damage['by_mod']),dealt=sum(m['monster_health_damage'] for m in damage['by_mod']),
            received=damage['received_health_damage'],fallback_frames=fallback,policy_frames=rows,route_active_rows=route_active))
    assert len(results)==24 and len({r['seed'] for r in results})==4
    totals=[]
    for controller in ('source','candidate','rules'):
        rr=[r for r in results if r['controller']==controller]
        assert len(rr)==8
        totals.append(dict(controller=controller,battles=len(rr),wins=sum(r['goal'] for r in rr),deaths=sum(r['death'] for r in rr),
            kills=sum(r['kills'] for r in rr),dealt=sum(r['dealt'] for r in rr),received=sum(r['received'] for r in rr),
            ownership_failures=sum(bool(r['fallback_frames']) for r in rr) if controller!='rules' else None))
    active_rows=sum(r['route_active_rows'] or 0 for r in results)
    save(a.out,dict(state='passed',results=results,totals=totals,independent_worlds=4,paired_weapon_conditions=8,
        route_active_rows=active_rows,mechanism_activated=active_rows>0,
        promotion=False,scope='Small new TRAIN development cohort, three controllers. Ownership failures count as losses; no held-out superiority claim.',
        inputs={str((root/p).resolve()):sha(root/p) for p in ('spec.json','pool/report.json','cuda-batch/report.json')}))
    print(totals,flush=True)


if __name__=='__main__':main()

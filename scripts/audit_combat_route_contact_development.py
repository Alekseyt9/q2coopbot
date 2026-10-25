"""Matched fresh-seed TRAIN contact-loss development cohort, all cases retained."""
import argparse
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from audit_combat_contact_loss_probe import load_steps


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--screen',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args();assert not a.out.exists();root=a.screen.resolve()
    spec,pool=read(root/'spec.json'),read(root/'pool/report.json')
    assert spec['version']=='combat_memory_route_candidate_native_v1' and spec['fixed_health_100'] and spec['fresh_seed_offsets']
    assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==spec['native_budget']
    batch=read(root/'cuda-batch/report.json');assert batch['state']=='complete' and len(batch['tasks'])==spec['cuda_budget']
    jobs={j['plan_index']:j for j in pool['jobs']};results=[]
    for v in spec['variants']:
        i=v['index'];capture=Path(jobs[i]['root']);rpt=read(capture/'report.json');r=rpt['results'][0]
        assert rpt['capture_complete'] and rpt['provenance_valid'] and r['capture_valid'] and r['dispatch_valid'] and r['seed_confirmed']
        assert r['generated_fixture']==v['context']['instance'] and r['generated_fixture']['health']==100
        d=root/f'case-{i:02d}';meta=read(d/'cuda/report.json');proof=read(d/'cuda/cuda-verification.json')
        assert meta['model_sha256']==v['model_sha256']==sha(v['model']) and proof['state']=='passed' and proof['device']=='cuda'
        assert sha(d/'cuda/cuda-verification.json')==meta['cuda_verification_sha256']
        active=0
        if v['label']=='zero':
            for row in load_steps(d/'cuda/rollout.jsonl'):
                x=row['features']
                active+=x[0]>0 and x[1489]==1 and not any(x[73+12*j] for j in range(8)) and any(x[881+30*j]==1 and x[1490+8*j]==1 and x[1491+8*j]==1 for j in range(8))
        life=r['first_life'];damage=life['damage'];end=r['goal_stop']['kill_frame'] if r['goal_stop'] else life['end_frame']
        steps=load_steps(next((d/'native').glob('replay-*/steps.jsonl')))
        owned=[s for s in steps if s['observation']['identity']['life']==1 and s['observation']['health']>0 and s['observation']['identity']['frame']<end]
        results.append(dict(map=v['context']['map'],seed=v['context']['seed'],loadout=v['context']['loadout'],
            controller='candidate' if v['label']=='zero' else 'source',goal=bool(r['goal_stop']),death=bool(r['death_stop']),
            kills=sum(m['monster_kills'] for m in damage['by_mod']),dealt=sum(m['monster_health_damage'] for m in damage['by_mod']),
            received=damage['received_health_damage'],fallback_frames=sum(s['owner']!='provider' for s in owned),route_active_rows=active))
    totals=[]
    for controller in ('source','candidate'):
        rr=[r for r in results if r['controller']==controller]
        totals.append(dict(controller=controller,battles=len(rr),wins=sum(r['goal'] for r in rr),deaths=sum(r['death'] for r in rr),
            kills=sum(r['kills'] for r in rr),dealt=sum(r['dealt'] for r in rr),received=sum(r['received'] for r in rr),
            ownership_failures=sum(bool(r['fallback_frames']) for r in rr),route_active_rows=sum(r['route_active_rows'] for r in rr),
            route_active_battles=sum(bool(r['route_active_rows']) for r in rr)))
    assert len({r['seed'] for r in results})==spec['independent_worlds']
    save(a.out,dict(state='passed',results=results,totals=totals,promotion=False,
        independent_worlds=spec['independent_worlds'],map_sites=3,source_only_geometry_selection=True,
        scope='Near-layout TRAIN diagnosis with two new seeds per previously route-capable site and fixed100HP. No selection by outcome; all cases count, no held-out or new-geometry superiority claim.',
        inputs={str((root/p).resolve()):sha(root/p) for p in ('spec.json','pool/report.json','cuda-batch/report.json')}))
    print(totals,flush=True)


if __name__=='__main__':main()

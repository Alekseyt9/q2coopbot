"""Read-only credit audit of recorded ownership handoffs; never alters rewards."""
import argparse
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from audit_combat_contact_loss_probe import load_steps


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--screens',nargs='+',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args();assert not a.out.exists();results=[];pins={}
    for directory in a.screens:
        root=directory.resolve();spec=read(root/'spec.json');pool=read(root/'pool/report.json');score=read(root/'acceptance.json')
        assert score['state']=='passed' and pool['state']=='complete'
        jobs={j['plan_index']:j for j in pool['jobs']}
        for v in spec['variants']:
            if v['label']!='zero':continue
            r=next(s for s in score['results'] if s['controller']=='candidate' and s['seed']==v['context']['seed'])
            if not r['fallback_frames']:continue
            d=root/f'case-{v["index"]:02d}'/'cuda';meta=read(d/'report.json');proof=read(d/'cuda-verification.json')
            assert proof['state']=='passed' and proof['device']=='cuda'
            assert sha(d/'cuda-verification.json')==meta['cuda_verification_sha256']
            path=d/'rollout.jsonl';assert sha(path)==meta['rollout_sha256']
            rows=load_steps(path);last=rows[-1]
            capture=Path(jobs[v['index']]['root']);result=read(capture/'report.json')['results'][0]
            bot=Path(result['root'])/'bot.jsonl'
            event=next((o for o in load_steps(bot) if o.get('frame')==last['next_frame']),None)
            assert event is not None
            reason=event.get('combat_policy',{}).get('selection',{}).get('fallback')
            row=dict(screen=str(root),map=v['context']['map'],seed=v['context']['seed'],
                last_policy_frame=last['frame'],next_frame=last['next_frame'],last_reward=last['reward'],
                terminal=last['terminal'],truncated=last['truncated'],next_value=last['next_value'],
                last_value=last['sample']['value'],next_health=event['health'],handoff_reason=reason,
                episode_goal=bool(result['goal_stop']),handoff_bootstrap=meta['handoff_bootstrap'],
                verified_alive_visibility_timeout=event['health']>0 and reason=='combat_visibility_timeout' and not result['goal_stop'])
            results.append(row)
            for p in (root/'acceptance.json',d/'report.json',d/'cuda-verification.json',path,bot):pins[str(p.resolve())]=sha(p)
    alive=[r for r in results if r['verified_alive_visibility_timeout']]
    assert alive and all(not r['terminal'] and r['truncated'] and r['handoff_bootstrap']=='nonterminal_v1' for r in alive)
    save(a.out,dict(state='passed',results=results,alive_visibility_timeouts=len(alive),
        positive_bootstraps=sum(r['next_value']>0 for r in alive),inputs=pins,
        rewards_changed=False,training_performed=False,
        conclusion='Existing continuity semantics retain critic bootstrap on verified alive visibility timeouts with unfinished goals. This is observed credit, not proof of a bug or of improvement from changing objectives. A terminal autonomous-task failure would require an explicit separate objective and controlled comparison.'))
    print(dict(alive_visibility_timeouts=len(alive),positive_bootstraps=sum(r['next_value']>0 for r in alive)),flush=True)


if __name__=='__main__':main()

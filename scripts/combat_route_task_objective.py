"""Evidence-bound task bootstrap overlay. Original native labels stay untouched."""
import copy
from pathlib import Path
from evaluate_registered_combat_models import read, sha
from audit_combat_contact_loss_probe import load_steps

CONTINUITY = 'combat_continuity_v1'
AUTONOMOUS = 'combat_autonomous_fixture_v1'


def qualifies(row, evidence):
    """A positive proof, not absence of a goal or a generic truncation."""
    identity=(row['seed'],row['index'],row['frame'],row['next_frame'])
    return (identity==tuple(evidence['identity']) and evidence['capture_verified']
        and evidence['dispatch_verified'] and evidence['same_life']
        and evidence['owner_before']=='provider' and evidence['owner_after']=='rules'
        and evidence['fallback']=='combat_visibility_timeout'
        and evidence['health_before']>0 and evidence['health_after']>0
        and not evidence['goal_before_or_at_handoff'] and not evidence['death_before_or_at_handoff']
        and not evidence['episode_goal']
        and row['next_frame']==row['frame']+1 and row['truncated'] and not row['terminal']
        and not row['bootstrap_zero'] and not row['next_reset'])


def collect_evidence(capture, rows):
    spec,pool=read(capture/'spec.json'),read(capture/'pool/report.json')
    assert pool['state']=='complete' and pool['source_unchanged']
    jobs={j['plan_index']:j for j in pool['jobs']};proofs=[];pins={}
    for seed in sorted({r['seed'] for r in rows}):
        selected=[r for r in rows if r['seed']==seed];last=selected[-1]
        variants=[v for v in spec['variants'] if v['label']=='zero' and v['context']['seed']==seed]
        assert len(variants)==1
        variant=variants[0];member=Path(jobs[variant['index']]['root'])
        report=read(member/'report.json');r=report['results'][0]
        manifest=read(member/'manifest.json')
        assert r['seed']==seed and manifest['model_weights_sha256'].lower()==variant['model_sha256']
        steps_path=next((capture/f'case-{variant["index"]:02d}'/'native').glob('replay-*/steps.jsonl'))
        steps=load_steps(steps_path)
        before=next((s for s in steps if s['observation']['identity']['frame']==last['frame']),None)
        after=next((s for s in steps if s['observation']['identity']['frame']==last['next_frame']),None)
        bot_path=Path(r['root'])/'bot.jsonl'
        event=next((b for b in load_steps(bot_path) if b.get('frame')==last['next_frame']),None)
        goal,death=r['goal_stop'],r['death_stop']
        evidence=dict(identity=[seed,last['index'],last['frame'],last['next_frame']],
            capture_verified=bool(report['capture_complete'] and report['provenance_valid'] and r['capture_valid'] and r['seed_confirmed']),
            dispatch_verified=bool(r['dispatch_valid']),owner_before=before['owner'] if before else None,
            owner_after=after['owner'] if after else None,
            fallback=(event or {}).get('combat_policy',{}).get('selection',{}).get('fallback'),
            health_before=before['observation']['health'] if before else 0,health_after=after['observation']['health'] if after else 0,
            same_life=bool(before and after and all(before['observation']['identity'][k]==after['observation']['identity'][k]
                for k in ('life','map','connection','spawncount','actor'))),
            goal_before_or_at_handoff=bool(goal and goal['kill_frame']<=last['next_frame']),
            death_before_or_at_handoff=bool(death),episode_goal=bool(goal))
        proofs.append(evidence)
        for p in (capture/'spec.json',capture/'pool/report.json',member/'report.json',member/'manifest.json',steps_path,bot_path):pins[str(p.resolve())]=sha(p)
    return proofs,pins


def overlay(rows, proofs, objective):
    assert objective in (CONTINUITY,AUTONOMOUS)
    result=copy.deepcopy(rows);changed=[]
    by_identity={tuple(e['identity']):e for e in proofs}
    assert len(by_identity)==len(proofs)
    for row in result:
        evidence=by_identity.get((row['seed'],row['index'],row['frame'],row['next_frame']))
        if objective==AUTONOMOUS and evidence and qualifies(row,evidence):
            row['task_terminal']=True
            row['task_bootstrap_zero']=True
            row['task_end_reason']='verified_alive_unfinished_visibility_timeout'
            changed.append(dict(seed=row['seed'],index=row['index'],frame=row['frame'],next_frame=row['next_frame'],
                original_next_value=row['next_value'],native_terminal=row['terminal'],native_truncated=row['truncated']))
    return result,changed


def task_advantages(rows,gamma,lam,calculate):
    # The algorithm sees task terminal state, not a fabricated native death.
    scratch=[{**r,'terminal':r['terminal'] or r.get('task_terminal',False),
              'next_value':0. if r.get('task_bootstrap_zero') else r['next_value']} for r in rows]
    return calculate(scratch,gamma,lam)

"""Positive recorded proof of provider control while a picked-up weapon is active."""
import argparse,collections,hashlib,json,pathlib
from process_combat_architecture_pool import read,save,sha


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=pathlib.Path,required=True)
    a=ap.parse_args(); root=a.root.resolve(); protocol=read(root/'protocol.json'); proof=read(root/'recovery/verified-members.json')
    assert proof['state']=='complete' and proof['protocol_sha256']==sha(root/'protocol.json')
    counts={}; examples=[]; rules_examples=[]; inputs=[]
    for entry in protocol['evaluations']:
        if entry['model']=='rules': continue
        assert sha(entry['plan'])==entry['plan_sha256']; plan=read(entry['plan']); c=collections.Counter()
        for ti,task in enumerate(plan['tasks']):
            for seed in task['seeds']:
                folder=pathlib.Path(entry['root'])/f"case-{ti}-{task['modes'][0]}"/f's-{seed}'
                assert sha(folder/'report.json')==proof['members'][str(folder)]['report_sha256']
                steps=pathlib.Path(read(folder/'report.json')['results'][0]['root'])/'dataset/steps.jsonl'; digest=hashlib.sha256()
                with steps.open('rb') as stream:
                    for line in stream:
                        digest.update(line); row=json.loads(line); obs=row['observation']
                        if obs['health']<=0 or obs['identity']['life']!=1: continue
                        assert row['server_execution']['matched']
                        if obs['weapon'] not in ('Shotgun','models/weapons/v_shotg/tris.md2'): continue
                        c['shotgun_frames']+=1; c[row['owner']+'_shotgun_frames']+=1
                        if row['owner']=='rules':
                            clear=any(e.get('clear_shot') for e in obs.get('enemies',[]))
                            c['rules_shotgun_clear_target_frames']+=int(clear)
                            if len(rules_examples)<12:
                                rules_examples.append(dict(variant=entry['model']+'-'+entry['label'],family=task['episode']['id'],seed=seed,
                                    worker=str(steps.parent.parent),identity=obs['identity'],clear_target=clear,
                                    observed_enemies=len(obs.get('enemies',[])),applied_action=row['applied_action']))
                        if row['owner']=='provider':
                            c['provider_shotgun_attack_sent']+=int(row['applied_action']['attack'])
                            if row['applied_action']['attack'] and len(examples)<12:
                                examples.append(dict(variant=entry['model']+'-'+entry['label'],family=task['episode']['id'],seed=seed,
                                    identity=obs['identity'],action=row['action'],applied_action=row['applied_action']))
                inputs.append(dict(member=str(folder),steps_sha256=digest.hexdigest()))
        counts[entry['model']+'-'+entry['label']]=dict(c)
    # Preserve diagnostic evidence even when the strict all-frame acceptance
    # below fails. This observation file never substitutes for accepted proof.
    save(root/'pickup-command-observations.json',dict(state='observations_complete',groups=counts,
        rules_examples=rules_examples,inputs=inputs,protocol_sha256=sha(root/'protocol.json'),
        member_proof_sha256=sha(root/'recovery/verified-members.json'),
        scope='All matched alive first-life equipped Shotgun commands, including frames without observed enemies. Strict provider-only acceptance is separate; no fallback reason or discharge attribution.'))
    assert sum(c.get('provider_shotgun_frames',0) for c in counts.values())>0, 'Pickup correction not exercised'
    assert not any(c.get('rules_shotgun_frames',0) for c in counts.values()), 'Rules still own picked-up Shotgun frames'
    save(root/'pickup-command-proof.json',dict(state='complete',protocol_sha256=sha(root/'protocol.json'),member_proof_sha256=sha(root/'recovery/verified-members.json'),
        groups=counts,provider_shotgun_attack_exercised=bool(sum(c.get('provider_shotgun_attack_sent',0) for c in counts.values())),
        examples=examples,inputs=inputs,scope='Positive matched native command ownership while observed equipped Shotgun. Attack is reported separately and may be unexercised; ownership is not proof of learned Shotgun shooting skill. Does not equate held attack with a ready-weapon discharge or causal kill. No NN replay.'))
    print(json.dumps(counts))


if __name__=='__main__': main()

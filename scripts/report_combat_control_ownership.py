"""First-life executed-frame ownership and fallback reasons from sealed captures."""
import argparse, collections, hashlib, json, pathlib
from process_combat_architecture_pool import read,save,sha


def identity(observation):
    i=observation['identity']
    return tuple(i[k] for k in ('map','connection','spawncount','actor','life','frame'))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=pathlib.Path,required=True)
    a=ap.parse_args(); root=a.root.resolve(); protocol=read(root/'protocol.json')
    proof=read(root/'recovery/verified-members.json'); quality=read(root/'quality-report.json')
    assert proof['state']==quality['state']=='complete'
    assert proof['protocol_sha256']==quality['protocol_sha256']==sha(root/'protocol.json')
    groups={}; inputs=[]
    for entry in protocol['evaluations']:
        assert sha(entry['plan'])==entry['plan_sha256']; plan=read(entry['plan']); counts=collections.Counter(); reasons=collections.Counter(); weapons=collections.Counter(); episodes=[]
        for ti,task in enumerate(plan['tasks']):
            for seed in task['seeds']:
                folder=pathlib.Path(entry['root'])/f"case-{ti}-{task['modes'][0]}"/f's-{seed}'
                receipt=proof['members'][str(folder)]
                for name in ('report','manifest'): assert sha(folder/(name+'.json'))==receipt[name+'_sha256']
                result=read(folder/'report.json')['results'][0]; physical=pathlib.Path(result['root'])
                trace=physical/'bot.jsonl'; steps=physical/'dataset/steps.jsonl'
                selections=collections.defaultdict(set); trace_digest=hashlib.sha256()
                with trace.open('rb') as stream:
                    for line in stream:
                        trace_digest.update(line); row=json.loads(line); cp=row.get('combat_policy')
                        if not cp or not cp.get('selection') or not cp.get('observation'): continue
                        sel=cp['selection']; selections[identity(cp['observation'])].add((sel['owner'],sel.get('fallback','')))
                first_life=None; steps_digest=hashlib.sha256(); episode_rules_clear=0; episode_equip=0
                with steps.open('rb') as stream:
                    for line in stream:
                        steps_digest.update(line); row=json.loads(line); obs=row['observation']
                        if first_life is None and obs['health']>0: first_life=obs['identity']['life']
                        if obs['health']<=0 or obs['identity']['life']!=first_life: continue
                        assert row['server_execution']['matched']
                        matches=selections[identity(obs)]; assert len(matches)==1, 'Ambiguous/missing captured selection'
                        owner,reason=next(iter(matches)); assert owner==row['owner']
                        counts['alive_executed_frames']+=1; counts[owner+'_frames']+=1
                        if owner=='rules':
                            reasons[reason or 'unmarked']+=1
                            if reason=='pilot_equip_not_ready': weapons[obs['weapon']]+=1; episode_equip+=1
                            if obs.get('enemies'): counts['rules_with_observed_enemies']+=1
                            if any(e.get('clear_shot') for e in obs.get('enemies',[])):
                                counts['rules_with_clear_target']+=1; counts['rules_clear_attack_sent']+=int(row['applied_action']['attack']); episode_rules_clear+=1
                inputs.append(dict(member=str(folder),trace_sha256=trace_digest.hexdigest(),steps_sha256=steps_digest.hexdigest()))
                episodes.append(dict(seed=seed,family=task['episode']['id'],rules_with_clear_target=episode_rules_clear,equip_fallback=episode_equip,
                    win=(result.get('goal_stop') or {}).get('reason')=='combat_goal_complete'))
        groups[entry['model']+'-'+entry['label']]=dict(counts=dict(counts),fallback_reasons=dict(reasons),equip_fallback_weapons=dict(weapons),episodes=episodes)
    save(root/'control-ownership.json',dict(state='complete',protocol_sha256=sha(root/'protocol.json'),member_proof_sha256=sha(root/'recovery/verified-members.json'),
        quality_sha256=sha(root/'quality-report.json'),groups=groups,inputs=inputs,
        scope='Only matched native commands in first observed life, alive precommand. Trace selection matched by full observation identity. Preparation trace callbacks excluded through executed-step membership. Observed enemies/clear_shot do not prove weapon readiness. No NN replay or causal attribution.'))
    print(json.dumps({k:{f:v[f] for f in ('counts','fallback_reasons','equip_fallback_weapons')} for k,v in groups.items()}))


if __name__=='__main__': main()

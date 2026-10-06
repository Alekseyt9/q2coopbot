"""Native outcome, sequence, frozen lineage and matched-start audit."""
import argparse,collections,json,pathlib,math,re
from ppo_combat import torch,network,sha
from ppo_recurrent import Recurrent
from combat_attention import CausalAttention,EntityAttention
from audit_combat_closure import audit as audit_closure

read=lambda p:json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
lines=lambda p:[json.loads(s) for s in pathlib.Path(p).read_text(encoding='utf-8-sig').splitlines()]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);a=ap.parse_args();root=a.root.resolve()
    protocol=read(root/'protocol.json');seed=protocol['seed']
    for key,file in [('parent','parent.json'),('config','config.json'),('reward','reward.json'),('bank','bank.json')]:assert sha(root/file)==protocol[key+'_sha256'].lower()
    parent=read(root/'parent.json');arms=('mlp','gru','attention','entity');fingerprints=set();records=[];training={};evaluation={};prefixes={};counts=collections.Counter()
    def batch(path,expected_seeds,mixed,skill):
        report=read(path/'report.json');manifest=read(path/'manifest.json')
        assert report['capture_complete'] and report['provenance_valid'] and report['parallelism']==4
        assert manifest['workers']==4 and manifest['episodes_per_worker']==1 and manifest['timescale']==2 and manifest['game_frames']==300 and manifest['release_game_frame']==100
        assert manifest['loadout']=='blaster' and manifest['mixed']==mixed and manifest['skill']==skill
        assert {e['seed'] for e in report['results']}==set(expected_seeds)
        fingerprints.add((manifest['source_fingerprint'],manifest['native_source_fingerprint']))
        outcomes=[];traces={};elapsed=[];over_budget=0
        for ep in report['results']:
            assert all(ep[k] for k in ('capture_valid','dispatch_valid','seed_confirmed'))
            ep_root=pathlib.Path(ep['root']);ss=lines(ep_root/'dataset/steps.jsonl');rr=lines(ep_root/'dataset/rewards.jsonl');assert len(ss)==len(rr)
            first=ss[0]['observation'];assert first['health']==100 and first['weapon']=='Blaster' and first['gun_frame']==9
            assert read(ep_root/'dataset/report.json')['observed_reset_confirmed']
            assert re.findall(r'^g_test_skill_start game_frame=100 skill=(\d+)$',(ep_root/'server.log').read_text(encoding='utf-8-sig'),re.M)==[str(skill)]
            native=ep['first_life'];classes={t['target_class']:t['kills'] for t in native['outgoing_by_target_class'] if t['target_class'].startswith('monster_')}
            kills=sum(classes.values());death=int(native['end_reason']=='first_observed_death');killframes=[];provider_kills=0
            for s,r in zip(ss,rr):
                if s['observation']['identity']['life']==1 and r['available'] and r['components'].get('monster_kill',0)>0:
                    assert s['server_execution']['matched'] and s['server_execution']['window_exclusive']
                    k=int(r['components']['monster_kill']/5)
                    if s['owner']=='provider':provider_kills+=k
                    killframes.extend([s['observation']['identity']['frame']]*k)
            assert kills==len(killframes);needed=2 if mixed else 1;started=False;interruptions=[];commands=[]
            for row in lines(ep_root/'bot.jsonl'):
                cp=row.get('combat_policy',{});o=cp.get('observation');sel=cp.get('selection',{})
                if not o or o['identity']['life']!=1:continue
                if sel.get('owner')=='provider':
                    started=True;elapsed.append(sel['elapsed_us']);over_budget+=bool(sel.get('inference_budget_exceeded',False))
                    sample=dict(sel.get('sample') or {});sample.pop('version',None);sample.pop('memory',None)
                    proposed=dict(cp['proposed']);commands.append(dict(frame=o['identity']['frame'],command=row['sent_command'],proposed=proposed,sample=sample))
                if started and o['health']>0 and sel.get('owner')=='rules' and (kills<needed or o['identity']['frame']<max(killframes)):
                    interruptions.append(o['identity']['frame'])
            complete=classes.get('monster_parasite',0)>=1 and (not mixed or classes.get('monster_gunner',0)>=1)
            outcomes.append(dict(seed=ep['seed'],kills=kills,provider_kills=provider_kills,kills_by_class=classes,deaths=death,uninterrupted_wins=int(complete and provider_kills==kills and not death and not interruptions),received=native['damage']['received_health_damage'],rules_before_success=len(interruptions)))
            traces[ep['seed']]=dict(commands=commands,native=native,events=[e.get('events',[]) for e in lines(ep_root/'dataset/server_outcomes.jsonl')])
        ordered=sorted(elapsed)
        quantile=lambda q:ordered[min(len(ordered)-1,int(q*(len(ordered)-1)))] if ordered else None
        counts['captures']+=4;counts['batches']+=1
        result=dict(totals={k:sum(r[k] for r in outcomes) for k in ('kills','deaths','uninterrupted_wins','received')},episodes=outcomes,go_inference_us=dict(p50=quantile(.5),p95=quantile(.95),max=max(elapsed,default=0),samples=len(elapsed),over_5ms=over_budget))
        records.append(dict(path=str(path.relative_to(root)),manifest_sha256=sha(path/'manifest.json')))
        return result,traces
    for arm in arms:
        initial=read(root/arm/'initial/weights.json')
        baseline={k:v for k,v in initial.items() if k not in ('memory','attention','entity_attention')};assert baseline==parent
        if arm!='mlp':
            key={'gru':'memory','attention':'attention','entity':'entity_attention'}[arm]
            for name in ('actor','value'):
                cell=initial[key][name];projection=cell['output'] if arm=='gru' else cell['attention']['residual'] if arm=='entity' else cell['residual']
                assert all(v==0 for row in projection['weight'] for v in row) and all(v==0 for v in projection['bias'])
        previous=root/arm/'initial/weights.json';entries=[];consumed=[];actor_steps=0
        for i in (1,2):
            folder=root/arm/f'iteration-{i}';meta=read(folder/'rollout/report.json');r=read(folder/'update/report.json');rows=lines(folder/'rollout/rollout.jsonl')
            assert r['device']=='cuda' and r['final_approx_kl']<=.010001
            assert r['behavior_sha256']==sha(previous)==meta['model_sha256'];assert r['rollout_sha256']==sha(folder/'rollout/rollout.jsonl')==meta['rollout_sha256']
            assert meta['rollout_sha256'] not in consumed;consumed.append(meta['rollout_sha256'])
            assert r['rows']==len(rows)==meta['rows'] and {x['seed'] for x in rows}==set(range(seed+4*(i-1),seed+4*i))
            for path,digest in meta['source_sha256'].items():assert sha(pathlib.Path(path))==digest
            if arm!='mlp' and arm!='entity':assert meta['sequence_sha256']==sha(folder/'rollout/sequence.jsonl')==r['sequence_sha256']
            checkpoint=torch.load(folder/'update/checkpoint.pt',map_location='cpu',weights_only=True);updated=read(folder/'update/weights.json')
            assert checkpoint['weights_sha256']==r['weights_sha256']==sha(folder/'update/weights.json')
            actor_steps+=r['actor_steps'];assert checkpoint['updates_completed']==i and checkpoint['total_actor_steps']==actor_steps and checkpoint['consumed_rollouts']==consumed
            if i==2:assert r['resume_sha256']==sha(root/arm/'iteration-1/update/checkpoint.pt')
            if arm=='mlp':modules={n:network(updated[n]) for n in ('actor','value')}
            elif arm=='gru':modules={n:Recurrent(updated[n],updated['memory'][n]) for n in ('actor','value')}
            elif arm=='attention':modules={n:CausalAttention(updated[n],updated['attention'][n],4,32) for n in ('actor','value')}
            else:modules={n:EntityAttention(updated[n],updated['entity_attention'][n],4) for n in ('actor','value')}
            for name,module in modules.items():
                expected=module.state_dict();assert expected.keys()==checkpoint[name].keys()
                assert all(torch.equal(v,checkpoint[name][k].cpu()) for k,v in expected.items())
            assert torch.equal(checkpoint['log_std'].cpu(),torch.tensor(updated['log_std']))
            assert all(int(v['step'])==actor_steps for v in checkpoint['actor_optimizer']['state'].values())
            assert all(int(v['step'])==40*i for v in checkpoint['value_optimizer']['state'].values())
            result,trace=batch(folder/'batch',range(seed+4*(i-1),seed+4*i),True,1)
            if i==1:prefixes[arm]=trace
            entries.append(dict(iteration=i,rows=len(rows),actor_steps=r['actor_steps'],seconds=r['seconds'],outcomes=result,old_log_probability_max_error=r['old_log_probability_max_error'],old_value_max_error=r['old_value_max_error'],old_actor_memory_max_error=r.get('old_actor_memory_max_error'),old_value_memory_max_error=r.get('old_value_memory_max_error')))
            previous=folder/'update/weights.json'
        actor_parameters=sum(p.numel() for p in modules['actor'].parameters())+4;critic_parameters=sum(p.numel() for p in modules['value'].parameters())
        training[arm]=dict(iterations=entries,rows=sum(e['rows'] for e in entries),actor_steps=actor_steps,parameters=actor_parameters+critic_parameters)
    for arm in arms[1:]:assert prefixes[arm]==prefixes['mlp'],f'Initial native/sampling behavior differs: {arm}'
    for fixture,eval_seed in [('mixed',seed+100),('solo',seed+200),('hard-solo',seed+300)]:
        evaluation[fixture]={}
        for arm in ('parent',)+arms:
            path=root/arm/f'evaluation-{fixture}';result,_=batch(path,range(eval_seed,eval_seed+4),fixture=='mixed',3 if fixture=='hard-solo' else 1)
            manifest=read(path/'manifest.json');actual=read(manifest['model_weights']);expected=parent if arm=='parent' else read(root/arm/'iteration-2/update/weights.json')
            actual['deterministic']=expected['deterministic'];assert actual==expected
            evaluation[fixture][arm]=result
    assert dict(counts)==dict(captures=92,batches=23) and len(fingerprints)==1
    closure={arm:audit_closure(root/arm) for arm in arms}
    out=dict(complete=True,counts=dict(counts),initial_native_behavior_equal=True,source_native_fingerprints=list(fingerprints)[0],training=training,evaluation=evaluation,closure=closure,manifests=records,live_promoted=False,scope='Two CUDA updates per architecture, paired fresh seeds, fixed Blaster. Current-history GRU/causal attention vs current-entity attention and MLP. Four seeds per condition, zero-memory historical bank and no full regression suite: not statistical superiority or production acceptance.')
    (root/'completion-status.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(counts=dict(counts),evaluation={f:{a:r['totals'] for a,r in arms.items()} for f,arms in evaluation.items()}),indent=2))

if __name__=='__main__':main()

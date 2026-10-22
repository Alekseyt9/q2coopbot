"""Re-export sealed TRAIN captures and diagnose fire reward signals on CUDA.

Diagnostic only: never edits PPO metadata, consumes a rollout or writes weights.
"""
import argparse
import collections
import concurrent.futures
import copy
import json
import math
import pathlib
import subprocess
import sys
from evaluate_registered_combat_models import read,save,sha


def lines(path):
    with pathlib.Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    args=ap.parse_args();root=args.corpus.resolve();out=args.out.resolve()
    assert not out.exists()
    pins={}
    def pin(path,expected=None):
        path=pathlib.Path(path).resolve();digest=sha(path)
        assert expected is None or digest==expected,str(path)
        pins[str(path)]=digest;return path
    closed=read(pin(root/'complete.json'));assert closed['state']=='complete' and not closed['training_performed']
    protocol=read(pin(root/'protocol.json',closed['protocol_sha256']))
    assert protocol['train_only'] and protocol['ppo_training_objective'] and not protocol['diagnostic_intervention']
    pool=read(pin(root/'pool/report.json',closed['pool_sha256']))
    assert pool['state']=='complete' and pool['source_unchanged'] and not any(j['error'] for j in pool['jobs'])
    assert len(pool['plans'])==1 and len(pool['jobs'])==32
    plan_path=pin(pool['plans'][0]['path'],pool['plans'][0]['sha256']);plan=read(plan_path)
    assert all(t['split']=='train' and t['modes']==['learned'] for t in plan['tasks'])
    model_path=pin(plan['model_path'],plan['model_sha256']);model=read(model_path)
    assert model.get('conditional_fire') and not model['deterministic']
    processing=root/'processing-train';meta=read(pin(processing/'merged/report.json'))
    original_rows_path=pin(processing/'merged/rollout.jsonl',meta['rollout_sha256'])
    sequence_path=pin(processing/'merged/sequence.jsonl',meta['sequence_sha256'])
    rows=lines(original_rows_path);lookup={(v['seed'],v['index']):v for v in rows}
    assert len(lookup)==len(rows)==meta['rows'] and meta['model_sha256']==sha(model_path)
    reward_paths={i['path'] for i in protocol['inputs'] if i['sha256']==meta['reward_config_sha256']}
    assert len(reward_paths)==1
    reward_path=pin(next(iter(reward_paths)),meta['reward_config_sha256']);reward=read(reward_path)
    assert reward['version']=='combat_reward_v13' and reward['machinegun_miss']==0
    out.mkdir(parents=True)
    (out/'control-reward.json').write_bytes(reward_path.read_bytes())
    cost_reward={**reward,'machinegun_miss':-.005};save(out/'cost-reward.json',cost_reward)
    assert {k:v for k,v in cost_reward.items() if k!='machinegun_miss'}=={k:v for k,v in reward.items() if k!='machinegun_miss'}
    jobs=[]
    for job in sorted(pool['jobs'],key=lambda v:v['job']):
        capture_path=pin(pathlib.Path(job['root'])/'report.json');capture=read(capture_path)
        assert capture['capture_complete'] and capture['provenance_valid'] and len(capture['results'])==1
        result=capture['results'][0];worker=pathlib.Path(result['root'])
        manifest=read(pin(pathlib.Path(job['root'])/'manifest.json'))
        exporter=pin(pathlib.Path(job['root'])/'q2combat-export.exe',manifest['exporter_sha256'].lower())
        assert manifest['model_weights_sha256'].lower()==sha(model_path)
        for name in ('bot.jsonl','server.log','reset-expectation.json'):
            pin(worker/name)
        if result.get('goal_stop'):pin(worker/'goal-stop.json')
        if result.get('death_stop'):pin(worker/'death-stop.json')
        original=processing/f"case-{job['job']:04d}/native/replay-0"
        for name in ('steps.jsonl','server_outcomes.jsonl','rewards.jsonl'):pin(original/name)
        jobs.append((job,result,worker,exporter,original))
    save(out/'protocol.json',dict(version='conditional_fire_reward_diagnostic_v1',corpus=str(root),
        cases=32,control_cost=0,cost=-.005,training_performed=False,ppo_ready=False,promotion=False,
        source_model_sha256=sha(model_path),source_rollout_sha256=meta['rollout_sha256']))
    save(out/'progress.json',dict(stage='native-relabel',cases=32))

    def export_case(item):
        job,result,worker,exporter,original=item
        case=out/f"case-{job['job']:04d}";case.mkdir()
        for label in ('control','cost'):
            target=case/label
            reason='combat_goal_complete' if result.get('goal_stop') else 'combat_first_life_death' if result.get('death_stop') else 'game_frame_limit'
            command=[exporter,'--trace',worker/'bot.jsonl','--server-log',worker/'server.log',
                '--reset-expectation',worker/'reset-expectation.json','--reward-config',out/(label+'-reward.json'),
                '--out',target,'--worker',f"worker-{result['worker']}",'--episode',f"seed-{job['seed']}",
                '--end-reason',reason,'--client-name','SoloRetreatBot','--require-execution','--synchronous']
            if result.get('goal_stop'):command+=['--goal-observed-frame',str(result['goal_stop']['observed_frame'])]
            if result.get('death_stop'):command+=['--death-stop',worker/'death-stop.json']
            with (case/(label+'.log')).open('w',encoding='utf-8') as log:
                run=subprocess.run(list(map(str,command)),stdout=log,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            assert run.returncode==0,str(case/(label+'.log'))
        assert sha(original/'steps.jsonl')==sha(case/'control/steps.jsonl')==sha(case/'cost/steps.jsonl')
        assert sha(original/'rewards.jsonl')==sha(case/'control/rewards.jsonl')
        assert sha(original/'server_outcomes.jsonl')==sha(case/'control/server_outcomes.jsonl')
        control,cost=lines(case/'control/rewards.jsonl'),lines(case/'cost/rewards.jsonl')
        before_out,after_out=lines(case/'control/server_outcomes.jsonl'),lines(case/'cost/server_outcomes.jsonl')
        assert len(control)==len(cost)==len(before_out)==len(after_out)
        steps={s['index']:s for s in lines(case/'control/steps.jsonl')}
        changes=[];statuses=collections.Counter()
        for a,b,native,after_native in zip(control,cost,before_out,after_out):
            assert a['step']==b['step']==native['step']==after_native['step']
            na=dict(native);nb=dict(after_native);na.pop('reward_score',None);nb.pop('reward_score',None)
            assert na==nb,'Native outcome evidence changed'
            ac,bc=a.get('components') or {},b.get('components') or {}
            assert {k:v for k,v in ac.items() if k!='machinegun_miss'}=={k:v for k,v in bc.items() if k!='machinegun_miss'}
            av,bv=dict(a),dict(b)
            for v in (av,bv):v.pop('score',None);v.pop('components',None)
            assert av==bv,'Non-reward transition label changed'
            misses=(native.get('hitscan_misses') or {}).get('misses',[])
            if a['available']:
                assert math.isclose(bc['machinegun_miss'],len(misses)*-.005,abs_tol=1e-10)
                assert math.isclose(b['score']-a['score'],bc['machinegun_miss'],abs_tol=1e-8)
            key=(job['seed'],a['step'])
            if key not in lookup:continue
            row=lookup[key];assert a['available'] and math.isclose(a['score'],row['reward'],abs_tol=1e-8)
            step=steps[a['step']]
            if misses:
                dispatch=step['server_execution']
                assert step['owner']=='provider' and dispatch['matched'] and dispatch['window_exclusive'] and dispatch.get('recovery_commands',0)==0
                assert row['sample']['attack'],'Miss cost attributed to a non-attack sample'
                for miss in misses:
                    assert miss['window']==step['native_step'] and miss['outcome'] in ('geometry','sky','no_contact','corpse')
                    statuses[miss['outcome']]+=1
            if result.get('goal_stop'):assert row['next_frame']<=result['goal_stop']['observed_frame']
            changes.append(dict(seed=key[0],index=key[1],reward=b['score'],misses=len(misses)))
        return dict(job=job['job'],seed=job['seed'],rows=len(changes),miss_outcomes=dict(statuses),changes=changes,
            control_steps_sha256=sha(case/'control/steps.jsonl'),cost_rewards_sha256=sha(case/'cost/rewards.jsonl'))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as workers:
        cases=list(workers.map(export_case,jobs))
    updates={(v['seed'],v['index']):v for case in cases for v in case['changes']}
    assert set(updates)==set(lookup)
    with (out/'reward-deltas.jsonl').open('w',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(updates[(row['seed'],row['index'])])+'\n')
    save(out/'native-complete.json',dict(state='complete',cases=[{k:v for k,v in c.items() if k!='changes'} for c in cases],
        rows=len(rows),control_replay_byte_exact=True,observation_actions_and_labels_unchanged=True,
        other_reward_components_unchanged=True,training_performed=False,ppo_ready=False))
    save(out/'progress.json',dict(stage='cuda-signal',cases=32,rows=len(rows)))
    from ppo_combat import torch,training_devices,advantages
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    from migrate_combat_navigation_cuda import branch
    from ppo_recurrent import prepare_context,sequence
    from combat_conditional_fire import logits
    actor=branch(model,'actor');prepared=prepare_context(lines(sequence_path),rows,'cuda')
    x=torch.tensor([v['features'] for v in rows]);z=torch.tensor([v['sample']['latent'] for v in rows])
    category=lambda name:torch.tensor([v['sample'].get(name,0) for v in rows],dtype=torch.long)
    target,mode,vertical,weapon=(category(k) for k in ('target','aim_mode','vertical','weapon'))
    attack=torch.tensor([float(v['sample']['attack']) for v in rows])
    misses=torch.tensor([updates[(v['seed'],v['index'])]['misses'] for v in rows])
    cfg=read(root/'training-config.json');variants={}
    with torch.no_grad():
        raw,_=sequence(actor,prepared,check_states=True)
        probability=logits(actor.conditional_fire_branch,x,raw,target,mode,z,vertical,weapon).sigmoid()
        for label in ('control','cost'):
            altered=[{**v,'reward':updates[(v['seed'],v['index'])]['reward']} for v in rows] if label=='cost' else rows
            adv,_=advantages(altered,cfg['gamma'],cfg['lambda']);adv=torch.tensor(adv)
            normalized=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
            score=normalized*(attack-probability);groups={}
            for name,mask in [('all',torch.ones_like(target,dtype=torch.bool)),('no_selected_target',target==0),
                ('selected_target',target>0),('native_mg_miss',misses>0),('no_target_mg_miss',(target==0)&(misses>0))]:
                if bool(mask.any()):groups[name]=dict(rows=int(mask.sum()),mean_advantage=float(adv[mask].mean()),
                    normalized_mean_advantage=float(normalized[mask].mean()),mean_fire_score=float(score[mask].mean()))
            variants[label]=dict(reward_sum=sum(v['reward'] for v in altered),advantage_std=float(adv.std(unbiased=False)),groups=groups)
    for path,digest in pins.items():assert sha(pathlib.Path(path))==digest,'Input changed: '+path
    outcomes=collections.Counter()
    for case in cases:outcomes.update(case['miss_outcomes'])
    save(out/'report.json',dict(state='complete',version='conditional_fire_reward_diagnostic_v1',device='cuda',cases=32,rows=len(rows),
        affected_rows=sum(v['misses']>0 for v in updates.values()),native_miss_outcomes=dict(outcomes),variants=variants,
        inputs=pins,script_sha256=sha(__file__),native_complete_sha256=sha(out/'native-complete.json'),
        training_performed=False,ppo_ready=False,promotion=False,
        scope='Counterfactual TRAIN reward calibration only. Original actor/actions/values/causal histories unchanged; exact native attribution. Local score is normalized advantage*(sampled attack-probability), not a causal effect of every miss. No policy update, checkpoint fork or PPO data promotion.'))
    save(out/'complete.json',dict(state='complete',report_sha256=sha(out/'report.json'),training_performed=False,ppo_ready=False,promotion=False))
    save(out/'progress.json',dict(stage='complete',rows=len(rows)))
    print(json.dumps(dict(affected_rows=sum(v['misses']>0 for v in updates.values()),native_miss_outcomes=dict(outcomes),variants=variants)))


if __name__=='__main__':main()

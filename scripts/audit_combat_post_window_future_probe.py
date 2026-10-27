"""Paired independent suffixes after a completed movement option; CUDA replay."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from audit_combat_checked_ground_native_probe import normalized, same
from combat_teacher_ownership import rules_ownership
from process_combat_architecture_pool import read, save, sha
from report_combat_machinegun_hits import scan
from report_registered_machinegun_outcomes import summary


def run(protocol,root,out,reference):
    assert not out.exists()
    spec,pool=read(protocol),read(root/'report.json')
    assert spec['version']=='combat_post_window_future_native_probe_v1'
    assert pool['state']=='complete' and pool['source_unchanged'] and len(pool['jobs'])==6
    assert pool['protocol_sha256']==sha(protocol)
    pins={**spec['source_files'],str(protocol):sha(protocol),str(Path(__file__).resolve()):sha(Path(__file__))}
    accepted=read(reference/'analysis/long-complete.json')
    assert accepted['state']=='complete' and sha(reference/'analysis/long-report.json')==accepted['report_sha256']
    pins[str(reference/'analysis/long-report.json')]=accepted['report_sha256']
    for group in spec['variants'].values():
        for v in group['streams']:
            assert sha(Path(v['path']))==v['sha256'];pins[v['path']]=v['sha256']
    assert all(sha(Path(p))==digest for p,digest in pins.items())
    out.mkdir(parents=True);shutil.copyfile(__file__,out/Path(__file__).name)
    env=dict(os.environ,GOCACHE=str(Path.cwd()/'workspace/build/gocache'))

    def command(label,cmd):
        with (out/(label+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(list(map(str,cmd)),env=env,stdout=log,stderr=log)
        assert result.returncode==0,label

    def trace(path):
        pins[str(path)]=sha(path);result={}
        for line in path.open(encoding='utf-8-sig'):
            raw=json.loads(line);o=raw.get('combat_policy',{}).get('observation',{})
            if o.get('identity',{}).get('life')==1 and o.get('health',0)>0:
                assert o['identity']['frame'] not in result;result[o['identity']['frame']]=raw
        return result

    def signature(raw):
        c=raw['combat_policy'];return normalized({k:c[k] for k in ('observation','proposed','proposed_command','applied_command')})

    references={}
    for label in ('control','back'):
        references[label]=trace(reference/f'jail3-forced-duck-{label}-0/worker-0-episode-0/bot.jsonl')
    exporter=(out/'q2ppo-data.exe').resolve();command('build',['go','build','-o',exporter,'./cmd/q2ppo-data'])
    tasks,records,all_steps=[],[],{}
    for job in sorted(pool['jobs'],key=lambda j:j['job']):
        assert not job['error']
        label,index=job['label'],job['repeat'];variant=spec['variants'][label]['streams'][index]
        folder=Path(job['root']);cap=read(folder/'report.json');manifest=read(folder/'manifest.json')
        assert cap['capture_complete'] and cap['provenance_valid'] and manifest['diagnostic_forced_duck_frame']==-1
        row=cap['results'][0];directory=Path(row['root'])
        assert row['capture_valid'] and row['dispatch_valid'] and row['seed_confirmed']
        values=trace(directory/'bot.jsonl');old=references[label]
        frames=[f for f in old if 98<=f<379]
        assert len(frames)==281 and all(f in values and same(signature(old[f]),signature(values[f]),1e-9) for f in frames), 'Pre-fork option changed'
        target=(out/f"case-{job['job']}").resolve();target.mkdir();model=Path(variant['path'])
        command(f"export-{job['job']}",[exporter,'--batch',folder,'--model',model,'--cuda-only','--out',target/'native'])
        paths=[target/'native/replay-0/steps.jsonl',target/'native/replay-0/server_outcomes.jsonl',directory/'server.log']
        for path in paths:pins[str(path)]=sha(path)
        steps=[json.loads(line) for line in paths[0].open(encoding='utf-8-sig')]
        admitted_path=target/'native/native-rollout.jsonl'
        pins[str(admitted_path)]=sha(admitted_path)
        admitted={r['index'] for r in map(json.loads,admitted_path.open(encoding='utf-8-sig'))}
        effects={r['step']:r for r in map(json.loads,paths[1].open(encoding='utf-8-sig'))}
        windows={};receipts={}
        for s in steps:
            o,n,x=s['observation'],s.get('native_step'),s.get('server_execution') or {}
            if s['index'] not in admitted or s['owner']!='provider' or o['identity']['life']!=1 or o['health']<=0:continue
            assert n and x['matched'] and x['window_exclusive'] and x['recovery_commands']==0
            frame=o['identity']['frame'];receipt=s['sample'].get('continuation_probe')
            if frame<379:
                assert receipt is None
            else:
                assert receipt['version']=='combat_post_movement_window_rng_fork_v2' and receipt['fork_frame']==379
                assert receipt['seed']==variant['suffix_seed'] and receipt['reset']==(frame==379)
                assert s['sample']['sampling_seed']==job['seed']
                receipts[frame]=receipt
            if frame>=374:
                key=tuple(n[k] for k in ('spawncount','begin_frame','actor','sequence'));assert key not in windows;windows[key]=s
        assert 379 in receipts and sum(r['reset'] for r in receipts.values())==1
        all_steps[label,index]=receipts
        shots=[q for q in scan(paths[2].open(encoding='utf-8-sig'),set(windows)) if q['eligible']]
        native_effects=[effects[s['index']] for s in windows.values()]
        assert all(e['available'] for e in native_effects)
        ownership=rules_ownership(steps,row.get('goal_stop'),row['first_life']['end_frame'])
        records.append(dict(label=label,suffix_seed=variant['suffix_seed'],prefix_frames=len(frames),
            live_monster_hp=sum(e['monster_health_damage'] for e in native_effects),kills=sum(e['monster_kills'] for e in native_effects),
            received_hp=sum(e['received_health_damage'] for e in native_effects),shots=summary(shots),
            death=bool(row['death_stop'] or row['first_life']['minimum_health']<=0),goal=bool(row.get('goal_stop')),
            rules_combat_rows=ownership['rules_combat_rows']))
        tasks.append(dict(model=str(model),model_sha256=sha(model),data=str(target/'native'),out=str(target/'cuda'),native_report_sha256=sha(target/'native/report.json')))
    paired=[]
    for index in range(3):
        a,b=all_steps['control',index],all_steps['back',index]
        shared=sorted(a.keys()&b.keys());assert shared and shared[0]==379
        assert all(a[f]['gaussian_noise']==b[f]['gaussian_noise'] and a[f]['fire_uniform']==b[f]['fire_uniform'] for f in shared),'Paired RNG draws diverged'
        left,right=[next(r for r in records if r['label']==label and r['suffix_seed']==spec['variants'][label]['streams'][index]['suffix_seed']) for label in ('control','back')]
        paired.append(dict(suffix_seed=left['suffix_seed'],shared_draw_frames=len(shared),hp_delta_back_minus_control=right['live_monster_hp']-left['live_monster_hp'],
            control_death=left['death'],back_death=right['death'],control_goal=left['goal'],back_goal=right['goal']))
    save(out/'native-results.json',dict(records=records,paired=paired))
    save(out/'cuda-request.json',dict(version='combat_cuda_finalize_batch_v1',compact_finalized_streams=True,tasks=tasks))
    command('cuda',[sys.executable,'scripts/finalize_combat_cuda_batch.py','--request',out/'cuda-request.json','--receipt-root',out/'cuda-batch'])
    for task in tasks:
        meta,proof=read(Path(task['out'])/'report.json'),read(Path(task['out'])/'cuda-verification.json')
        assert proof['state']=='passed' and proof['device']=='cuda' and proof['diagnostic_continuation_version']=='combat_post_movement_window_rng_fork_v2'
        assert meta['diagnostic_only'] and not meta['ppo_training_ready'] and not proof['ordinary_training_eligible']
    assert all(sha(Path(p))==digest for p,digest in pins.items())
    save(out/'report.json',dict(state='complete',records=records,paired=paired,native_cases=6,cuda_cases=6,inputs=pins,
        independent_worlds=1,independent_suffix_streams=3,training_performed=False,promotion=False,scope=spec['scope']))
    save(out/'complete.json',dict(state='complete',report_sha256=sha(out/'report.json')))
    print(json.dumps(paired,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for k in ('protocol','root','out','reference'):parser.add_argument('--'+k,required=True,type=Path)
    args=parser.parse_args()
    run(args.protocol.resolve(),args.root.resolve(),args.out.resolve(),args.reference.resolve())

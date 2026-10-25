"""Explicit successful-rule TRAIN demonstrations; no PPO labels or neural work."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from evaluate_registered_combat_models import read,save,sha
from audit_combat_contact_loss_probe import load_steps


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--screen',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--route-exporter',required=True,type=Path);ap.add_argument('--game-dir',required=True,type=Path)
    a=ap.parse_args();assert not a.out.exists();root=a.screen.resolve();out=a.out.resolve();out.mkdir(parents=True)
    spec=read(root/'spec.json');pool=read(root/'pool/report.json');score=read(root/'acceptance.json')
    assert pool['state']=='complete' and pool['source_unchanged'] and score['state']=='passed'
    jobs={j['plan_index']:j for j in pool['jobs']};episodes=[];pins={}
    def pin(p):p=Path(p).resolve();pins[str(p)]=sha(p);return p
    pin(root/'spec.json');pin(root/'pool/report.json');pin(root/'acceptance.json');pin(a.route_exporter)
    selected=[]
    for v in spec['variants']:
        if v['label']!='rules':continue
        capture=Path(jobs[v['index']]['root']);report=read(pin(capture/'report.json'));r=report['results'][0]
        if not r['goal_stop'] or r['death_stop']:continue
        assert report['capture_complete'] and report['provenance_valid'] and r['capture_valid'] and r['dispatch_valid'] and r['seed_confirmed']
        assert v['context']['instance']['split']=='train'
        selected.append((v,capture,r))
    assert len(selected)>=4
    # Largest seed of each represented map is diagnostic only; fixed before fit.
    diagnostic={max(v['context']['seed'] for v,_,_ in selected if v['context']['map']==name) for name in {v['context']['map'] for v,_,_ in selected}}
    for k,(v,capture,r) in enumerate(selected):
        dest=out/f'episode-{k:02d}';dest.mkdir();worker=Path(r['root']);manifest=read(pin(capture/'manifest.json'))
        binary=pin(capture/'q2combat-export.exe');assert sha(binary)==manifest['exporter_sha256'].lower()
        trace=pin(worker/'bot.jsonl');server=pin(worker/'server.log');reset=pin(worker/'reset-expectation.json')
        recipe=read(v['context']['source_plan'])['tasks'][0]['episode']['recipe']
        reward=pin(Path(__file__).resolve().parents[1]/recipe['reward_config']);assert sha(reward)==manifest['reward_config_sha256'].lower()
        goal=r['goal_stop']
        cmd=[str(binary),'--trace',str(trace),'--out',str(dest/'native'),'--worker','rules','--episode',str(v['context']['seed']),
             '--end-reason','combat_goal_complete','--server-log',str(server),'--client-name','SoloRetreatBot','--require-execution',
             '--reset-expectation',str(reset),'--synchronous','--goal-observed-frame',str(goal['observed_frame']),
             '--goal-kill-frame',str(goal['kill_frame']),'--reward-config',str(reward)]
        with (dest/'export.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        native=read(pin(dest/'native/report.json'));assert native['command_proof']['accepted'] and native['observed_reset_confirmed']
        steps=load_steps(pin(dest/'native/steps.jsonl'))
        steps=[s for s in steps if s['observation']['identity']['frame']>=98 and s['observation']['identity']['life']==1 and s['observation']['health']>0]
        frames=[s['observation']['identity']['frame'] for s in steps]
        assert frames==list(range(frames[0],frames[-1]+1))
        inp=dest/'observations.jsonl';enriched=dest/'enriched.jsonl'
        with inp.open('x') as f:
            for s in steps:f.write(json.dumps(s['observation'])+'\n')
        subprocess.run([str(a.route_exporter.resolve()),'--input',str(inp),'--output',str(enriched),'--game-dir',str(a.game_dir.resolve())],check=True)
        obs=load_steps(enriched);assert len(obs)==len(steps)
        features=np.asarray([o['features'] for o in obs],dtype=np.float32);targets=[];indices=[];yaw=[]
        for i,s in enumerate(steps):
            frame=s['observation']['identity']['frame'];proof=s.get('server_execution') or {}
            if frame<100 or frame>=goal['kill_frame'] or not s.get('native_step') or not proof.get('matched') or not proof.get('window_exclusive'):continue
            action=s['applied_action'];command=s['sent_command'];assert s['owner']=='rules'
            assert abs(action['forward']-command['Forward']/400)<1e-6 and abs(action['side']-command['Side']/400)<1e-6
            targets.append([action['forward'],action['side']]);indices.append(i);yaw.append(action['yaw_delta_degrees'])
        assert indices and features.shape[1]==1554 and np.isfinite(features).all()
        pack=dest/'movement.npz';np.savez_compressed(pack,features=features,indices=np.asarray(indices,dtype=np.int64),
            movement=np.asarray(targets,dtype=np.float32),applied_yaw_degrees=np.asarray(yaw,dtype=np.float32))
        split='diagnostic' if v['context']['seed'] in diagnostic else 'train'
        episodes.append(dict(map=v['context']['map'],seed=v['context']['seed'],split=split,pack=str(pack),pack_sha256=sha(pack),
            rows=len(indices),context_rows=len(steps),source_variant=v['index'],source_trace_sha256=sha(trace),
            native_export_report_sha256=sha(dest/'native/report.json')))
        print(v['context']['map'],v['context']['seed'],split,len(indices),'movement labels',flush=True)
    assert all(sha(Path(p))==h for p,h in pins.items())
    save(out/'report.json',dict(version='combat_movement_demonstrations_v1',state='passed',feature_version='combat_features_v13_memory_static_routes',
        episodes=episodes,inputs=pins,selection='Successful alive TRAIN rules trajectories; labels are executed commands, not optimal-action claims.',
        diagnostic_split='Largest successful teacher seed per represented map; no frame mixing or checkpoint selection.',
        ppo_eligible=False,neural_execution=False,training_performed=False,promotion=False))


if __name__=='__main__':main()

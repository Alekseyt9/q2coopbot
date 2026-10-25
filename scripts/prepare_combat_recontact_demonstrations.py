"""Executed successful recontact examples; curated BC data, never on-policy PPO."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from audit_combat_contact_loss_probe import load_steps
from audit_combat_route_motion import command_yaw
from evaluate_registered_combat_models import read,save,sha

VERSION='combat_executed_recontact_demonstrations_v1'
IDENTITY=('map','connection','spawncount','actor','life')
NONRESPAWNING={'monster_parasite','monster_gunner','monster_infantry','monster_soldier',
               'monster_soldier_light','monster_soldier_ss'}


def route_matches(observation,entity,last_seen):
    frame=observation['identity']['frame'];group=observation.get('remembered_static_routes') or {}
    if observation.get('enemies') or group.get('frame')!=frame:return False
    if group.get('version')!='combat_observed_memory_static_routes_v1':return False
    memory=[m for m in observation.get('remembered_threats',[]) if (m['id'],m['class'])==entity
            and m['age_frames']==frame-last_seen]
    return len(memory)==1 and any(h['observed_id']==entity[0] and h['memory_age_frames']==frame-last_seen
        and h['static_geometry_known'] and h['static_route_found'] and h.get('next_static_waypoint_relative') is not None
        for h in group.get('hints',[]))


def recontacts(steps,rows,outcomes):
    by_frame={s['observation']['identity']['frame']:s for s in steps}
    admitted={r['frame']:r for r in rows};seen={};windows=[]
    deaths=defaultdict(list)
    for outcome in outcomes:
        if not outcome.get('available'):continue
        for event in outcome.get('events',[]):
            if event.get('health_before',0)>0 and event.get('health_after',1)<=0 and event.get('target_class') in NONRESPAWNING:
                deaths[event['target']].append(outcome['step'])
    for frame in sorted(admitted):
        step=by_frame.get(frame)
        if step is None:continue
        o=step['observation']
        for enemy in o.get('enemies',[]):
            key=(enemy['id'],enemy['class']);last=seen.get(key);seen[key]=frame
            if last is None or not 3<=frame-last<=200:continue
            span=[by_frame.get(f) for f in range(last,frame+1)]
            if any(s is None or s['observation']['identity']['frame'] not in admitted for s in span):continue
            identity=tuple(o['identity'][k] for k in IDENTITY)
            if any(tuple(s['observation']['identity'][k] for k in IDENTITY)!=identity
                   or s['observation']['health']<=0 or s['owner']!='provider' or not s.get('native_step')
                   or not (s.get('server_execution')or{}).get('matched')
                   or not (s.get('server_execution')or{}).get('window_exclusive') for s in span):continue
            # The first native death must follow rediscovery. This also rejects
            # a corpse reappearing as an apparent enemy. These fixed fixtures
            # contain no respawning/resurrecting/spawning monster classes.
            death=min(deaths.get(key[0],[]),default=-1)
            if death<=step['index']:continue
            active=[s['observation']['identity']['frame'] for s in span[1:-1]
                    if route_matches(s['observation'],key,last)]
            if active:windows.append(dict(entity=key[0],observed_class=key[1],last_seen=last,
                reobserved=frame,native_death_step=death,route_frames=active))
    return windows


def world_movement(forward,side,yaw):
    angle=yaw*2*math.pi/65536;c,s=math.cos(angle),math.sin(angle)
    return [forward*c+side*s,forward*s-side*c]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--screens',type=Path,nargs='+',required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();out=a.out.resolve();out.mkdir(parents=True)
    pins={};episodes=[];physical_seen=set();rejected=Counter();poses={s:Counter() for s in ('train','diagnostic')}
    def pin(p):p=Path(p).resolve();pins[str(p)]=sha(p);return p
    candidates=[]
    for root in a.screens:
        root=root.resolve();spec=read(pin(root/'spec.json'));quality=read(pin(root/'acceptance.json'));pool=read(pin(root/'pool/report.json'))
        assert quality['state']=='passed' and pool['state']=='complete' and pool['source_unchanged']
        for p,h in quality['inputs'].items():assert sha(p)==h
        scores={(r['controller'],r['seed']):r for r in quality['results']};jobs={j['plan_index']:j for j in pool['jobs']}
        for v in spec['variants']:
            if v['label']=='rules':continue
            role={'zero':'candidate','source':'source','original':'original_source'}[v['label']]
            if scores[role,v['context']['seed']].get('inference_verified') is False:
                rejected['inference_not_verified']+=1;continue
            if not scores[role,v['context']['seed']]['autonomous_goal']:continue
            assert v['context']['instance']['split']=='train'
            if any(m['class'] not in NONRESPAWNING for m in v['context']['instance']['monsters']):
                rejected['unsupported_monster_lifecycle']+=1;continue
            d=root/f'case-{v["index"]:02d}';meta=read(pin(d/'cuda/report.json'))
            if meta['feature_version']!='combat_features_v13_memory_static_routes':continue
            assert sha(v['model'])==meta['model_sha256']==v['model_sha256'];pin(v['model'])
            proof=read(pin(d/'cuda/cuda-verification.json'))
            assert proof['state']=='passed' and proof['device']=='cuda' and sha(d/'cuda/cuda-verification.json')==meta['cuda_verification_sha256']
            for p,h in meta['source_sha256'].items():assert sha(p)==h
            rows=load_steps(pin(d/'cuda/rollout.jsonl'));history=load_steps(pin(d/'cuda/sequence.jsonl'))
            assert sha(d/'cuda/rollout.jsonl')==meta['rollout_sha256'] and sha(d/'cuda/sequence.jsonl')==meta['sequence_sha256']
            path=next((d/'native').glob('replay-*/steps.jsonl'));steps=load_steps(pin(path))
            outcomes=load_steps(pin(path.parent/'server_outcomes.jsonl'))
            capture=Path(jobs[v['index']]['root']);report=read(pin(capture/'report.json'));manifest=read(pin(capture/'manifest.json'));r=report['results'][0]
            assert report['capture_complete'] and report['provenance_valid'] and r['capture_valid'] and r['dispatch_valid'] and r['seed_confirmed']
            assert r['generated_fixture']==v['context']['instance'] and r['goal_stop'] and not r['death_stop']
            assert manifest['model_weights_sha256'].lower()==v['model_sha256']
            windows=recontacts(steps,rows,outcomes)
            if not windows:rejected['no_verified_recontact']+=1;continue
            assert all(w['reobserved']<r['goal_stop']['kill_frame'] for w in windows)
            label_frames=sorted({f for w in windows for f in w['route_frames']})
            native={s['observation']['identity']['frame']:s for s in steps};admitted={r['frame']:r for r in rows}
            # Physical duplicates across models are one demonstration, not
            # independent new experience. Source policy samples are retained
            # separately, while the fingerprint uses observations/commands.
            physical=[dict(frame=f,observation={k:native[f]['observation'].get(k) for k in
                ('position','view_angles','remembered_threats','remembered_static_routes')},command=native[f]['sent_command']) for f in label_frames]
            digest=hashlib.sha256(json.dumps(dict(map=v['context']['map'],seed=v['context']['seed'],physical=physical),sort_keys=True).encode()).hexdigest()
            if digest in physical_seen:rejected['duplicate_physical_trajectory']+=1;continue
            physical_seen.add(digest)
            candidates.append(dict(variant=v,root=root,directory=d,history=history,native=native,admitted=admitted,
                windows=windows,label_frames=label_frames,physical_sha256=digest))
    assert candidates,'No verified recontact examples found'
    worlds=defaultdict(set)
    for c in candidates:worlds[c['variant']['context']['map']].add(c['variant']['context']['seed'])
    diagnostic={(m,s) for m,seeds in worlds.items() for s in sorted(seeds)[-max(1,math.ceil(len(seeds)*.25)): ]}
    for i,c in enumerate(candidates):
        v=c['variant'];context=v['context'];split='diagnostic' if (context['map'],context['seed']) in diagnostic else 'train'
        history=c['history'];frames=[r['frame'] for r in history]
        assert frames==list(range(frames[0],frames[-1]+1)) and frames[0]<=98
        assert all(r['seed']==context['seed'] and len(r['features'])==1554 for r in history)
        location={f:j for j,f in enumerate(frames)};labels=[];local=[];world=[];yaws=[];vertical=[];indices=[]
        dest=out/f'episode-{i:02d}';dest.mkdir()
        for frame in c['label_frames']:
            s=c['native'][frame];o=s['observation'];cmd=s['sent_command'];action=s['applied_action']
            assert action['identity']==o['identity'] and o['identity']['life']==1 and not o['enemies']
            assert abs(action['forward']-cmd['Forward']/400)<1e-6 and abs(action['side']-cmd['Side']/400)<1e-6
            yaw=command_yaw(o,action);assert yaw==s['next_observation']['view_angles'][1]
            fwd,side=cmd['Forward']/400,cmd['Side']/400;movement=world_movement(fwd,side,yaw)
            # Geometry unit checks; this is a coordinate transform, not NN.
            assert abs(sum(t*t for t in movement)-fwd*fwd-side*side)<1e-12
            pose=('release','jump','crouch').index(action['vertical']);assert cmd['Up']==(0,400,-400)[pose]
            assert np.allclose(history[location[frame]]['features'],c['admitted'][frame]['features'],atol=0,rtol=0)
            indices.append(location[frame]);local.append([fwd,side]);world.append(movement);yaws.append(yaw);vertical.append(pose);poses[split][action['vertical']]+=1
            labels.append(dict(frame=frame,identity=o['identity'],source_sample=c['admitted'][frame]['sample'],
                applied=action,command=cmd,applied_yaw_short=yaw,world_movement=movement,
                qualifying_windows=[j for j,w in enumerate(c['windows']) if frame in w['route_frames']]))
        pack=dest/'recontact.npz'
        np.savez_compressed(pack,features=np.asarray([r['features'] for r in history],dtype=np.float32),
            frames=np.asarray(frames,dtype=np.int64),indices=np.asarray(indices,dtype=np.int64),
            local_movement=np.asarray(local,dtype=np.float32),world_movement=np.asarray(world,dtype=np.float32),
            applied_yaw_short=np.asarray(yaws,dtype=np.int16),vertical=np.asarray(vertical,dtype=np.int64))
        with (dest/'labels.jsonl').open('x') as f:
            for label in labels:f.write(json.dumps(label,allow_nan=False)+'\n')
        save(dest/'windows.json',dict(windows=c['windows']))
        episodes.append(dict(map=context['map'],seed=context['seed'],split=split,rows=len(labels),context_rows=len(history),
            pack=str(pack),pack_sha256=sha(pack),labels=str(dest/'labels.jsonl'),labels_sha256=sha(dest/'labels.jsonl'),
            windows=str(dest/'windows.json'),windows_sha256=sha(dest/'windows.json'),source_screen=str(c['root']),source_variant=v['index'],
            source_model=v['model'],source_model_sha256=v['model_sha256'],source_sequence=str(c['directory']/'cuda/sequence.jsonl'),
            source_steps=str(next((c['directory']/'native').glob('replay-*/steps.jsonl'))),physical_sha256=c['physical_sha256']))
    assert all(sha(p)==h for p,h in pins.items())
    counts={s:dict(episodes=sum(e['split']==s for e in episodes),rows=sum(e['rows'] for e in episodes if e['split']==s),
        worlds=len({(e['map'],e['seed']) for e in episodes if e['split']==s}),poses=dict(poses[s])) for s in ('train','diagnostic')}
    assert not ({(e['map'],e['seed']) for e in episodes if e['split']=='train'} & {(e['map'],e['seed']) for e in episodes if e['split']=='diagnostic'})
    save(out/'report.json',dict(version=VERSION,state='passed',feature_version='combat_features_v13_memory_static_routes',
        episodes=episodes,counts=counts,maps=sorted(worlds),inputs=pins,rejected=dict(rejected),
        ppo_eligible=False,neural_execution=False,training_performed=False,promotion=False,
        split_rule='Top ceil25percent seed worlds per represented map are diagnostic; all models of a world share split; no frame mixing.',
        action_coordinates='Native applied command view plus world movement preserved. Local F/S alone must not be treated as view-invariant labels.',
        limitations='Curated autonomous successful trajectories and later native kills establish executed recontact, not causal optimality. Coverage limited to represented maps, poses and short gaps; future labels only curate offline BC, never become observations.'))
    print(json.dumps(dict(counts=counts,maps=sorted(worlds),rejected=dict(rejected))),flush=True)


if __name__=='__main__':main()

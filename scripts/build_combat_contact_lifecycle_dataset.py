"""Keep lost-target states, observable memory directions and known native horizons."""
import argparse
import collections
import json
import math
from pathlib import Path
import numpy as np
from evaluate_registered_combat_models import read,save,sha
from native_combat_release import release_boundary
from report_combat_machinegun_hits import scan
from report_registered_machinegun_outcomes import outcome


def wrap(value):return (value+180)%360-180


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path);h=pins[key] if key in pins else sha(path)
        assert expected is None or h==expected,str(path);pins[key]=h;return path
    pin(__file__);p=read(pin(a.protocol));assert p['version']=='combat_target_lifecycle_dataset_v1'
    role_root=Path(p['world_roles_dataset']);rc=read(pin(role_root/'complete.json'))
    roles=read(pin(role_root/'report.json',rc['report_sha256']));assert roles['version']=='combat_auxiliary_world_roles_v1'
    world_roles={c['seed']:c['split'] for c in roles['worlds']}
    assert collections.Counter(world_roles.values())==dict(train=24,development=8)
    root=Path(p['source_corpus']);closed=read(pin(root/'complete.json'));assert closed['native_cases']==closed['cuda_cases']==32
    assert root.resolve()==Path(roles['source_corpus']).resolve() and sha(root/'complete.json')==roles['source_corpus_complete_sha256']
    assert p['inventory_transport_version']=='native_frame_inventory_sparse_v2'
    pool=read(pin(root/'pool/report.json',closed['pool_sha256']));assert pool['source_unchanged']
    assert {j['seed'] for j in pool['jobs']}==set(world_roles)
    fields=('received_health_damage','monster_health_damage','self_health_damage','teammate_health_damage','monster_kills','deaths')
    horizons=p['horizons_frames'];assert horizons==[1,5,10]
    a.out.mkdir(parents=True);(a.out/'chunks').mkdir();chunks=[];sequences=[];counts=collections.Counter()
    for job in sorted(pool['jobs'],key=lambda j:j['job']):
        seed=job['seed'];split=world_roles[seed];folder=root/'processing-train'/f"case-{job['job']:04d}"
        meta=read(pin(folder/'cuda/report.json'));assert meta['model_sha256']==p['source_weights_sha256'] and meta['observation_timing_contract']=='frozen_world_v1'
        proof=read(pin(folder/'cuda/cuda-verification.json',meta['cuda_verification_sha256']));assert proof['state']=='passed' and proof['device']=='cuda'
        capture=read(pin(Path(job['root'])/'report.json'));result=capture['results'][0]
        assert capture['capture_complete'] and capture['provenance_valid'] and result['seed']==seed and result['dispatch_valid']
        manifest=read(pin(Path(job['root'])/'manifest.json'));log=pin(Path(result['root'])/'server.log')
        assert manifest['inventory_transport_version']==p['inventory_transport_version'] and manifest['inventory_drop_once_frame']==0
        with log.open(encoding='utf-8-sig') as stream:generation,release,_=release_boundary(stream,manifest,seed)
        stops=[v[key] for v,key in ((result['goal_stop'],'kill_frame'),(result['death_stop'],'death_frame')) if v]
        stop=min(stops) if stops else None
        sp=next((folder/'native').glob('replay-*/steps.jsonl'))
        all_steps=list(map(json.loads,pin(sp,meta['source_sha256'][str(sp.resolve())]).open(encoding='utf-8-sig')))
        ep=sp.with_name('server_outcomes.jsonl');effect_list=list(map(json.loads,pin(ep,meta['source_sha256'][str(ep.resolve())]).open(encoding='utf-8-sig')))
        effects={e['step']:e for e in effect_list};assert len(effects)==len(effect_list)
        rows={r['frame']:r for r in map(json.loads,pin(folder/'cuda/rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig'))}
        pin(folder/'cuda/sequence.jsonl',meta['sequence_sha256'])
        byframe={s['observation']['identity']['frame']:s for s in all_steps if s['observation']['identity']['life']==1}
        active={f:s for f,s in byframe.items() if s['owner']=='provider' and s.get('sample') and s.get('native_step')}
        def accepted(s):
            if s is None:return False
            o=s['observation'];f=o['identity']['frame'];dispatch=s.get('server_execution') or {};ns=s['native_step'];nxt=s.get('next_observation');effect=effects.get(s['index'])
            return bool(o['health']>0 and o['identity']['spawncount']==generation and dispatch.get('matched') and dispatch.get('window_exclusive') and
                dispatch.get('recovery_commands')==0 and ns['begin_frame']==f and ns['end_frame']==f+1 and effect and effect['available'] and
                nxt and nxt['identity']['life']==1 and nxt['identity']['spawncount']==generation and nxt['identity']['frame']==f+1)
        windows={tuple(s['native_step'][k] for k in ('spawncount','begin_frame','actor','sequence')):s for s in active.values() if accepted(s)}
        with log.open(encoding='utf-8-sig') as stream:shots=scan(stream,set(windows))
        shot_counts=collections.defaultdict(collections.Counter)
        for shot in shots:
            if not shot['eligible']:continue
            f=windows[shot['window']]['observation']['identity']['frame'];kind=outcome(shot)
            shot_counts[f]['shots']+=1
            if kind=='live_monster_damage':shot_counts[f]['damage']+=1
            if kind=='solid_contact':shot_counts[f]['solid']+=1
            if kind=='corpse_contact':shot_counts[f]['corpse']+=1
        records=[];bbox={};phase_by_frame={}
        for frame,s in sorted(byframe.items()):
            o=s['observation']
            for enemy in o['enemies']:bbox[enemy['id']]=(enemy['class'],enemy.get('observed_solid',0))
            if frame not in rows or frame not in active or frame<release or (stop is not None and frame>=stop) or not accepted(s):continue
            sample=s['sample'];row=rows[frame];assert row['sample']==sample and row['seed']==seed
            assert o['inventory_age_frames']==0 and o['inventory'] is not None,'Stale/unknown inventory in accepted source row'
            memory=sorted(o.get('remembered_threats') or [],key=lambda t:(t['age_frames'],t['id']))
            fresh=bool(memory and memory[0]['age_frames']<=p['fresh_memory_max_age_frames'])
            phase=0 if o['enemies'] else 1 if fresh else 2 if memory else 3
            phase_by_frame[frame]=phase
            reference=s['action'].get('target_entity',0) or (memory[0]['id'] if memory else 0)
            direction=[0.,0.];direction_known=False
            if phase==1 and not sample.get('target',0) and sample.get('aim_mode',0)==0:
                threat=memory[0];old=bbox.get(threat['id']);solid=old[1] if old and old[0]==threat['class'] else 0
                bottom=-((solid>>5)&31)*8;top=((solid>>10)&63)*8-32
                if solid not in (0,31) and solid&31 and top>bottom:
                    height=(top+bottom)/2 if top-bottom<16 else min(max(22,bottom+8),top-8)
                    x,y,z=threat['last_observed_relative'];z+=height-(-2 if o['ducked'] else 22)
                    if math.hypot(x,y)>1:
                        desired=(-math.degrees(math.atan2(z,math.hypot(x,y))),math.degrees(math.atan2(y,x)))
                        direction=[wrap(d-o['view_angles'][i]*360/65536) for i,d in enumerate(desired)];direction_known=True
            ys=[];known=[];end_states=[];end_known=[];reference_known=[]
            for horizon in horizons:
                valid=stop is None or frame+horizon<=stop;pieces=[];last=None
                for f in range(frame,frame+horizon):
                    step=active.get(f)
                    if not accepted(step):valid=False;break
                    pieces.append(effects[step['index']]);last=step['next_observation']
                values=[sum(e[k] for e in pieces) for k in fields]
                values.extend(sum(shot_counts[f][k] for f in range(frame,frame+horizon)) for k in ('shots','damage','solid','corpse'))
                assert all(math.isfinite(v) for v in values)
                known.append(valid);ys.append(values if valid else [0.]*10)
                endpoint=valid and last is not None and last['health']>0
                end_known.append(endpoint);reference_known.append(endpoint and bool(reference))
                end_states.append([float(bool(last['enemies'])),float(any(e['id']==reference for e in last['enemies'])),last['health'],
                    *[b-a for a,b in zip(o['position'],last['position'])]] if endpoint else [0.]*6)
            assert known[0]
            candidate=sample.get('aim_candidate') or {}
            records.append(dict(features=row['features'],source_action=candidate.get('source_latent') or sample['latent'],action=sample['latent'],
                categories=[sample.get(k,0) for k in ('target','aim_mode','vertical','weapon')],attack=float(sample['attack']),phase=phase,
                recovery_direction=direction,recovery_known=direction_known,effects=ys,known=known,end_states=end_states,
                end_known=end_known,reference_known=reference_known,world=seed,frame=frame,step=s['index']))
            counts[f'{split}_{p["phases"][phase]}']+=1;counts[f'{split}_recovery_labels']+=direction_known
            for h,valid in zip(horizons,known):counts[f'{split}_known_h{h}']+=valid
        for frame,phase in sorted(phase_by_frame.items()):
            if phase!=1 or phase_by_frame.get(frame-1)!=0:continue
            lost=[];f=frame;reason='trace_end_unknown'
            while f in byframe:
                s=byframe[f];o=s['observation']
                if stop is not None and f>=stop:reason='verified_task_end';break
                if o['health']<=0:reason='death';break
                if s['owner']!='provider':reason='handoff';break
                if o['enemies']:reason='reacquired';break
                if f not in phase_by_frame:reason='unaccepted_or_missing_row';break
                lost.append(f);f+=1
            sequences.append(dict(seed=seed,split=split,map=result['generated_fixture']['map'],start_frame=frame,
                last_visible_frame=frame-1,lost_frames=len(lost),end_frame=f,end_reason=reason,
                shots_while_lost=sum(shot_counts[f]['shots'] for f in lost),solid_shots_while_lost=sum(shot_counts[f]['solid'] for f in lost)))
        assert records
        dtypes=dict(features=np.float32,source_action=np.float32,action=np.float32,categories=np.int64,attack=np.float32,phase=np.uint8,
            recovery_direction=np.float32,recovery_known=np.bool_,effects=np.float32,known=np.bool_,end_states=np.float32,
            end_known=np.bool_,reference_known=np.bool_,world=np.int64,frame=np.int32,step=np.int32)
        data={k:np.asarray([r[k] for r in records],dtype=d) for k,d in dtypes.items()}
        assert data['features'].shape==(len(records),1147) and data['effects'].shape==(len(records),3,10)
        path=a.out/'chunks'/f"{job['job']:04d}-{split}.npz";np.savez_compressed(path,**data)
        chunks.append(dict(path=str(path.resolve()),sha256=sha(path),rows=len(records),split=split,case=dict(seed=seed,
            map=result['generated_fixture']['map'],folder=str(folder.resolve()),source_rollout_sha256=meta['rollout_sha256'],source_sequence_sha256=meta['sequence_sha256'])))
        save(a.out/'progress.json',dict(stage='building',cases_done=len(chunks),cases=32))
    for path,h in pins.items():assert sha(path)==h,path
    save(a.out/'sequences.json',sequences)
    report=dict(state='complete',version='combat_target_lifecycle_dataset_v1',protocol_sha256=sha(a.protocol),chunks=chunks,counts=dict(counts),
        sequences=len(sequences),sequence_endings=dict(collections.Counter(s['end_reason'] for s in sequences)),sequences_sha256=sha(a.out/'sequences.json'),
        compressed_bytes=sum(Path(c['path']).stat().st_size for c in chunks),inputs=pins,neural_execution=False,new_games=False,
        training_performed=False,ordinary_training_eligible=False,promotion=False,
        scope='Auxiliary whole-world split source trajectories including lost visibility. Memory-direction labels use past observations only and are not an optimal action teacher. Native future labels explicitly masked, no terminal/handoff zero filling.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    save(a.out/'progress.json',dict(stage='complete',cases=32))
    print(dict(rows=sum(c['rows'] for c in chunks),counts=dict(counts),sequences=report['sequences'],endings=report['sequence_endings'],compressed_bytes=report['compressed_bytes']))


if __name__=='__main__':main()

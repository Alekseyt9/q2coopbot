"""Observed provider weapon intents and native responses in sealed captures.

No model execution, training, hidden inventory inference, or shot attribution.
"""
import argparse
import collections
import json
import pathlib

from combat_equipped_weapon_features import CATEGORIES, equipped_weapon_index
from process_combat_architecture_pool import read, save, sha
from process_registered_combat_pool_cuda import select_learned_jobs


def audit_frames(frames):
    counts=collections.Counter();events=[];pending=None;previous=None;streak=maximum=0
    def close(status,frame=None):
        nonlocal pending
        if pending:
            events.append(dict(pending,status=status,observed_frame=frame,
                               observed_delay_frames=frame-pending['requested_frame'] if frame is not None else None))
            pending=None
    for observation,applied,owner in frames:
        assert owner in ('provider','rules'), 'Unknown controller owner'
        identity=observation['identity'];frame=identity['frame']
        world=tuple(identity.get(k) for k in ('map','spawncount','connection','actor','life'))
        if previous is not None and previous[0]==world:
            assert frame>previous[1], 'Non-increasing observation frame'
        contiguous=previous is not None and previous==(world,frame-1)
        if previous is not None and not contiguous:
            close('observation_gap');streak=0
        previous=(world,frame)
        if owner!='provider':
            close('ownership_changed');streak=0;continue
        assert applied is not None, 'Provider applied intent is required'
        current=equipped_weapon_index(observation)
        counts['provider_frames']+=1
        counts['unknown_equipped_frames']+=int(current==0)
        if pending and current==pending['requested_index']:
            close('observed_equipped',frame)
        request=applied.get('weapon') or ''
        request_index=equipped_weapon_index(dict(weapon=request))
        if request:
            assert request_index, 'Unknown provider weapon request'
            counts['request_frames']+=1
            if request_index==current:
                counts['already_equipped_request_frames']+=1
                close('superseded_by_already_equipped_request')
            elif not pending or pending['requested_index']!=request_index:
                close('superseded')
                pending=dict(requested_frame=frame,requested_index=request_index,
                             requested_weapon=CATEGORIES[request_index])
        visible=any(e.get('clear_shot') for e in observation.get('enemies') or [])
        outside=current not in (0,1,2,3,4)
        held=outside and visible and not request
        counts['outside_pilot_fire_set_frames']+=int(outside)
        counts['outside_pilot_fire_set_keep_clear_enemy_frames']+=int(held)
        streak=(streak+1 if contiguous else 1) if held else 0
        maximum=max(maximum,streak)
    close('capture_ended')
    return dict(counts=dict(counts),max_outside_pilot_keep_clear_streak=maximum,
                request_events=events)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pool',type=pathlib.Path,required=True)
    parser.add_argument('--plan-index',type=int,required=True)
    parser.add_argument('--out',type=pathlib.Path,required=True)
    args=parser.parse_args();assert not args.out.exists()
    pool_path=args.pool/'report.json';digest=sha(pool_path);pool=read(pool_path)
    cases=[]
    for job in select_learned_jobs(pool,args.plan_index):
        capture_path=pathlib.Path(job['root'])/'report.json';capture=read(capture_path)
        assert capture['capture_complete'] and capture['provenance_valid'] and len(capture['results'])==1
        result=capture['results'][0];assert result['capture_valid'] and result['dispatch_valid']
        manifest_path=capture_path.parent/'manifest.json';manifest=read(manifest_path)
        assert manifest['synchronous'] and manifest['fixed_world_hold'] and manifest['release_game_frame']>0
        trace=pathlib.Path(result['root'])/'bot.jsonl';trace_hash=sha(trace)
        frames=[];seen=set()
        for line in trace.open(encoding='utf-8-sig'):
            row=json.loads(line);control=row.get('combat_policy') or {}
            owner=(control.get('selection') or {}).get('owner')
            if not owner:continue
            observation=control.get('observation');assert observation, 'Missing owned observation'
            identity=observation['identity']
            if identity['life']!=1 or observation['health']<=0 or identity['frame']<manifest['release_game_frame']:continue
            key=tuple(identity.get(k) for k in ('map','spawncount','connection','actor','life','frame'))
            assert key not in seen;seen.add(key)
            frames.append((observation,control.get('applied'),owner))
        case=audit_frames(frames)
        assert sha(trace)==trace_hash
        case.update(job=job['job'],seed=job['seed'],goal=bool(result['goal_stop']),
                    trace_sha256=trace_hash,capture_sha256=sha(capture_path),manifest_sha256=sha(manifest_path))
        cases.append(case)
    assert sha(pool_path)==digest
    totals=dict(sum((collections.Counter(c['counts']) for c in cases),collections.Counter()))
    statuses=collections.Counter(e['status'] for c in cases for e in c['request_events'])
    delays=[e['observed_delay_frames'] for c in cases for e in c['request_events'] if e['status']=='observed_equipped']
    save(args.out,dict(version='registered_weapon_choice_audit_v1',state='complete',
        pool_sha256=digest,selected_plan_index=args.plan_index,script_sha256=sha(__file__),
        feature_oracle_sha256=sha(pathlib.Path(__file__).with_name('combat_equipped_weapon_features.py')),
        cases=cases,totals=totals,request_statuses=dict(statuses),
        observed_delay_frames_mean=sum(delays)/len(delays) if delays else None,
        observed_delay_frames_max=max(delays) if delays else None,
        max_outside_pilot_keep_clear_streak=max(c['max_outside_pilot_keep_clear_streak'] for c in cases),
        scope='Post-release first-life provider intents to observed equipment, not network dispatch/cooldown or shot attribution. Gaps, ownership changes, superseded and unfinished requests censored. Pilot fire set is Blaster/Shotgun/Super Shotgun/Machinegun; inventory-available choices are not restricted.'))


if __name__=='__main__':main()

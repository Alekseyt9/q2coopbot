"""Observed stock weapon animation phases and transitions; no shot inference."""
import argparse
import collections
import json
import pathlib
import re

from audit_registered_weapon_choices import audit_frames
from combat_equipped_weapon_features import CATEGORIES,equipped_weapon_index
from process_combat_architecture_pool import read, save, sha

EXPECTED={'Blaster':(4,8,52,55),'Shotgun':(7,18,36,39),
          'Super Shotgun':(6,17,57,61),'Machinegun':(3,5,45,49)}
FUNCTIONS={'Blaster':'Blaster','Shotgun':'Shotgun','Super Shotgun':'SuperShotgun','Machinegun':'Machinegun'}


def observed_phase(weapon,frame):
    if weapon not in EXPECTED: return 'unsupported_weapon'
    assert isinstance(frame,int) and frame>=0
    activate,fire,idle,lower=EXPECTED[weapon]
    if frame<=activate: return 'activation_animation'
    if frame<=fire: return 'firing_animation'
    if frame<=idle: return 'idle_animation'
    if frame<=lower: return 'lowering_animation'
    return 'outside_stock_animation_bounds'


def audit(frames):
    counts=collections.Counter();previous=None;transitions=[]
    for observation,applied,owner in frames:
        identity=observation['identity'];frame=identity['frame']
        world=tuple(identity[k] for k in ('map','spawncount','connection','actor','life'))
        assert owner in ('provider','rules')
        current=equipped_weapon_index(observation)
        contiguous=previous is not None and previous['world']==world and previous['frame']==frame-1
        if previous is not None and previous['world']==world: assert frame>previous['frame']
        if owner=='provider':
            assert applied is not None
            phase=observed_phase(CATEGORIES[current],observation['gun_frame'])
            counts['provider_frames']+=1;counts['phase/'+phase]+=1
            counts['applied_attack/'+phase]+=int(bool(applied.get('attack')))
            if contiguous and previous['owner']=='provider' and previous['weapon'] and current and previous['weapon']!=current:
                counts['observed_equipped_changes']+=1
                transitions.append(dict(frame=frame,previous_weapon_index=previous['weapon'],weapon_index=current,
                    observed_phase=phase))
            if not contiguous or previous['owner']!='provider': counts['provider_segment_starts']+=1
        previous=dict(world=world,frame=frame,owner=owner,weapon=current)
    return dict(counts=dict(counts),observed_transitions=transitions,weapon_requests=audit_frames(frames))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pool',type=pathlib.Path,required=True)
    parser.add_argument('--execution',type=pathlib.Path,required=True)
    parser.add_argument('--native-weapon-source',type=pathlib.Path,required=True)
    parser.add_argument('--out',type=pathlib.Path,required=True)
    args=parser.parse_args();assert not args.out.exists()
    source_sha=sha(args.native_weapon_source);source=args.native_weapon_source.read_text(encoding='utf-8')
    for name,function in FUNCTIONS.items():
        match=re.search(r'\bWeapon_'+function+r'\s*\([^)]*\)\s*\{.*?Weapon_Generic\(ent,\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)',source,re.S)
        assert match and tuple(map(int,match.groups()))==EXPECTED[name], 'Stock phase definitions differ'
    digest=sha(args.pool/'report.json');pool=read(args.pool/'report.json')
    assert pool['state']=='complete' and pool['source_unchanged']
    execution_sha=sha(args.execution);execution=read(args.execution)
    assert execution['state']=='complete' and execution['pool_sha256']==digest
    index=execution['selected_plan_index'];cases=[]
    jobs={job['job']:job for job in pool['jobs'] if job['plan_index']==index}
    assert len(jobs)==len(execution['cases'])
    for binding in execution['cases']:
        job=jobs[binding['job']];assert not job['error'] and job['mode']=='learned'
        capture_path=pathlib.Path(job['root'])/'report.json'
        assert sha(capture_path)==binding['capture_sha256']
        capture=read(capture_path);trace=pathlib.Path(capture['results'][0]['root'])/'bot.jsonl'
        assert sha(trace)==binding['trace_sha256'];frames=[];seen=set()
        with trace.open(encoding='utf-8-sig') as stream:
            for line in stream:
                control=json.loads(line).get('combat_policy') or {}
                owner=(control.get('selection') or {}).get('owner')
                if not owner: continue
                observation=control['observation'];identity=observation['identity']
                if identity['life']!=1 or observation['health']<=0 or identity['frame']<binding['network_release_frame']: continue
                assert identity['spawncount']==binding['release_generation']
                key=tuple(identity[k] for k in ('map','spawncount','connection','actor','life','frame'))
                assert key not in seen;seen.add(key)
                frames.append((observation,control.get('applied'),owner))
        result=audit(frames)
        assert result['counts']['provider_frames']==binding['counts']['provider_frames']
        assert sha(trace)==binding['trace_sha256']
        cases.append(dict(job=job['job'],seed=job['seed'],trace_sha256=binding['trace_sha256'],
            network_release_frame=binding['network_release_frame'],**result))
    assert sha(args.pool/'report.json')==digest and sha(args.execution)==execution_sha and sha(args.native_weapon_source)==source_sha
    totals=sum((collections.Counter(c['counts']) for c in cases),collections.Counter())
    statuses=collections.Counter(e['status'] for c in cases for e in c['weapon_requests']['request_events'])
    save(args.out,dict(version='registered_observed_weapon_phases_v1',state='complete',pool_sha256=digest,
        execution_sha256=execution_sha,native_source_sha256=source_sha,script_sha256=sha(__file__),
        native_weapon_source=str(args.native_weapon_source.resolve()),selected_plan_index=index,
        cases=cases,totals=dict(totals),request_statuses=dict(statuses),
        scope='Correct network release clock. Observed equipped changes and stock gun-frame animation ranges; not hidden WeaponState, fire-ready guarantees, completed network commands, shot counts, accuracy or causal damage attribution. Pending requests/gaps censored. Native source constants checked; runtime acceptance remains bound by capture manifests.'))
    print(json.dumps(dict(cases=len(cases),totals=totals,request_statuses=statuses)))


if __name__=='__main__': main()

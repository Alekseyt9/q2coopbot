"""Observed native next-frame motion versus logged diagnostic; no inference."""
import argparse
import collections
import json
import math
import pathlib
import re
from audit_registered_execution_actions import read, sha


def summarize(samples):
    def stats(key):
        values=sorted(s[key] for s in samples)
        return dict(count=len(values),mean=sum(values)/len(values),p95=values[min(len(values)-1,int(.95*len(values)))],maximum=values[-1]) if values else dict(count=0)
    return dict(position_error=stats('position_error'),xy_error=stats('xy_error'),velocity_error=stats('velocity_error'),
        above_quarter_unit=sum(s['position_error']>.25 for s in samples),
        above_one_unit=sum(s['position_error']>1 for s in samples))


def measure(rows,release,generation):
    grouped=collections.defaultdict(list)
    for row in rows:
        c=row.get('combat_policy') or {};o=c.get('observation') or {};identity=o.get('identity') or {}
        if identity.get('life')!=1 or o.get('health',0)<=0 or identity.get('frame',0)<release:continue
        assert identity['spawncount']==generation
        grouped[row['observation_frame']].append(row)
    counts=collections.Counter();samples=[]
    for frame, entries in sorted(grouped.items()):
        counts['released_frames']+=len(entries)
        if len(entries)!=1 or len(grouped.get(frame+1,[]))!=1:
            counts['ambiguous_or_missing_next_frame']+=1;continue
        row,nxt=entries[0],grouped[frame+1][0]
        prediction=row.get('pmove_ground_prediction')
        if not prediction:counts['no_supported_prediction']+=1;continue
        assert row['pm_state_known'] and row['pm_type']==row['pm_time']==0
        cmd=row['sent_command']
        if cmd['Msec']!=100:counts['non_tick_command']+=1;continue
        if any(row.get(k)!=nxt.get(k) for k in ('connection','spawncount','map','health','armor')):
            counts['external_damage_or_reset']+=1;continue
        if not nxt['on_ground']:counts['next_frame_airborne']+=1;continue
        distance=lambda a,b:math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
        sample=dict(frame=frame,position_error=distance(prediction['position'],nxt['self']),
            xy_error=distance(prediction['position'][:2],nxt['self'][:2]),
            velocity_error=distance(prediction['velocity'],nxt['self_velocity']),
            requested_movement=bool(cmd['Forward'] or cmd['Side']),contacts=len(prediction['contacts'] or []),
            grounded_vertical_velocity=abs(row['self_velocity'][2])>1,
            before=row['self'],predicted=prediction['position'],observed=nxt['self'],command=cmd)
        assert all(math.isfinite(sample[k]) for k in ('position_error','xy_error','velocity_error'))
        samples.append(sample);counts['measured_pairs']+=1
    return dict(counts=dict(counts),all=summarize(samples),
        commanded_motion=summarize([s for s in samples if s['requested_movement']]),
        contacts=summarize([s for s in samples if s['contacts']]),
        grounded_vertical_velocity=summarize([s for s in samples if s['grounded_vertical_velocity']]),
        worst=sorted(samples,key=lambda s:s['position_error'],reverse=True)[:5],samples=samples)


def main():
    ap=argparse.ArgumentParser();sources=ap.add_mutually_exclusive_group(required=True)
    sources.add_argument('--evaluation',type=pathlib.Path);sources.add_argument('--pool',type=pathlib.Path)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    args=ap.parse_args();assert not args.out.exists()
    if args.evaluation:
        root=args.evaluation.resolve();seal=read(root/'complete.json')
        assert seal['state']=='complete' and not seal['training_performed']
        pool_path=root/'pool/report.json';assert sha(pool_path)==seal['pool_sha256']
        evaluation_sha=sha(root/'complete.json')
    else:
        pool_path=args.pool.resolve()/'report.json';pool_snapshot=read(pool_path)
        assert pool_snapshot['state']=='complete'
        seal=dict(pool_sha256=sha(pool_path),native_cases=pool_snapshot['usable_captures'])
        evaluation_sha=None
    pool=read(pool_path);assert pool['source_unchanged'] and pool['usable_captures']==seal['native_cases']
    assert len(pool['jobs'])==seal['native_cases']
    cases=[]
    pattern=re.compile(r'^sv_test_combat spawncount=(\d+) server_frame=(\d+) g_test_combat_start game_frame=(\d+) ready=1 seed=(\d+)$')
    for job in pool['jobs']:
        assert not job['error']
        cp=pathlib.Path(job['root'])/'report.json';cap=read(cp);assert cap['capture_complete'] and cap['provenance_valid']
        result=cap['results'][0];assert result['capture_valid'] and result['dispatch_valid']
        directory=pathlib.Path(result['root']);trace=directory/'bot.jsonl';log=directory/'server.log'
        td,ld=sha(trace),sha(log)
        release=[tuple(map(int,m.groups())) for line in log.read_text(encoding='utf-8-sig').splitlines() if (m:=pattern.fullmatch(line))]
        assert len(release)==1 and release[0][3]==job['seed']
        with trace.open(encoding='utf-8-sig') as stream:rows=[json.loads(line) for line in stream]
        data=measure(rows,release[0][1],release[0][0])
        assert sha(trace)==td and sha(log)==ld
        cases.append(dict(job=job['job'],seed=job['seed'],plan_index=job['plan_index'],capture_sha256=sha(cp),trace_sha256=td,server_log_sha256=ld,**data))
    assert sha(pool_path)==seal['pool_sha256']
    samples=[s for c in cases for s in c['samples']]
    report=dict(state='complete',version='native_ground_prediction_audit_v1',cases=cases,all=summarize(samples),
        commanded_motion=summarize([s for s in samples if s['requested_movement']]),contacts=summarize([s for s in samples if s['contacts']]),
        evaluation_sha256=evaluation_sha,pool_sha256=seal['pool_sha256'],script_sha256=sha(__file__),training_performed=False,neural_execution=False,
        scope='Unique successive post-release network frames of first life, 100ms commands, unchanged health/armor and grounded next snapshot. Quantized observed next position vs diagnostic; damage/reset/duplicates excluded. Collision impulses without health changes, ladders, PM snapping and command synchronization can still differ. Errors diagnose applicability; no guard activation or safety acceptance.')
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps({k:report[k] for k in ('all','commanded_motion','contacts')}))


if __name__=='__main__':main()

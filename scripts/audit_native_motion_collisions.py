"""Join native pmove commands and collision candidates to recorded predictions."""
import argparse
import collections
import json
import math
import pathlib
import re
from audit_registered_execution_actions import read,sha
from audit_native_ground_prediction import measure

PREFIX=re.compile(r'^sv_test_motion spawncount=(\d+) server_frame=(\d+) g_test_motion version=[12] frame=(\d+) actor=(\d+) call=(\d+) kind=(begin|trace|end) (.*)$')
KEYS=dict(begin=set('msec forward side up pitch yaw pm_type pm_time pm_flags snapinitial origin velocity'.split()),
    trace=set('index fraction startsolid allsolid blocker class model start end normal'.split()),
    end=set('traces printed touches waterlevel pm_type pm_time pm_flags origin velocity'.split()))
VECTORS={'origin','velocity','start','end','normal','blocker_origin','blocker_mins','blocker_maxs'}


def native_calls(lines):
    calls={}
    for line in lines:
        if 'g_test_motion version=' not in line:continue
        match=PREFIX.fullmatch(line.rstrip('\r\n'));assert match,'Malformed/unwrapped motion event'
        spawn,frame,game,actor,number=map(int,match.groups()[:5]);kind=match[6]
        tokens=match[7].split();fields=dict(t.split('=',1) for t in tokens)
        expected=KEYS[kind]
        if kind=='trace' and 'g_test_motion version=2 ' in line:expected=expected|{'blocker_origin','blocker_mins','blocker_maxs'}
        assert len(tokens)==len(fields) and set(fields)==expected
        for key,value in list(fields.items()):
            if key in ('class','model'):continue
            if key in VECTORS:
                fields[key]=[float(v) for v in value.split(',')]
                assert len(fields[key])==3 and all(math.isfinite(v) for v in fields[key])
            elif key=='fraction':
                fields[key]=float(value);assert math.isfinite(fields[key]) and 0<=fields[key]<=1
            else:fields[key]=int(value)
        identity=(spawn,actor,number)
        if kind=='begin':
            assert identity not in calls and 0<=fields['msec']<=255
            calls[identity]=dict(spawncount=spawn,server_frame=frame,game_frame=game,actor=actor,call=number,begin=fields,traces=[])
        else:
            assert identity in calls and 'end' not in calls[identity]
            item=calls[identity];assert item['server_frame']==frame and item['game_frame']==game
            if kind=='trace':
                assert 1<=fields['index'] and (not item['traces'] or fields['index']>item['traces'][-1]['index'])
                item['traces'].append(fields)
            else:
                assert fields['printed']==len(item['traces'])<=32
                assert fields['traces']>=max([t['index'] for t in item['traces']] or [0])
                item['end']=fields
    assert calls and all('end' in c for c in calls.values()),'Incomplete or absent native motion telemetry'
    return list(calls.values())


def match_motion(row,calls):
    cmd=row['sent_command'];expected={k:cmd[v] for k,v in [('msec','Msec'),('forward','Forward'),('side','Side'),('up','Up'),('pitch','Pitch'),('yaw','Yaw')]}
    matches=[c for c in calls if c['spawncount']==row['spawncount'] and c['actor']==1 and
        c['server_frame'] in (row['observation_frame'],row['observation_frame']+1) and
        all(c['begin'][k]==v for k,v in expected.items()) and
        max(abs(v/8-x) for v,x in zip(c['begin']['origin'],row['self']))<=.125]
    return matches[0] if len(matches)==1 else None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pool',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True)
    args=ap.parse_args();assert not args.out.exists();pool_path=args.pool/'report.json';pd=sha(pool_path);pool=read(pool_path)
    assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==len(pool['jobs'])
    cases=[];totals=collections.Counter();classes=collections.Counter()
    release_pattern=re.compile(r'^sv_test_combat spawncount=(\d+) server_frame=(\d+) g_test_combat_start game_frame=(\d+) ready=1 seed=(\d+)$')
    for job in pool['jobs']:
        assert not job['error'];capture_path=pathlib.Path(job['root'])/'report.json';cap=read(capture_path)
        assert cap['capture_complete'] and cap['provenance_valid'];directory=pathlib.Path(cap['results'][0]['root'])
        trace,log=directory/'bot.jsonl',directory/'server.log';td,ld=sha(trace),sha(log)
        lines=log.read_text(encoding='utf-8-sig').splitlines();calls=native_calls(lines)
        releases=[tuple(map(int,m.groups())) for line in lines if (m:=release_pattern.fullmatch(line))]
        assert len(releases)==1 and releases[0][3]==job['seed']
        rows=[json.loads(s) for s in trace.open(encoding='utf-8-sig')];data=measure(rows,releases[0][1],releases[0][0])
        by_frame=collections.defaultdict(list)
        for row in rows:by_frame[row['observation_frame']].append(row)
        details=[];counts=collections.Counter(native_calls=len(calls),measured_pairs=len(data['samples']))
        for sample in data['samples']:
            frame=sample['frame'];entries=by_frame[frame]
            if len(entries)!=1:counts['unmatched']+=1;continue
            row=entries[0];native=match_motion(row,calls)
            if not native:counts['unmatched']+=1;continue
            counts['matched']+=1
            if sample['position_error']<=1:continue
            counts['large_error_matched']+=1
            candidates=[t for t in native['traces'] if math.hypot(*t['normal'][:2])>.1 or t['startsolid'] or t['allsolid']]
            labels=sorted({t['class'] for t in candidates});classes.update(labels)
            details.append(dict(frame=frame,error=sample['position_error'],before=row['self'],predicted=row['pmove_ground_prediction']['position'],
                observed=sample['observed'],native_origin=[v/8 for v in native['end']['origin']],native=native,
                collision_candidate_classes=labels,observed_mover_models=[m['model'] for m in row.get('movers',[])]))
        assert sha(trace)==td and sha(log)==ld
        totals.update(counts);cases.append(dict(job=job['job'],seed=job['seed'],map=rows[-1]['map'],counts=dict(counts),
            capture_sha256=sha(capture_path),trace_sha256=td,server_log_sha256=ld,large_errors=details))
    assert sha(pool_path)==pd
    args.out.write_text(json.dumps(dict(state='complete',version='native_collision_diagnostic_v1',pool_sha256=pd,cases=cases,
        totals=dict(totals),collision_candidate_classes=dict(classes),script_sha256=sha(__file__),training_performed=False,neural_execution=False,
        scope='Native PM_trace candidates include ground probes and rejected step paths, not proof of the selected blocking plane. Exact command/source pose and network-frame join; ambiguous pairs excluded. Seeds intentionally replayed for diagnostics, not independent quality evidence.'),indent=2),encoding='utf-8')
    print(json.dumps(dict(totals=totals,collision_candidate_classes=classes)))


if __name__=='__main__':main()

"""Finite known-state movement intervention, native effects and CUDA replay."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from audit_combat_checked_ground_native_probe import normalized, same
from process_combat_architecture_pool import read, save, sha
from report_combat_machinegun_hits import scan
from report_registered_machinegun_outcomes import summary


def run(protocol, root, out):
    assert not out.exists()
    spec, pool = read(protocol), read(root/'report.json')
    assert spec['version'] == 'combat_corpse_motion_native_probe_v1'
    assert pool['state'] == 'complete' and pool['source_unchanged'] and len(pool['jobs']) == 8
    assert pool['protocol_sha256'] == sha(protocol)
    pins = {**spec['source_files'], str(protocol):sha(protocol), str(Path(__file__).resolve()):sha(Path(__file__))}
    for v in spec['variants'].values():
        assert sha(Path(v['path'])) == v['sha256']
        pins[v['path']] = v['sha256']
    out.mkdir(parents=True)
    shutil.copyfile(__file__,out/Path(__file__).name)
    env = dict(os.environ, GOCACHE=str(Path.cwd()/'workspace/build/gocache'))

    def command(label, cmd):
        with (out/(label+'.log')).open('w',encoding='utf-8') as log:
            p = subprocess.run(list(map(str,cmd)),env=env,stdout=log,stderr=log)
        assert p.returncode == 0, label

    def trace(path):
        pins[str(path)] = sha(path)
        result = {}
        for line in path.open(encoding='utf-8-sig'):
            r = json.loads(line)
            o = r.get('combat_policy',{}).get('observation',{})
            if o.get('identity',{}).get('life') == 1 and o.get('health',0)>0:
                frame=o['identity']['frame']
                assert frame not in result
                result[frame]=r
        return result

    def signature(r):
        c=r['combat_policy']
        return normalized({k:c[k] for k in ('observation','proposed','proposed_command','applied_command')})

    reference = trace(Path(spec['cases'][0]['original_trace']))
    assert sha(Path(spec['cases'][0]['original_trace'])) == spec['cases'][0]['original_trace_sha256']
    exporter=(out/'q2ppo-data.exe').resolve()
    command('build',['go','build','-o',exporter,'./cmd/q2ppo-data'])
    traces, tasks, records = {}, [], []
    for job in sorted(pool['jobs'],key=lambda j:j['job']):
        assert not job['error']
        folder=Path(job['root']);cap=read(folder/'report.json');manifest=read(folder/'manifest.json')
        assert cap['capture_complete'] and cap['provenance_valid'] and manifest['diagnostic_forced_duck_frame'] == -1
        row=cap['results'][0]
        assert row['dispatch_valid'] and row['capture_valid'] and row['seed_confirmed']
        directory=Path(row['root']);values=trace(directory/'bot.jsonl')
        label,repeat=job['label'],job['repeat'];traces[label,repeat]=values
        prefix=[f for f in reference if 98<=f<374]
        assert all(f in values and same(signature(reference[f]),signature(values[f]),1e-9) for f in prefix), 'Prefix drift'
        if label=='control':
            assert all(f in values and same(signature(reference[f]),signature(values[f]),1e-9) for f in reference if f>=98)
        paths=[directory/'dataset/steps.jsonl',directory/'dataset/server_outcomes.jsonl',directory/'server.log']
        for path in paths:pins[str(path)]=sha(path)
        steps=[json.loads(line) for line in paths[0].open(encoding='utf-8-sig')]
        effects={r['step']:r for r in map(json.loads,paths[1].open(encoding='utf-8-sig'))}
        windows={}
        for s in steps:
            o,n,x=s['observation'],s.get('native_step'),s.get('server_execution') or {}
            if not n or s['owner']!='provider' or o['identity']['life']!=1 or o['health']<=0 or o['identity']['frame']<374:
                continue
            if not x.get('matched') or not x.get('window_exclusive') or x.get('recovery_commands')!=0:continue
            key=tuple(n[k] for k in ('spawncount','begin_frame','actor','sequence'))
            assert key not in windows;windows[key]=s
        shots=[q for q in scan(paths[2].open(encoding='utf-8-sig'),set(windows)) if q['eligible']]
        native_effects=[effects[s['index']] for s in windows.values()]
        assert all(e['available'] for e in native_effects)
        record=dict(label=label,repeat=repeat,shots=summary(shots),
            live_monster_hp=sum(e['monster_health_damage'] for e in native_effects),
            kills=sum(e['monster_kills'] for e in native_effects),
            received_hp=sum(e['received_health_damage'] for e in native_effects),
            death=bool(row['death_stop'] or row['first_life']['minimum_health']<=0),goal=bool(row.get('goal_stop')),
            end_health=steps[-1]['observation']['health'],
            changed_movement_commands=sum(values[f]['combat_policy']['applied_command'][k]!=reference[f]['combat_policy']['applied_command'][k]
                for f in range(374,379) if f in values and f in reference for k in ('Forward','Side')),
            ending_position=steps[-1]['observation']['position'])
        records.append(record)
        model=Path(spec['variants'][label]['path'])
        target=(out/f"case-{job['job']}").resolve();target.mkdir()
        command(f"export-{job['job']}",[exporter,'--batch',folder,'--model',model,'--cuda-only','--out',target/'native'])
        tasks.append(dict(model=str(model),model_sha256=sha(model),data=str(target/'native'),out=str(target/'cuda'),native_report_sha256=sha(target/'native/report.json')))
    for label in spec['variants']:
        a,b=traces[label,0],traces[label,1]
        assert a.keys()==b.keys() and all(same(signature(a[f]),signature(b[f]),1e-9) for f in a if f>=98), 'Cold repeat drift'
        first,second=[r for r in records if r['label']==label]
        assert {k:v for k,v in first.items() if k!='repeat'}=={k:v for k,v in second.items() if k!='repeat'}
    save(out/'native-results.json',dict(records=records))
    save(out/'cuda-request.json',dict(version='combat_cuda_finalize_batch_v1',compact_finalized_streams=True,tasks=tasks))
    command('cuda',[sys.executable,'scripts/finalize_combat_cuda_batch.py','--request',out/'cuda-request.json','--receipt-root',out/'cuda-batch'])
    for task in tasks:
        meta,proof=read(Path(task['out'])/'report.json'),read(Path(task['out'])/'cuda-verification.json')
        assert proof['state']=='passed' and proof['device']=='cuda'
        assert meta['diagnostic_only'] and not meta['ppo_training_ready'] and not proof['ordinary_training_eligible']
        assert meta['diagnostic_forced_duck_frame']==-1
    assert all(sha(Path(p))==digest for p,digest in pins.items())
    save(out/'report.json',dict(state='complete',records=records,native_cases=8,cuda_cases=8,inputs=pins,
        independent_worlds=1,training_performed=False,promotion=False,scope=spec['scope']))
    save(out/'complete.json',dict(state='complete',report_sha256=sha(out/'report.json')))
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('protocol','root','out'):p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args()
    run(args.protocol.resolve(),args.root.resolve(),args.out.resolve())

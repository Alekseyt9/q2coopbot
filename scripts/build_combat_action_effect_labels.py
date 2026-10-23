"""Extract motor-effect labels from sealed TRAIN shot recipes; no NN replay."""
import argparse
import collections
import gzip
import json
import subprocess
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from process_combat_architecture_pool import compact_closed_exports
from report_combat_machinegun_hits import scan,vector
from report_registered_machinegun_outcomes import outcome


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('legacy-registry','legacy-dataset','exporter','out'):
        ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--captures',type=Path,nargs='+',required=True)
    args=ap.parse_args();assert not args.out.exists();args.out.mkdir(parents=True)
    pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path);assert expected is None or digest==expected,str(path)
        pins[str(path)]=digest;return path
    registry=read(pin(args.legacy_registry))
    assert registry['state']=='sources_verified'
    folders=[args.legacy_dataset]+[r/'dataset' for r in args.captures]
    requests=[];records=[];source_cases=0;seen=set()
    save(args.out/'protocol.json',dict(version='combat_action_effect_capture_labels_v1',
        split='train',hand=2,purpose='motor_effect_prediction_train_only',
        datasets=[str(p.resolve()) for p in folders],policy_training=False,neural_execution=False,
        scope='Pre-command observations/actions only in requests. Native shot origin/view/recoil/velocity only in labels. Existing source datasets select exact shot membership; no recapture or PPO replay.'))
    for dataset_index,folder in enumerate(folders):
        folder=folder.resolve();closed=read(pin(folder/'complete.json'))
        report=read(pin(folder/'report.json',closed['report_sha256']))
        assert report['state']=='complete' and not closed['ppo_ready']
        assert report['device']=='cuda';source_cases+=report['source_cases']
        with gzip.open(pin(folder/'members.jsonl.gz',report['members_sha256']),'rt',encoding='utf-8') as f:
            members=[json.loads(line) for line in f]
        assert len(members)==report['shots']
        origin_by_source={}
        if dataset_index==0:
            for i,entry in enumerate(registry['entries']):
                origin=Path(entry['corpus'])
                cp=read(pin(origin/'complete.json',entry['complete_sha256']))
                assert cp['state']=='complete'
                recipe=read(pin(origin/'protocol.json',cp['protocol_sha256']));assert recipe['train_only']
                pool=read(pin(origin/'pool/report.json',entry['pool_sha256']))
                origin_by_source[i]=(origin,pool)
        else:
            origin=args.captures[dataset_index-1].resolve()
            cp=read(pin(origin/'native-complete.json'))
            recipe=read(pin(origin/'recipe.json',cp['recipe_sha256']))
            assert recipe['split']=='train' and recipe['purpose']=='shot_predictive_train_only'
            pool=read(pin(origin/'pool/report.json',cp['pool_sha256']))
            origin_by_source[0]=(origin,pool)
        group=collections.defaultdict(list)
        for m in members:group[(m['source'],m['job'])].append(m)
        joined={}
        for (source,job_id),ms in sorted(group.items()):
            origin,pool=origin_by_source[source]
            assert pool['state']=='complete' and pool['source_unchanged']
            jobs={j['job']:j for j in pool['jobs']};job=jobs[job_id];assert not job['error']
            capture_path=Path(job['root'])/'report.json'
            capture=read(pin(capture_path))
            assert capture['capture_complete'] and capture['provenance_valid']
            raw=Path(capture['results'][0]['root'])
            steps=origin/f'processing-train/case-{job_id:04d}/native/replay-0/steps.jsonl' if dataset_index==0 else raw/'dataset/steps.jsonl'
            steps=pin(steps,report['inputs'][str(steps.resolve())])
            log=pin(raw/'server.log',report['inputs'][str((raw/'server.log').resolve())])
            wanted={m['index'] for m in ms}
            with steps.open(encoding='utf-8-sig') as f:
                lookup={s['index']:s for s in map(json.loads,f) if s['index'] in wanted}
            assert len(lookup)==len(wanted)
            windows={}
            for s in lookup.values():
                n=s['native_step'];dispatch=s['server_execution']
                assert s['owner']=='provider' and dispatch['matched'] and dispatch['window_exclusive'] and not dispatch['recovery_commands']
                key=tuple(n[k] for k in ('spawncount','begin_frame','actor','sequence'))
                assert key not in windows;windows[key]=s
            with log.open(encoding='utf-8-sig') as f:
                shots={int(s['fields']['shot']):s for s in scan(f,set(windows)) if s['eligible']}
            for m in ms:
                s=lookup[m['index']];o=s['observation'];shot=shots[m['shot']];n=s['native_step']
                assert o['identity']['life']==1 and o['health']>0 and o['identity']['frame']==m['frame']
                assert shot['window']==tuple(n[k] for k in ('spawncount','begin_frame','actor','sequence'))
                assert outcome(shot)==m['outcome']
                key=(dataset_index,source,m['seed'],m['shot']);assert key not in seen;seen.add(key)
                sample={k:s['sample'][k] for k in ('latent','vertical','weapon','target','aim_mode') if k in s['sample']}
                request=dict(observation=o,sample=sample)
                fields=shot['fields']
                record=dict(m,source_dataset=dataset_index,capture=str(origin),map=o['identity']['map'],
                    native_muzzle_label=vector(fields['start']),native_view_label=vector(fields['view']),
                    native_recoil_label=vector(fields['recoil']),native_aim_label=vector(fields['aim']),
                    native_velocity_label=vector(fields['velocity']),native_game_frame_label=int(fields['frame']),
                    native_server_frame_label=shot['server_frame'])
                joined[(source,m['job'],m['shot'])]=(request,record)
        for m in members:
            request,record=joined[(m['source'],m['job'],m['shot'])]
            requests.append(request);records.append(record)
        save(args.out/'progress.json',dict(datasets_done=dataset_index+1,shots=len(records)))
        print('Dataset',dataset_index,'joined; shots',len(records),flush=True)
    ip,op=args.out/'requests.jsonl',args.out/'pure-go-causal.jsonl'
    with ip.open('w',encoding='utf-8') as f:
        for q in requests:f.write(json.dumps(q,separators=(',',':'))+'\n')
    with gzip.open(args.out/'members.jsonl.gz','wt',encoding='utf-8') as f:
        for q in records:f.write(json.dumps(q,separators=(',',':'))+'\n')
    pin(args.exporter)
    subprocess.run([str(args.exporter.resolve()),'--input',str(ip.resolve()),'--out',str(op.resolve()),
        '--geometry-version','combat_observed_action_geometry_v3'],check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    compact_closed_exports(args.out)
    for path,digest in pins.items():assert sha(Path(path))==digest,path
    save(args.out/'report.json',dict(state='complete',version='combat_action_effect_capture_labels_v1',
        hand=2,shots=len(records),source_cases=source_cases,
        source_battles_with_shots=len({(r['source_dataset'],r['source'],r['job']) for r in records}),
        episode_seed_groups=len({(r['episode'],r['seed']) for r in records}),
        inputs=pins,members_sha256=sha(args.out/'members.jsonl.gz'),request_sha256=sha(ip),
        pure_go_causal_sha256=sha(op),script_sha256=sha(__file__),datasets=[str(p.resolve()) for p in folders],
        policy_training=False,neural_execution=False,promotion=False,
        scope='Native outcomes are labels only, source split is TRAIN. Pure Go export never evaluates a model; all causal185 rows must be rechecked on CUDA against original compact datasets before fitting.'))
    save(args.out/'complete.json',dict(state='complete',report_sha256=sha(args.out/'report.json'),ppo_ready=False))
    print(json.dumps(dict(shots=len(records),source_cases=source_cases)))


if __name__=='__main__':main()

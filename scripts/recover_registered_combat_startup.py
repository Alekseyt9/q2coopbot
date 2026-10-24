"""Derive a repaired pool from terminal empty startup failures, preserving inputs."""
import argparse,copy,datetime,pathlib,shutil,subprocess,sys,time
from process_combat_architecture_pool import read,save,sha,member


def classify(job):
    folder=pathlib.Path(job['root']);capture=read(folder/'report.json')
    assert capture['provenance_valid'] and not capture['capture_complete'] and len(capture['results'])==1
    result=capture['results'][0];worker=pathlib.Path(result['root'])
    assert worker.resolve().is_relative_to(folder.resolve())
    report=read(worker/'report.json')
    reason=report['reason']
    paging=reason=='This command cannot be run due to the error: The paging file is too small for this operation to complete.'
    bot_error=(worker/'bot.err').read_text(errors='replace') if (worker/'bot.err').exists() else ''
    bot_oom=(reason=='Bot failed' and (bot_error.startswith('fatal error: runtime: cannot allocate memory')
        or ('VirtualAlloc' in bot_error and 'errno=1455' in bot_error)))
    assert (reason=='Server startup failed' or paging or bot_oom) and not report['accepted']
    assert not any(worker.glob('bot.jsonl*')),'Partial or actual gameplay cannot be retried'
    assert 'g_test_combat_start' not in (worker/'server.log').read_text(errors='replace')
    return read(folder/'manifest.json')


def main():
    ap=argparse.ArgumentParser()
    for name in ('root','spec','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    ap.add_argument('--port',type=int,default=41632)
    args=ap.parse_args();repo=pathlib.Path(__file__).resolve().parents[1]
    root,out=args.root.resolve(),args.out.resolve();base=repo/'workspace/artifacts'
    assert root.is_relative_to(base) and out.is_relative_to(base) and root!=out
    assert not out.exists() and read(root/'progress.json')['stage']=='failed'
    spec=read(args.spec);fixture_path=root/'fixture-proof.json';fixture=read(fixture_path)
    assert fixture['state']=='matched' and sha(args.spec)==fixture['spec_sha256']
    original_pool=root/'pool/report.json';pool=read(original_pool)
    assert pool['state']=='failed' and pool['source_unchanged']
    assert len(pool['jobs'])==fixture['native_cases'] and len(pool['plans'])==len(spec['variants'])
    assert len({j['job'] for j in pool['jobs']})==len(pool['jobs'])
    failures=[j for j in pool['jobs'] if j['error']];assert failures
    assert 1024<=args.port<=65535
    plans=[];pins={original_pool:sha(original_pool),args.spec:sha(args.spec),fixture_path:sha(fixture_path)}
    for binding in pool['plans']:
        path=pathlib.Path(binding['path']);assert sha(path)==binding['sha256'];pins[path]=binding['sha256']
        plans.append(read(path))
    for binding in fixture['models']:
        update=pathlib.Path(binding['update']);seal_path=update/'complete.json';seal=read(seal_path)
        assert sha(seal_path)==binding['update_seal_sha256'];pins[seal_path]=binding['update_seal_sha256']
        for filename,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
            path=update/filename;assert sha(path)==seal[key];pins[path]=seal[key]
        path=pathlib.Path(binding['model']);assert sha(path)==binding['model_sha256'];pins[path]=binding['model_sha256']
    for binding in fixture['readiness']:
        path=pathlib.Path(binding['path']);assert sha(path)==binding['sha256'];pins[path]=binding['sha256']
    manifests={}
    for j in failures:
        manifests[j['job']]=classify(j)
        folder=pathlib.Path(j['root']);result=read(folder/'report.json')['results'][0];worker=pathlib.Path(result['root'])
        for path in [folder/'report.json',folder/'manifest.json',worker/'report.json',worker/'server.log',root/f"pool/jobs/job-{j['job']}-result.json"]:
            pins[path]=sha(path)
    out.mkdir();(out/'pool').mkdir();receipts=[];repaired=copy.deepcopy(pool)
    save(out/'progress.json',dict(stage='recovering-empty-startup',training_performed=False))
    def run(command,log):
        with log.open('w',encoding='utf-8') as stream:
            p=subprocess.run([str(v) for v in command],cwd=repo,stdout=stream,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert p.returncode==0,str(log)
    try:
        for j in repaired['jobs']:
            if not j['error']:continue
            old=copy.deepcopy(j);manifest=manifests[j['job']];plan=plans[j['plan_index']]
            task=plan['tasks'][j['task_index']];seed_index=task['seeds'].index(j['seed'])
            folder=out/f"retry-job-{j['job']}"
            started=datetime.datetime.now(datetime.timezone.utc).isoformat()
            timer=time.perf_counter()
            run([shutil.which('pwsh'),'-NoProfile','-File',repo/'scripts/run_registered_combat_pool_episode.ps1',
                '-Plan',pool['plans'][j['plan_index']]['path'],'-TaskIndex',j['task_index'],'-SeedIndex',seed_index,
                '-Mode',j['mode'],'-Port',args.port,'-OutputRoot',folder,'-BinaryBundle',manifest['binary_bundle']],
                out/f"retry-job-{j['job']}.log")
            _,accepted,new_manifest=member(folder,j['seed'])
            for field in ('source_fingerprint','native_source_fingerprint','model_weights_sha256','reward_config_sha256','client_sha256','exporter_sha256','sources','native_sources'):
                assert new_manifest[field]==manifest[field],field
            assert accepted['results'][0]['generated_fixture']==task['instances'][seed_index]
            assert accepted['results'][0]['generated_start']['confirmed']
            j.update(root=str(folder),error='',port=args.port,slot=None,wall_seconds=time.perf_counter()-timer,
                started_utc=started,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                recovery_reason='empty_precombat_server_startup',original_failed_attempt=old)
            receipts.append(dict(original=old,replacement=copy.deepcopy(j),capture_sha256=sha(folder/'report.json')))
        seen=set();source_cache={};native_frames=0
        for j in repaired['jobs']:
            key=(j['plan_index'],j['task_index'],j['seed'],j['mode']);assert key not in seen;seen.add(key)
            plan=plans[j['plan_index']];task=plan['tasks'][j['task_index']];si=task['seeds'].index(j['seed'])
            assert j['mode'] in task['modes'] and not j['error']
            folder,capture,manifest=member(j['root'],j['seed']);r=capture['results'][0]
            assert r['frame_budget_valid'] and r['generated_fixture']==task['instances'][si] and r['generated_start']['confirmed']
            expected=plan.get('model_sha256') if j['mode']=='learned' else ''
            assert (manifest.get('model_weights_sha256') or '').lower()==(expected or '')
            assert manifest['reward_config_sha256'].lower()==task['reward_sha256']
            for kind,source_root in [('sources',repo),('native_sources',repo.parent/'yquake2')]:
                for record in manifest[kind]:
                    path=source_root/record['path'];digest=record['sha256'].lower()
                    if path not in source_cache:source_cache[path]=sha(path)
                    assert source_cache[path]==digest,str(path)
            native_frames+=r['actual_game_frames']
        expected_keys={(pi,ti,seed,mode) for pi,plan in enumerate(plans) for ti,task in enumerate(plan['tasks']) for seed in task['seeds'] for mode in task['modes']}
        assert seen==expected_keys
        for path,digest in pins.items():assert sha(path)==digest,str(path)
        for label,binding in zip((v['label'] for v in spec['variants']),pool['plans']):
            shutil.copy2(binding['path'],out/(label+'-plan.json'))
        shutil.copy2(fixture_path,out/'fixture-proof.json')
        repaired.update(state='complete',usable_captures=len(repaired['jobs']),actual_game_frames=native_frames,
            recovery=dict(original_pool=str(original_pool),original_pool_sha256=pins[original_pool],
                reused_valid_cases=len(repaired['jobs'])-len(receipts),retried_empty_startups=len(receipts),
                scope='Original pool and all attempts retained. Only empty precombat startup failures replaced by same frozen fixture/seed/model runs; no valid outcome replaced.'))
        save(out/'pool/report.json',repaired)
        save(out/'recovery-native-complete.json',dict(state='complete',native_cases=len(repaired['jobs']),
            pool_sha256=sha(out/'pool/report.json'),original_pool_sha256=pins[original_pool],attempts=receipts,
            inputs=[dict(path=str(p),sha256=d) for p,d in pins.items()],script_sha256=sha(pathlib.Path(__file__)),
            training_performed=False,promotion=False))
        save(out/'progress.json',dict(stage='native-recovered',native_cases=len(repaired['jobs']),training_performed=False))
        print('Native pool recovered:',len(repaired['jobs']),'cases; retries:',len(receipts),flush=True)
    except BaseException as error:
        save(out/'progress.json',dict(stage='failed',error=str(error),training_performed=False));raise


if __name__=='__main__':main()

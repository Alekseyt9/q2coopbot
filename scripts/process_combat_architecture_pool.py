"""Repair terminal captures, assemble own-policy corpora and run CUDA PPO.

Uses existing native harness/exporter. No game decisions or Go model replay tests.
Call only after the collecting process is authoritatively terminal.
"""
import argparse, concurrent.futures, copy, hashlib, json, pathlib, shutil, subprocess, sys, time


def read(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def save(path, value):
    path=pathlib.Path(path)
    pending=path.with_name(path.name+'.partial')
    pending.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    pending.replace(path)


def run(command, log):
    with pathlib.Path(log).open('w',encoding='utf-8') as output:
        result=subprocess.run([str(x) for x in command],stdout=output,stderr=subprocess.STDOUT,
                              creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode:
        raise RuntimeError('Command failed; inspect '+str(log))


def member(root, seed):
    root=pathlib.Path(root)
    report, manifest=read(root/'report.json'),read(root/'manifest.json')
    assert report['capture_complete'] and report['provenance_valid'] and len(report['results'])==1
    row=report['results'][0]
    assert row['seed']==seed and row['capture_valid'] and row['dispatch_valid'] and row['seed_confirmed']
    exporter=root/'q2combat-export.exe'
    assert sha(exporter).lower()==manifest['exporter_sha256'].lower()
    return root,report,manifest


def update_receipt(binding,plan,directory):
    seal=read(directory/'complete.json');update=read(directory/'report.json')
    for filename,field in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
        assert sha(directory/filename)==seal[field],'Sealed CUDA update changed'
    assert update['device']=='cuda' and update['behavior_sha256']==sha(binding['model'])
    assert update['updates_completed']==binding.get('parent_updates_completed',0)+1
    assert update['resume_sha256']==binding.get('resume_checkpoint_sha256')
    return dict(model=binding['id'],architecture=binding['architecture'],initialization_seed=binding.get('initialization_seed'),
        allocated_episodes=sum(len(t['seeds']) for t in plan['tasks']),eligible_transitions=update['rows'],actor_steps=update['actor_steps'],
        device=update['device'],weights=str(directory/'weights.json'),weights_sha256=update['weights_sha256'],
        checkpoint=str(directory/'checkpoint.pt'),updates_completed=update['updates_completed'],total_actor_steps=update['total_actor_steps'],
        resume_checkpoint_sha256=update['resume_sha256'])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--capture-root',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    ap.add_argument('--config',type=pathlib.Path,required=True)
    ap.add_argument('--exporter',type=pathlib.Path,required=True)
    ap.add_argument('--anchor',type=pathlib.Path,required=True)
    ap.add_argument('--bank',type=pathlib.Path,required=True)
    ap.add_argument('--retry-port',type=int,default=34400)
    ap.add_argument('--export-workers',type=int,default=4)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--cuda-only-export',action='store_true',help='Native/features export, then CUDA numerical verification and exact next-state bootstrap')
    args=ap.parse_args()
    repo=pathlib.Path(__file__).resolve().parents[1]
    capture=args.capture_root.resolve(); out=args.out.resolve()
    assert out.is_dir() if args.resume else not out.exists(),'Fresh output required, or explicit --resume'
    assert 1<=args.export_workers<=16
    pool=read(capture/'pool/report.json')
    assert pool['state'] in ('complete','failed') and pool['source_unchanged']
    assert pool['scheduler']=='independent_episode_queue'
    models=read(capture/'models.json')
    jobs=pool['jobs']
    expected={}
    plans={}
    for p,binding in enumerate(models):
        if binding.get('resume_checkpoint'):
            assert binding['architecture']['architecture']!='mlp','Continuation adapter currently supports recurrent/attention checkpoints'
            assert sha(binding['resume_checkpoint'])==binding['resume_checkpoint_sha256']
            parent=read(binding['resume_report'])
            assert sha(binding['resume_report'])==binding['resume_report_sha256']
            assert parent['weights_sha256']==sha(binding['model']) and parent['device']=='cuda'
            assert parent['updates_completed']==binding['parent_updates_completed']>=1
            for source,field in [(args.config,'config_sha256'),(args.anchor,'anchor_sha256'),(args.bank,'bank_sha256')]:
                assert sha(source)==parent[field],'Continuation training contract differs'
        path=pathlib.Path(binding['plan']); plan=read(path); plans[p]=plan
        receipt=next(x for x in pool['plans'] if pathlib.Path(x['path']).resolve()==path.resolve())
        assert sha(path)==receipt['sha256'] and sha(binding['model'])==receipt['model_sha256']==plan['model_sha256']
        for t,task in enumerate(plan['tasks']):
            for mode in task['modes']:
                assert mode=='learned' and task['split']=='train' and task['episode']['recipe']['runner']=='combat-baseline'
                for i,seed in enumerate(task['seeds']):expected[(p,t,seed,mode)]=i
    by_key={(j['plan_index'],j['task_index'],j['seed'],j['mode']):j for j in jobs}
    assert len(by_key)==len(jobs)==len(expected) and by_key.keys()==expected.keys(),'Incomplete allocation'
    snapshots=out/'python-sources'
    if not args.resume:
        out.mkdir();snapshots.mkdir()
        for path in repo.joinpath('scripts').glob('*.py'):shutil.copy2(path,snapshots/path.name)
        shutil.copy2(args.config,out/'config.json');shutil.copy2(args.anchor,out/'anchor.json');shutil.copy2(args.bank,out/'bank.json')
        shutil.copy2(args.exporter,out/'q2ppo-data.exe')
    protocol=dict(version='combat_architecture_pool_processing_v1',capture_root=str(capture),
        pool_sha256=sha(capture/'pool/report.json'),models_sha256=sha(capture/'models.json'),
        config_sha256=sha(out/'config.json'),exporter_sha256=sha(out/'q2ppo-data.exe'),
        python_sources={p.name:sha(p) for p in snapshots.glob('*.py')},
        allocated_episodes=len(expected),final_test_deferred=True,
        scope='Own-policy native corpus; failed attempts excluded and retried; equal allocated episodes, actual eligible transitions reported separately.')
    if args.cuda_only_export:
        protocol['numerical_verification']='cuda_verified_v1'
    if args.resume:
        assert protocol==read(out/'protocol.json'),'Frozen processing inputs changed'
        for source,target in [(args.config,'config.json'),(args.exporter,'q2ppo-data.exe'),(args.anchor,'anchor.json'),(args.bank,'bank.json')]:
            assert sha(source)==sha(out/target),'Resume input differs'
    else:save(out/'protocol.json',protocol)
    retries=[];selected={};results=[]
    start=time.perf_counter()
    pwsh=repo/'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe'
    try:
        if args.resume:
            for entry in read(out/'selected-captures.json'):
                key=(entry['plan'],entry['task'],entry['seed'],entry['mode'])
                assert key in expected and key not in selected
                selected_root=pathlib.Path(entry['root'])
                assert sha(selected_root/'report.json')==entry['report_sha256'] and sha(selected_root/'manifest.json')==entry['manifest_sha256']
                selected[key]=member(selected_root,key[2])
            assert selected.keys()==expected.keys(),'Resume needs complete selected captures'
            retries=read(out/'retry-receipts.json') if (out/'retry-receipts.json').exists() else []
            results=read(out/'training-updates.json') if (out/'training-updates.json').exists() else []
            # A process can stop after the update seal is durable but before its
            # summary is saved. Restore only a verified contiguous prefix.
            for p in range(len(results),len(models)):
                directory=out/models[p]['id']/'update'
                if not (directory/'complete.json').exists():break
                receipt=update_receipt(models[p],plans[p],directory)
                assert read(directory/'report.json')['rollout_sha256']==read(out/models[p]['id']/'rollout/report.json')['rollout_sha256']
                results.append(receipt)
            save(out/'training-updates.json',results)
            assert len({x['model'] for x in results})==len(results)
            for p,entry in enumerate(results):
                assert entry['model']==models[p]['id'] and entry['device']=='cuda'
                update_root=out/entry['model']/'update';seal=read(update_root/'complete.json')
                for filename,field in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
                    assert sha(update_root/filename)==seal[field],'Completed update changed'
                update=read(update_root/'report.json')
                assert update['device']=='cuda' and update['updates_completed']==models[p].get('parent_updates_completed',0)+1 and entry['weights_sha256']==seal['weights_sha256']
                assert entry['eligible_transitions']==update['rows'] and entry['actor_steps']==update['actor_steps']
            receipt=out/('resume-'+str(time.time_ns())+'.json')
            save(receipt,dict(completed_models=[x['model'] for x in results],driver_sha256=sha(__file__),
                              scope='Frozen trainer snapshots retained; interrupted unsealed model assembly rebuilt.'))
        for key,job in by_key.items():
            if key in selected:continue
            p,t,seed,mode=key
            selected_root=pathlib.Path(job['root'])
            if job['error']:
                # Do not reuse a port until its endpoint is authoritatively free.
                command=f"if (@(Get-NetUDPEndpoint -LocalPort {args.retry_port} -ErrorAction SilentlyContinue).Count) {{ throw 'Retry port occupied' }}"
                run([pwsh,'-NoProfile','-Command',command],out/('port-'+str(job['job'])+'.log'))
                selected_root=None
                for attempt in range(1,4):
                    retry_root=out/('retry-'+str(job['job'])+'-'+str(attempt))
                    log=out/('retry-'+str(job['job'])+'-'+str(attempt)+'.log')
                    command=[pwsh,'-NoProfile','-File',repo/'scripts/run_registered_combat_pool_episode.ps1',
                        '-Plan',models[p]['plan'],'-TaskIndex',t,'-SeedIndex',expected[key],
                        '-Mode',mode,'-Port',args.retry_port,'-OutputRoot',retry_root]
                    try:
                        run(command,log);member(retry_root,seed)
                    except (RuntimeError,AssertionError,FileNotFoundError):
                        retries.append(dict(job=job['job'],attempt=attempt,root=str(retry_root),accepted=False))
                    else:
                        selected_root=retry_root
                        retries.append(dict(job=job['job'],attempt=attempt,root=str(retry_root),accepted=True))
                        break
                    finally:save(out/'retry-receipts.json',retries)
                if selected_root is None:raise RuntimeError('Native retry proof failed three times: '+str(job['job']))
            selected[key]=member(selected_root,seed)
        save(out/'selected-captures.json',[dict(plan=p,task=t,seed=seed,mode=mode,root=str(value[0]),
            report_sha256=sha(value[0]/'report.json'),manifest_sha256=sha(value[0]/'manifest.json'))
            for (p,t,seed,mode),value in selected.items()])
        for p,binding in enumerate(models):
            if p<len(results):continue
            model_root=out/binding['id']
            if model_root.exists():
                assert args.resume and model_root.resolve().parent==out
                preserved=out/('interrupted-'+binding['id']+'-'+str(time.time_ns()))
                assert preserved.resolve().parent==out and not preserved.exists()
                shutil.move(str(model_root),str(preserved))
            model_root.mkdir();exports=[]
            for t,task in enumerate(plans[p]['tasks']):
                case=model_root/('case-'+str(t));case.mkdir()
                members=[selected[(p,t,seed,'learned')] for seed in task['seeds']]
                manifest=copy.deepcopy(members[0][2]);rows=[];receipts=[]
                for member_root,report,mf in members:
                    for field in ('source_fingerprint','native_source_fingerprint','model_weights_sha256','reward_config_sha256','exporter_sha256'):
                        assert mf[field]==manifest[field],('Member lineage differs',field)
                    rows+=report['results']
                    receipts.append(dict(root=str(member_root),seed=report['results'][0]['seed'],
                        report_sha256=sha(member_root/'report.json'),manifest_sha256=sha(member_root/'manifest.json')))
                assert sorted(r['seed'] for r in rows)==sorted(task['seeds'])
                manifest.update(workers=1,episodes_per_worker=len(rows),seeds=[r['seed'] for r in rows],
                    seed_assignments=[{k:r[k] for k in ('worker','episode','seed','port','fixture_mixed','solo_fixture')} for r in rows],pool_members=receipts)
                if task.get('instances'):
                    save(case/'generated-fixtures.json',dict(instances=task['instances']))
                    manifest.update(generated_fixtures=str(case/'generated-fixtures.json'),generated_fixtures_sha256=sha(case/'generated-fixtures.json'))
                save(case/'manifest.json',manifest)
                save(case/'report.json',dict(version=1,provenance_valid=True,capture_complete=True,
                    timescale=2,parallelism=1,scheduler='verified_independent_member_assembly',
                    completed_episodes=len(rows),usable_captures=len(rows),results=rows,pool_members=receipts))
                shutil.copy2(members[0][0]/'q2combat-export.exe',case/'q2combat-export.exe')
                assert sha(case/'q2combat-export.exe').lower()==manifest['exporter_sha256'].lower()
                exports.append((case,model_root/('rollout-'+str(t))))
            save(out/'progress.json',dict(stage='native-export',model=binding['id'],cases=len(exports)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.export_workers) as workers:
                futures=[workers.submit(run,[out/'q2ppo-data.exe','--batch',case,'--model',binding['model'],'--out',
                         export.with_name(export.name+'-native') if args.cuda_only_export else export]
                         +(['--cuda-only'] if args.cuda_only_export else []),
                         model_root/('export-'+str(i)+'.log')) for i,(case,export) in enumerate(exports)]
                for future in futures:future.result()
            if args.cuda_only_export:
                for i,(_,export) in enumerate(exports):
                    run([sys.executable,snapshots/'finalize_combat_cuda_rollout.py','--model',binding['model'],
                         '--data',export.with_name(export.name+'-native'),'--out',export],model_root/('cuda-finalize-'+str(i)+'.log'))
            run([out/'q2ppo-data.exe','--merge',','.join(str(e) for _,e in exports),'--out',model_root/'rollout'],model_root/'merge.log')
            run([pwsh,'-NoProfile','-File',repo/'scripts/compress_completed_combat_streams.ps1',
                 '-Root',model_root],model_root/'compress.log')
            save(out/'progress.json',dict(stage='cuda-update',model=binding['id']))
            trainer='ppo_combat.py' if binding['architecture']['architecture']=='mlp' else 'ppo_recurrent.py'
            command=[sys.executable,snapshots/trainer,'--model',binding['model'],'--data',model_root/'rollout',
                '--config',out/'config.json','--out',model_root/'update','--retention-weight','0','--bank-weight','0']
            if trainer=='ppo_recurrent.py':command+=['--anchor-model',out/'anchor.json','--retention-bank',out/'bank.json']
            if binding.get('resume_checkpoint'):command+=['--resume',binding['resume_checkpoint']]
            run(command,model_root/'train.log')
            update=read(model_root/'update/report.json')
            assert update['device']=='cuda' and update['updates_completed']==binding.get('parent_updates_completed',0)+1 and sha(model_root/'update/weights.json')==update['weights_sha256']
            assert update['resume_sha256']==binding.get('resume_checkpoint_sha256')
            save(model_root/'update/complete.json',dict(version='combat_update_complete_v1',
                weights_sha256=sha(model_root/'update/weights.json'),checkpoint_sha256=sha(model_root/'update/checkpoint.pt'),
                report_sha256=sha(model_root/'update/report.json')))
            results.append(update_receipt(binding,plans[p],model_root/'update'))
            save(out/'training-updates.json',results)
            print(json.dumps(results[-1]),flush=True)
        assert sha(capture/'pool/report.json')==protocol['pool_sha256']
        save(out/'report.json',dict(state='complete',protocol=protocol,training=results,retries=retries,
             seconds=time.perf_counter()-start,promotion='Not assessed; paired validation and longer on-policy budget remain required'))
        save(out/'progress.json',dict(stage='complete'))
    except Exception as error:
        save(out/'report.json',dict(state='failed',error=str(error),training=results,retries=retries,
            scope='Incomplete pipeline, no promotion or silent partial training'))
        raise


if __name__=='__main__':
    main()

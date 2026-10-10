"""Inventory durable seed metadata without loading large traces or model weights."""
import argparse,json,pathlib,subprocess,hashlib,time,re
from process_combat_architecture_pool import read,save,sha

def collect(value,seeds):
    if isinstance(value,dict):
        for key,item in value.items():
            if key in ('seed','engine_seed') and isinstance(item,int):seeds.add(item)
            elif key=='seeds' and isinstance(item,list):seeds.update(v for v in item if isinstance(v,int))
            else:collect(item,seeds)
    elif isinstance(value,list):
        for item in value:collect(item,seeds)

def recover_interrupted_metadata(path,repo,recovered):
    """Bound a zero-filled offline report by intact capture receipts and rows.

    This is a seed-use audit only; it does not restore report quality or validate
    the old model. Never waive damaged native capture or arbitrary metadata.
    """
    mixed=repo/'workspace/artifacts/combat-mixed-registry-resume-v3-20261007/temporal28/epoch-1'
    if path.is_relative_to(mixed):return recover_mixed_metadata(path,mixed,recovered)
    relative=path.relative_to(repo/'workspace/artifacts/aproc-v1-20261007')
    parts=relative.parts
    assert len(parts) in (3,4) and re.fullmatch(r'interrupted-m\d+-\d+',parts[0])
    assert re.fullmatch(r'rollout-\d+',parts[1]) and parts[-1]=='report.json'
    if len(parts)==4:assert re.fullmatch(r'replay-\d+',parts[2])
    damaged=path.read_bytes();assert damaged and not any(damaged),'Only all-zero interrupted reports are recoverable'
    rollout=path.parents[1] if len(parts)==4 else path.parent
    if rollout not in recovered:
        case=rollout.parent/('case-'+parts[1].split('-')[1]);manifest=read(case/'manifest.json');report=read(case/'report.json')
        assert report['provenance_valid'] and report['capture_complete']
        expected=set(manifest['seeds']);assert expected=={row['seed'] for row in report['results']}
        receipts=manifest['pool_members'];assert receipts==report['pool_members']
        assert expected=={item['seed'] for item in receipts}
        for item in receipts:
            root=pathlib.Path(item['root']);assert root.resolve().is_relative_to((repo/'workspace/artifacts/apool-v1-20261007').resolve())
            assert sha(root/'report.json')==item['report_sha256'] and sha(root/'manifest.json')==item['manifest_sha256']
            upstream=read(root/'report.json');assert upstream['capture_complete'] and upstream['provenance_valid']
            assert {row['seed'] for row in upstream['results']}=={item['seed']}
        sequence=rollout/'sequence.jsonl';digest=hashlib.sha256();row_seeds=set();rows=0
        with sequence.open('rb') as stream:
            for line in stream:
                digest.update(line);obj=json.loads(line);assert isinstance(obj['seed'],int)
                row_seeds.add(obj['seed']);rows+=1
        assert rows and row_seeds==expected,'Sequence must cover exactly the verified upstream seed set'
        recovered[rollout]=dict(seeds=sorted(expected),case_manifest=str(case/'manifest.json'),case_manifest_sha256=sha(case/'manifest.json'),case_report_sha256=sha(case/'report.json'),sequence=str(sequence),sequence_sha256=digest.hexdigest(),sequence_rows=rows,pool_members=receipts,scope='Seed use bounded by intact assembled native capture receipts and every offline sequence row; damaged report remains unusable for model acceptance.')
    return dict(recovered[rollout],damaged_report_sha256=hashlib.sha256(damaged).hexdigest())

def recover_mixed_metadata(path,epoch,recovered):
    """The other known zero-filled offline archive, using registry receipts."""
    parts=path.relative_to(epoch).parts
    assert len(parts) in (2,3) and parts[0].startswith('rollout-') and parts[-1]=='report.json'
    if len(parts)==3:assert re.fullmatch(r'replay-\d+',parts[1])
    damaged=path.read_bytes();assert damaged and not any(damaged)
    rollout=epoch/parts[0]
    if rollout not in recovered:
        episode=parts[0].removeprefix('rollout-');plan=read(epoch/'plan.json')
        index,task=next((i,t) for i,t in enumerate(plan['tasks']) if t['episode']['id']==episode)
        binding=epoch/'capture'/('pool-job-'+str(index))/'report.json';job=read(binding)
        assert job['state']=='complete';records=[r for r in job['records'] if r['episode_id']==episode and r['mode']=='learned'];assert len(records)==1
        receipt=records[0];native=pathlib.Path(receipt['artifacts'])
        assert native.resolve().is_relative_to((epoch/'capture').resolve())
        assert receipt['capture_verified'] and receipt['split']=='train' and receipt['episode_sha256']==task['episode_sha256']
        assert sha(native/'report.json')==receipt['report_sha256']
        report=read(native/'report.json');manifest=read(native/'manifest.json');expected=set(task['seeds'])
        assert report['capture_complete'] and report['provenance_valid'] and expected==set(receipt['seeds'])==set(manifest['seeds'])=={r['seed'] for r in report['results']}
        assert manifest['model_weights_sha256'].lower()==receipt['model_sha256'].lower()
        assert all(all(r[k] for k in ('seed_confirmed','capture_valid','dispatch_valid','frame_budget_valid')) for r in report['results'])
        sequence=rollout/'sequence.jsonl';digest=hashlib.sha256();row_seeds=set();rows=0
        with sequence.open('rb') as stream:
            for line in stream:
                digest.update(line);obj=json.loads(line);assert isinstance(obj['seed'],int)
                row_seeds.add(obj['seed']);rows+=1
        assert rows and row_seeds==expected
        recovered[rollout]=dict(seeds=sorted(expected),plan=str(epoch/'plan.json'),plan_sha256=sha(epoch/'plan.json'),binding=str(binding),binding_sha256=sha(binding),native_report=str(native/'report.json'),native_report_sha256=sha(native/'report.json'),native_manifest_sha256=sha(native/'manifest.json'),sequence=str(sequence),sequence_sha256=digest.hexdigest(),sequence_rows=rows,scope='Seeds bounded by intact registry capture receipt and every offline sequence row; damaged report not restored or accepted for model quality.')
    return dict(recovered[rollout],damaged_report_sha256=hashlib.sha256(damaged).hexdigest())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evaluation',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    repo=pathlib.Path(__file__).resolve().parents[1];evaluation=a.evaluation.resolve()
    if a.resume:assert a.out.is_dir()
    else:assert not a.out.exists();a.out.mkdir(parents=True)
    protocol=read(evaluation/'protocol.json');reference=read(protocol['evaluations'][0]['plan']);families=reference['tasks'];assert len(families)==20
    tests={t['episode']['id']:t['episode']['splits']['test'] for t in families};test_universe={s for v in tests.values() for s in range(v['start'],v['start']+min(v['count'],64))}
    # The live cohort is excluded from filesystem scanning, but all its frozen
    # plans must be development validation and disjoint from the reserved seeds.
    for entry in protocol['evaluations']:
        assert sha(entry['plan'])==entry['plan_sha256'];plan=read(entry['plan'])
        assert all(t['split']=='validation' for t in plan['tasks'])
        assert not {s for t in plan['tasks'] for s in t['seeds']} & test_universe
    paths=subprocess.check_output(['rg','--files','--no-ignore',str(repo/'workspace/artifacts'),'-g','*-result.json','-g','report.json','-g','manifest.json'],text=True).splitlines()
    seen=set();evidence=[];unreadable=[];scanned=0;excluded=0;cached={};started=time.time();recovered={};recovery_evidence=[]
    cache=a.out/'metadata-cache.jsonl'
    if a.resume and cache.exists():
        for line in cache.read_text(encoding='utf-8').splitlines():
            try:entry=json.loads(line)
            except ValueError:continue # A interrupted last write is rescanned.
            cached[entry['path']]=entry
    with cache.open('a',encoding='utf-8') as stream:
        for index,filename in enumerate(paths):
            path=pathlib.Path(filename)
            if path.is_relative_to(evaluation) or path.is_relative_to(a.out.resolve()):excluded+=1;continue
            stamp=path.stat();entry=cached.get(filename)
            if entry is None or entry['size']!=stamp.st_size or entry['mtime_ns']!=stamp.st_mtime_ns or 'error' in entry or 'recovery' in entry:
                entry=dict(path=filename,size=stamp.st_size,mtime_ns=stamp.st_mtime_ns)
                try:
                    content=path.read_bytes();obj=json.loads(content)
                    local=set();collect(obj,local)
                    entry.update(seeds=sorted(local),sha256=hashlib.sha256(content).hexdigest())
                except (OSError,ValueError,UnicodeError) as error:
                    entry['error']=str(error)
                    try:
                        recovery=recover_interrupted_metadata(path,repo,recovered)
                    except (OSError,ValueError,KeyError,AssertionError) as recovery_error:
                        entry['recovery_error']=str(recovery_error)
                    else:
                        entry.update(seeds=recovery['seeds'],sha256=recovery['damaged_report_sha256'],recovery=recovery)
                        del entry['error']
                stream.write(json.dumps(entry)+'\n');stream.flush()
            if 'error' in entry:unreadable.append(entry)
            else:
                if 'recovery' in entry:recovery_evidence.append(dict(path=filename,**entry['recovery']))
                local=set(entry['seeds']);seen.update(local);scanned+=1
                hits=sorted(local&test_universe)
                if hits:evidence.append(dict(path=filename,sha256=entry['sha256'],seeds=hits))
            if index%1000==0:
                save(a.out/'progress.json',dict(stage='scanning',processed=index+1,total=len(paths),unreadable=len(unreadable),seconds=time.time()-started))
    save(a.out/'scan-report.json',dict(state='blocked_unreadable_metadata' if unreadable else 'scanned',metadata_files=scanned,unreadable=unreadable,recovery_evidence=recovery_evidence,recorded_seeds=sorted(seen),prior_test_seed_evidence=evidence))
    save(a.out/'progress.json',dict(stage='metadata_failed' if unreadable else 'reserving',metadata_files=scanned,unreadable=len(unreadable),recovered=len(recovery_evidence),seconds=time.time()-started))
    assert not unreadable,('Unverifiable historical metadata',unreadable[:5])
    selected=None
    for offset in range(57):
        wanted={v['start']+i for v in tests.values() for i in range(offset,offset+8)}
        if not wanted&seen:selected=offset;break
    assert selected is not None,'No common unseen eight-seed block in first64 test conditions'
    episode_hashes={t['episode']['id']:hashlib.sha256(json.dumps(t['episode'],sort_keys=True,separators=(',',':')).encode()).hexdigest() for t in families}
    conditions=[dict(episode=key,episode_sha256=episode_hashes[key],seeds=list(range(v['start']+selected,v['start']+selected+8))) for key,v in tests.items()]
    save(a.out/'seed-inventory.json',dict(state='audited',metadata_files=scanned,active_metadata_files_excluded=excluded,unique_recorded_seeds=len(seen),recorded_seed_set_sha256=hashlib.sha256(json.dumps(sorted(seen)).encode()).hexdigest(),prior_test_seed_evidence=evidence,selected_test_offset=selected,episodes_per_arm=160,conditions=conditions,evaluation_protocol_sha256=sha(evaluation/'protocol.json'),scope='All current artifact pool-result/report/manifest JSON seed metadata, excluding the active cohort after checking every frozen validation plan. Excludes trace JSONL and weights. Unseen means no recorded use in this workspace metadata snapshot; no claim about external/imported unreceipted data. Plans alone are not counted as executed use.'))
    inventory=read(a.out/'seed-inventory.json');inventory.update(scan_report_sha256=sha(a.out/'scan-report.json'),recovered_offline_report_count=len(recovery_evidence));inventory['scope']+=' Exception: every offline sequence row is inspected when bounding the two explicitly identified zero-filled archived exports by intact native receipts.';save(a.out/'seed-inventory.json',inventory)
    save(a.out/'selection-protocol.json',dict(version='combat_independent_test_selection_v1',state='prepared_not_dispatched',development_evaluation=str(evaluation),development_protocol_sha256=sha(evaluation/'protocol.json'),seed_inventory_sha256=sha(a.out/'seed-inventory.json'),split='test',seed_offset=selected,count_per_family=8,families=list(tests),episodes_per_arm=160,arms=['selected_candidate','parent3_before','firebc','rules'],total_episodes=640,slots=16,timescale=2,selection_order=['most development wins','fewest deaths','least mean received health damage','lexical candidate id'],eligibility='Candidate must exceed rules development wins, or tie wins with fewer deaths. Otherwise leave test reserved and improve using train/validation only.',scope='Choose exactly one candidate once from sealed full-registry development results. Freeze weights and all plans before test. Do not train or select another candidate on test results. Independent registered geometry distributions; no whole-map acceptance claim.'))
    save(a.out/'progress.json',dict(stage='prepared_not_dispatched',seed_offset=selected,episodes_per_arm=160,seed_inventory_sha256=sha(a.out/'seed-inventory.json')))
    print(json.dumps(dict(scanned=scanned,prior_test_metadata_files=len(evidence),reserved_offset=selected,episodes_per_arm=160)),flush=True)

if __name__=='__main__':main()

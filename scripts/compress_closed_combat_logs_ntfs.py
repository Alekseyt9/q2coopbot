"""Transparent NTFS compression of explicit closed roots; no trace conversion."""
import argparse,ctypes,json,os,pathlib,subprocess,hashlib,time

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=pathlib.Path,required=True);ap.add_argument('--include-completed-wide',action='store_true');ap.add_argument('--include-completed-ppo',action='store_true');ap.add_argument('--include-closed-ppo-evaluation-captures',action='store_true');a=ap.parse_args()
    repo=pathlib.Path(__file__).resolve().parents[1];base=repo/'workspace/artifacts';assert not a.out.exists();a.out.mkdir(parents=True)
    roots=['machinegun-shared-eval-v1-20261010','first-life-update4-eval-v3-20261010','miss-reward-ab-eval-20261010','machinegun-conditional-corpus-v1-20261010','first-life-update4-capture-20261010','cursor-death-stop-regression-20261010','movement-ppo-eval-v2-20261010','aeval-v1-20261008']
    active=base/'machinegun-full-eval-v1-20261010';files=[];samples=[]
    if a.include_completed_wide:
        progress=json.loads((active/'progress.json').read_bytes())
        assert progress['stage']=='complete' and progress['diagnostics_complete']
        terminal=json.loads((active/'pool/report.json').read_bytes())
        assert terminal['state']=='complete' and terminal['usable_captures']==960
        roots.append(active.name)
    if a.include_completed_ppo:
        capture=base/'machinegun-ppo-capture-v1-20261010';processing=base/'machinegun-ppo-processing-v1-20261010'
        execution=json.loads((capture/'execution.json').read_bytes());training=json.loads((processing/'report.json').read_bytes())
        assert execution['stage']=='complete' and execution['processing_report_sha256']==digest(processing/'report.json')
        assert training['state']=='complete' and len(training['training'])==2 and all(v['device']=='cuda' for v in training['training'])
        for item in training['training']:
            update=processing/item['model']/'update';audit=json.loads((update/'checkpoint-cuda-audit.json').read_bytes())
            assert audit['device']=='cuda' and audit['actor_value_std_exact'] and audit['optimizer_state_exact']
            assert audit['weights_sha256']==digest(update/'weights.json') and audit['checkpoint_sha256']==digest(update/'checkpoint.pt')
        roots.extend([capture.name,processing.name])
    closed_evaluation=base/'machinegun-ppo-eval-v1-20261010'
    if a.include_closed_ppo_evaluation_captures:
        terminal=json.loads((closed_evaluation/'pool/report.json').read_bytes())
        assert terminal['state']=='complete' and terminal['source_unchanged'] and terminal['usable_captures']==480
        assert len(terminal['jobs'])==480 and not any(j['error'] for j in terminal['jobs'])
        roots.append(closed_evaluation.name)
    for name in roots:
        root=base/name
        if not root.exists():continue
        assert root.resolve().is_relative_to(base.resolve()) and (root.resolve()!=active.resolve() or a.include_completed_wide)
        candidates=[]
        for path in root.rglob('*'):
            if root==closed_evaluation and 'capture' not in path.relative_to(root).parts:continue # Live diagnostic outputs excluded.
            if path.suffix not in ('.jsonl','.log') or not path.is_file():continue
            stat=path.stat()
            if stat.st_size<131072 or stat.st_file_attributes&0x800:continue
            assert path.resolve().is_relative_to(root.resolve())
            candidates.append(path)
        files.extend(candidates)
        for path in candidates[:1]+candidates[-1:]:samples.append(dict(path=str(path),sha256=digest(path),bytes=path.stat().st_size))
    total=sum(p.stat().st_size for p in files);save=lambda name,obj:(a.out/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
    save('inventory.json',dict(roots=roots,files=len(files),logical_bytes=total,samples=samples,completed_wide_verified=a.include_completed_wide,completed_ppo_verified=a.include_completed_ppo,closed_480_native_captures_verified=a.include_closed_ppo_evaluation_captures,excluded_active_roots=[str(base/'action-sampling-site02-pilot-v1-20261010')],live_480_diagnostic_outputs_excluded=True))
    started=time.time();exe=pathlib.Path(os.environ['SystemRoot'])/'System32/compact.exe';compressed=0
    with (a.out/'compact.log').open('w',encoding='utf-8') as log:
        for i in range(0,len(files),16):
            chunk=files[i:i+16]
            result=subprocess.run([str(exe),'/C','/F','/A','/I','/Q',*map(str,chunk)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            log.write(result.stdout.decode('utf-8',errors='replace'));log.flush()
            compressed+=sum(bool(p.stat().st_file_attributes&0x800) for p in chunk)
            save('progress.json',dict(stage='compressing',processed=min(i+16,len(files)),total=len(files),compressed=compressed,seconds=time.time()-started))
    for item in samples:
        path=pathlib.Path(item['path']);assert path.stat().st_size==item['bytes'] and digest(path)==item['sha256']
    save('report.json',dict(state='complete',files=len(files),compressed=compressed,logical_bytes=total,sample_byte_hashes_preserved=True,samples=samples,seconds=time.time()-started,completed_wide_verified=a.include_completed_wide,completed_ppo_verified=a.include_completed_ppo,scope='NTFS compression attributes only, explicit closed combat roots; optional wide root requires sealed960 fights and diagnostics, optional PPO roots require complete CUDA training and exact checkpoint audit receipts. Current480 evaluation and queued sampling pilot excluded. File paths, logical lengths and contents unchanged; sample SHA256 checked before/after. No deletion, gzip conversion, model changes or Go/C/PowerShell source mutation.'))
    save('progress.json',dict(stage='complete',files=len(files),compressed=compressed))
    print(json.dumps(dict(files=len(files),compressed=compressed,logical_bytes=total)),flush=True)

if __name__=='__main__':main()

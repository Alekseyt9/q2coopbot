"""Resume accepted cases in a failed CUDA batch using bounded fresh processes."""
import argparse,copy,pathlib,shutil,subprocess,sys
from process_combat_architecture_pool import read,save,sha,compact_closed_exports


def verify(task):
    out=pathlib.Path(task['out']);data=pathlib.Path(task['data']);meta=read(out/'report.json')
    assert meta['version']=='combat_ppo_rollout_v1' and meta['numerical_verification']=='cuda_verified_v1'
    assert meta['model_sha256']==task['model_sha256']==sha(task['model'])
    assert sha(data/'report.json')==task['native_report_sha256']
    assert sha(data/'native-rollout.jsonl')==meta['native_rollout_sha256']
    assert sha(out/'rollout.jsonl')==meta['rollout_sha256']
    assert sha(out/'sequence.jsonl')==meta['sequence_sha256']==sha(data/'sequence.jsonl')
    assert sha(out/'cuda-verification.json')==meta['cuda_verification_sha256']
    audit=read(out/'cuda-verification.json');assert audit['state']=='passed' and audit['device']=='cuda'
    assert audit['rows']==meta['rows'] and audit['model_sha256']==task['model_sha256']
    for p,d in meta['source_sha256'].items():assert sha(p)==d,p
    return dict(model_sha256=task['model_sha256'],native_report_sha256=task['native_report_sha256'],
        out=str(out),report_sha256=sha(out/'report.json'),rollout_sha256=meta['rollout_sha256'],
        cuda_verification_sha256=meta['cuda_verification_sha256'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--processing',type=pathlib.Path,required=True)
    ap.add_argument('--pool',type=pathlib.Path,required=True);ap.add_argument('--plan-index',type=int,required=True)
    ap.add_argument('--chunk-size',type=int,default=8);a=ap.parse_args()
    root=a.processing.resolve();repo=pathlib.Path(__file__).resolve().parents[1];base=repo/'workspace/artifacts'
    assert root.is_relative_to(base) and 1<=a.chunk_size<=16
    assert read(root/'progress.json')['stage']=='failed' and not (root/'report.json').exists()
    request_path=root/'cuda-request.json';request=read(request_path);pool_path=a.pool/'report.json';pool=read(pool_path)
    assert pool['state']=='complete' and pool['source_unchanged'] and all(not j['error'] for j in pool['jobs'])
    jobs=sorted((j for j in pool['jobs'] if j['plan_index']==a.plan_index),key=lambda j:j['job'])
    assert len(jobs)==len(request['tasks'])==112 and all(j['mode']=='learned' for j in jobs)
    receipt=root/'cuda-resume-1450-v1';assert not receipt.exists();receipt.mkdir()
    pins={request_path:sha(request_path),pool_path:sha(pool_path),root/'q2ppo-data.exe':sha(root/'q2ppo-data.exe')}
    pending=[];reused=[];archives=[]
    for i,(task,job) in enumerate(zip(request['tasks'],jobs)):
        data,out=pathlib.Path(task['data']).resolve(),pathlib.Path(task['out']).resolve()
        assert data.parent==out.parent and data.is_relative_to(root) and out.is_relative_to(root)
        assert data.parent.name==f"case-{job['job']:04d}"
        assert sha(data/'report.json')==task['native_report_sha256'] and sha(task['model'])==task['model_sha256']
        if (out/'report.json').exists():reused.append(dict(task=i,**verify(task)));continue
        if out.exists():
            target=out.with_name(out.name+'-failed-1450-v1')
            assert out.resolve().is_relative_to(root) and target.resolve().is_relative_to(root)
            assert not out.is_symlink() and not target.exists()
            out.rename(target);archives.append(dict(task=i,path=str(target)))
        pending.append((i,task))
    save(receipt/'preflight.json',dict(state='verified-reuse',reused=reused,pending=[i for i,_ in pending],archives=archives,
        source_pins={str(p):d for p,d in pins.items()},training_performed=False))
    def run(command,path):
        with path.open('w',encoding='utf-8') as log:
            p=subprocess.run(list(map(str,command)),cwd=repo,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert p.returncode==0,str(path)
    try:
        chunks=[]
        for offset in range(0,len(pending),a.chunk_size):
            group=pending[offset:offset+a.chunk_size];index=len(chunks)
            cp=receipt/f'chunk-{index}-request.json';cr=receipt/f'chunk-{index}-receipt'
            save(cp,dict(version=request['version'],compact_finalized_streams=True,tasks=[t for _,t in group]))
            save(root/'progress.json',dict(stage='cuda-resume',reused_cases=len(reused),completed_new_cases=offset,pending_cases=len(pending),chunk=index,training_performed=False))
            run([sys.executable,repo/'scripts/finalize_combat_cuda_batch.py','--request',cp,'--receipt-root',cr],receipt/f'chunk-{index}.log')
            completed=read(cr/'report.json');assert completed['state']=='complete' and completed['device']=='cuda'
            assert completed['request_sha256']==sha(cp) and len(completed['tasks'])==len(group)
            for _,task in group:verify(task)
            chunks.append(dict(request=str(cp),request_sha256=sha(cp),receipt=str(cr/'report.json'),receipt_sha256=sha(cr/'report.json'),tasks=[i for i,_ in group]))
        completed=[dict(task=i,**verify(t)) for i,t in enumerate(request['tasks'])]
        batch_path=receipt/'batch-report.json'
        save(batch_path,dict(version='combat_cuda_finalize_batch_receipt_v1',state='complete',device='cuda',
            request_sha256=pins[request_path],tasks=completed,reused_cases=len(reused),chunks=chunks,
            scope='Original valid CUDA outputs reused by source/rollout/sequence/audit SHA. Only unfinished cases finalized on CUDA in bounded fresh processes; failed partial files retained. No fitting or native recapture.'))
        assert not (root/'merged').exists()
        run([root/'q2ppo-data.exe','--merge',','.join(t['out'] for t in request['tasks']),'--out',root/'merged'],receipt/'merge.log')
        compact_closed_exports(root/'merged')
        merged=read(root/'merged/report.json')
        for p,d in pins.items():assert sha(p)==d,str(p)
        model_sha=request['tasks'][0]['model_sha256'];assert all(t['model_sha256']==model_sha for t in request['tasks'])
        plan=read(pool['plans'][a.plan_index]['path'])
        cases=[dict(job=j['job'],seed=j['seed'],split=plan['tasks'][j['task_index']]['split']) for j in jobs]
        save(root/'report.json',dict(state='complete',device='cuda',pool_sha256=pins[pool_path],selected_plan_index=a.plan_index,
            model_sha256=model_sha,cases=cases,rows=merged['rows'],training_performed=False,promotion='Not evaluated',
            cuda_batch_receipt=str(batch_path),cuda_batch_receipt_sha256=sha(batch_path),reused_cuda_cases=len(reused),
            recovery_script_sha256=sha(pathlib.Path(__file__))))
        save(receipt/'complete.json',dict(state='complete',cases=112,reused_cases=len(reused),completed_cases=len(pending),
            processing_report_sha256=sha(root/'report.json'),batch_report_sha256=sha(batch_path),training_performed=False))
        save(root/'progress.json',dict(stage='complete',rows=merged['rows'],resumed=True,training_performed=False))
        print('CUDA processing resumed:',len(reused),'reused,',len(pending),'completed',flush=True)
    except BaseException as error:
        save(root/'progress.json',dict(stage='failed',error=str(error),resumed=True,training_performed=False));raise


if __name__=='__main__':main()

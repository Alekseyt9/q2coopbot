"""Finish failed postprocessing of a closed native evaluation without new games."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from evaluate_registered_combat_models import read, save, sha
from process_combat_architecture_pool import compact_closed_exports
from process_registered_combat_pool_cuda import select_learned_jobs


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evaluation',type=Path,required=True)
    ap.add_argument('--spec',type=Path,required=True)
    ap.add_argument('--resume',action='store_true',help='Continue an interrupted recovery, preserving its original failure receipt')
    a=ap.parse_args();root=a.evaluation.resolve();repo=Path(__file__).resolve().parents[1]
    assert not (root/'complete.json').exists()
    assert read(root/'progress.json')['stage']=='failed'
    fixture=read(root/'fixture-proof.json');spec=read(a.spec)
    assert sha(a.spec)==fixture['spec_sha256'] and sha(spec['registry'])==fixture['registry_sha256']
    native=read(root/'native-complete.json');pool=read(root/'pool/report.json');pool_sha=sha(root/'pool/report.json')
    assert native['state']==pool['state']=='complete' and native['pool_sha256']==pool_sha
    assert pool['source_unchanged'] and pool['usable_captures']==len(pool['jobs'])==fixture['native_cases']
    pins={str(a.spec.resolve()):sha(a.spec),str(root/'pool/report.json'):pool_sha,
          str(root/'native-complete.json'):sha(root/'native-complete.json'),str(root/'fixture-proof.json'):sha(root/'fixture-proof.json')}
    for b in fixture['plans']:assert sha(b['path'])==b['sha256'];pins[b['path']]=b['sha256']
    for b in fixture['models']:
        update=Path(b['update']);assert sha(update/'complete.json')==b['update_seal_sha256']
        closed=read(update/'complete.json')
        for n,k in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):
            assert sha(update/n)==closed[k];pins[str(update/n)]=closed[k]
    for c in native['comparisons']:assert sha(root/('quality-'+c['name']+'.json'))==c['sha256']
    backup=root/'progress-before-cuda-recovery.json'
    if a.resume:assert backup.exists()
    else:assert not backup.exists();backup.write_bytes((root/'progress.json').read_bytes())

    def run(label,script,*args):
        with (root/(label+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,str(repo/'scripts'/script),*map(str,args)],cwd=repo,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert result.returncode==0,str(root/(label+'.log'))

    def verify_closed(task):
        out=Path(task['out']);data=Path(task['data']);r=read(out/'report.json')
        assert r['model_sha256']==task['model_sha256'] and sha(task['model'])==task['model_sha256']
        assert sha(data/'report.json')==task['native_report_sha256']
        assert sha(out/'rollout.jsonl')==r['rollout_sha256']
        assert sha(out/'cuda-verification.json')==r['cuda_verification_sha256']
        proof=read(out/'cuda-verification.json');assert proof['state']=='passed' and proof['device']=='cuda'
        for p,h in r['source_sha256'].items():assert sha(p)==h,p
        return dict(model_sha256=r['model_sha256'],native_report_sha256=task['native_report_sha256'],out=str(out),
                    report_sha256=sha(out/'report.json'),rollout_sha256=r['rollout_sha256'],cuda_verification_sha256=r['cuda_verification_sha256'])

    processed=[];behavior=[]
    try:
        for index,v in enumerate(spec['variants']):
            if v['mode']!='learned':continue
            label=v['label'];save(root/'progress.json',dict(stage='recovering-postprocessing',label=label,processed=processed))
            for script,prefix in (('audit_registered_combat_ownership.py','ownership'),('audit_registered_target_choices.py','behavior'),
                                  ('audit_registered_weapon_choices.py','weapon'),('audit_registered_execution_actions.py','execution')):
                target=root/(prefix+'-'+label+'.json')
                if target.exists():
                    r=read(target);assert (prefix=='behavior' or r['state']=='complete') and r['pool_sha256']==pool_sha
                    assert r['selected_plan_index']==index
                    assert r['script_sha256']==sha(repo/'scripts'/script)
                else:run('recovery-'+prefix+'-'+label,script,'--pool',root/'pool','--plan-index',index,'--out',target)
            behavior+=['--audit',label+'='+str(root/('behavior-'+label+'.json'))]
            if not v['cuda_replay']:continue
            directory=root/('processing-'+label)
            if not directory.exists():
                run('recovery-processing-'+label,'process_registered_combat_pool_cuda.py','--pool',root/'pool','--plan-index',index,'--out',directory)
            elif not (directory/'report.json').exists():
                request=read(directory/'cuda-request.json');tasks=request['tasks'];missing=[]
                for task in tasks:
                    out=Path(task['out'])
                    if (out/'report.json').exists():verify_closed(task)
                    else:assert not out.exists(),str(out);missing.append(task)
                receipts=[]
                for offset in range(0,len(missing),8):
                    n=offset//8;req=directory/f'cuda-recovery-{n}-request.json';receipt=directory/f'cuda-recovery-{n}'
                    assert not req.exists() and not receipt.exists()
                    save(req,dict(version='combat_cuda_finalize_batch_v1',compact_finalized_streams=True,tasks=missing[offset:offset+8]))
                    run('recovery-'+label+'-'+str(n),'finalize_combat_cuda_batch.py','--request',req,'--receipt-root',receipt)
                    receipts.append(dict(request=str(req),request_sha256=sha(req),report=str(receipt/'report.json'),report_sha256=sha(receipt/'report.json')))
                closed=[dict(verify_closed(t),task=i) for i,t in enumerate(tasks)]
                batch=directory/'cuda-batch';assert not batch.exists();batch.mkdir()
                save(batch/'report.json',dict(state='complete',device='cuda',version='combat_cuda_finalize_batch_receipt_v1',tasks=closed,chunks=receipts,
                    request_sha256=sha(directory/'cuda-request.json'),reused_cases=len(tasks)-len(missing),recovery=True))
                save(batch/'progress.json',dict(stage='complete',report_sha256=sha(batch/'report.json')))
                exporter=directory/'q2ppo-data.exe';assert exporter.exists()
                with (directory/'recovery-merge.log').open('w',encoding='utf-8') as log:
                    result=subprocess.run([str(exporter),'--merge',','.join(t['out'] for t in tasks),'--out',str(directory/'merged')],cwd=repo,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                assert result.returncode==0,str(directory/'recovery-merge.log');compact_closed_exports(directory/'merged')
                jobs=select_learned_jobs(pool,index)
                cases=[dict(job=j['job'],seed=j['seed'],split=spec['split']) for j in jobs]
                save(directory/'report.json',dict(state='complete',device='cuda',pool_sha256=pool_sha,selected_plan_index=index,
                     model_sha256=tasks[0]['model_sha256'],cases=cases,rows=read(directory/'merged/report.json')['rows'],
                     training_performed=False,promotion=False,recovery=True,reused_cases=len(tasks)-len(missing),cuda_receipts=receipts))
                save(directory/'progress-before-recovery.json',read(directory/'progress.json'))
                save(directory/'progress.json',dict(stage='complete',rows=read(directory/'merged/report.json')['rows'],recovery=True))
            meta=read(directory/'report.json');assert meta['state']=='complete' and meta['device']=='cuda' and meta['pool_sha256']==pool_sha
            processed.append(dict(label=label,cases=len(meta['cases']),rows=meta['rows'],report_sha256=sha(directory/'report.json')))
        run('recovery-action-efficiency','summarize_registered_action_efficiency.py',*behavior,'--damage',root/'damage-timing.json','--out',root/'action-efficiency.json')
        for p,h in pins.items():assert sha(p)==h,p
        save(root/'complete.json',dict(state='complete',native_cases=len(pool['jobs']),cuda_cases=sum(p['cases'] for p in processed),
             spec_sha256=sha(a.spec),pool_sha256=pool_sha,comparisons=native['comparisons'],processed=processed,readiness=fixture['readiness'],
             training_performed=False,promotion=False,cuda_postprocessing_recovery=True,original_failure_sha256=sha(backup),
             script_sha256=sha(__file__),inputs=pins,scope='Original accepted native captures unchanged. Completed CUDA cases reused; only missing numerical postprocessing recovered. No new games or fitting.'))
        save(root/'progress.json',dict(stage='complete',native_cases=len(pool['jobs']),processed=processed,cuda_postprocessing_recovery=True))
        print(json.dumps(read(root/'complete.json'),indent=2))
    except BaseException as e:
        save(root/'progress.json',dict(stage='failed',error=str(e),processed=processed,cuda_postprocessing_recovery=True));raise


if __name__=='__main__':main()

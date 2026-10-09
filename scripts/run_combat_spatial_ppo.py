"""Queue native spatial trajectories after sealed evaluation, then CUDA PPO."""
import argparse,ctypes,os,pathlib,time,sys
from ctypes import wintypes
from process_combat_architecture_pool import read,save,sha,run
from run_combat_target_refresh import pool

def wait_process(pid,progress,stage='waiting_for_evaluation'):
    # Retain an OS process handle, so PID reuse cannot attach to another job.
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
    handle=kernel.OpenProcess(0x00100000,False,pid)
    if not handle:
        error=ctypes.get_last_error()
        if error==87:return  # PID authoritatively absent; sealed state still required.
        raise OSError(error,'Unable to observe evaluation process')
    try:
        save(progress,dict(stage=stage,observed_process_pid=pid,process_handle_open=True))
        while True:
            status=kernel.WaitForSingleObject(handle,30000)
            if status==0:return
            if status!=258:raise OSError(ctypes.get_last_error(),'Evaluation process observation failed')
    finally:kernel.CloseHandle(handle)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evaluation-root',type=pathlib.Path,required=True);ap.add_argument('--evaluation-pid',type=int,required=True);ap.add_argument('--capture-root',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args()
    assert os.name=='nt' and a.evaluation_pid>0
    repo=pathlib.Path(__file__).resolve().parents[1];capture=a.capture_root.resolve();evaluation=a.evaluation_root.resolve();out=a.out.resolve()
    prep=read(capture/'preparation.json');assert prep['state']=='prepared' and prep['training_device']=='cuda'
    assert prep['models_sha256']==sha(capture/'models.json') and prep['config_sha256']==sha(capture/'config.json')
    assert not out.exists() and not (capture/'pool').exists() and not (capture/'execution.json').exists()
    save(capture/'execution.json',dict(stage='queued',evaluation_root=str(evaluation),evaluation_pid=a.evaluation_pid,driver_sha256=sha(__file__),processing_root=str(out)))
    try:
        wait_process(a.evaluation_pid,capture/'execution.json')
        terminal=read(evaluation/'progress.json');assert terminal['stage']=='complete'
        assert terminal['quality_report_sha256']==sha(evaluation/'quality-report.json')
        proof=read(evaluation/'recovery/verified-members.json');assert proof['state']=='complete' and proof['protocol_sha256']==sha(evaluation/'protocol.json')
        # The running evaluation predates the new strata reporting hook.
        run([sys.executable,repo/'scripts/report_combat_evaluation_strata.py','--root',evaluation],evaluation/'strata.log')
        models=read(capture/'models.json');plans=[pathlib.Path(p) for p in read(capture/'plans.json')]
        assert len(models)==len(plans)==2
        for binding,plan in zip(models,plans):
            schedule=read(plan);assert binding['plan']==str(plan) and sha(binding['model'])==schedule['model_sha256']
            assert all(t['split']=='train' and t['modes']==['learned'] for t in schedule['tasks'])
        save(capture/'execution.json',dict(stage='collecting_own_policy',episodes=160,evaluation_report_sha256=sha(evaluation/'quality-report.json')))
        pwsh=repo/'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe'
        pool(repo,pwsh,plans,capture/'pool',capture/'collect.log')
        os.environ['GOCACHE']=str(repo/'workspace/build/go-cache');os.environ['GOTOOLCHAIN']='auto'
        exporter=repo/'workspace/build/q2ppo-data-spatial-v1.exe'
        run(['go','build','-o',exporter,'./cmd/q2ppo-data'],capture/'build-exporter.log')
        save(capture/'execution.json',dict(stage='cuda_ppo_processing',episodes=160,exporter_sha256=sha(exporter)))
        legacy=repo/'workspace/artifacts/aproc-v1-20261007'
        run([sys.executable,repo/'scripts/process_combat_architecture_pool.py','--capture-root',capture,'--out',out,'--config',capture/'config.json','--exporter',exporter,'--anchor',legacy/'anchor.json','--bank',legacy/'bank.json','--export-workers',4],capture/'process.log')
        result=read(out/'report.json');assert result['state']=='complete' and len(result['training'])==2 and all(r['device']=='cuda' for r in result['training'])
        save(capture/'execution.json',dict(stage='complete',processing_report_sha256=sha(out/'report.json'),promotion='Pending paired native validation; no baseline replacement'))
    except Exception as error:
        save(capture/'execution.json',dict(stage='failed',error=str(error),promotion='None; partial captures/updates preserved'))
        raise

if __name__=='__main__':main()

"""Wait for a verified live training driver, then freeze/run paired validation."""
import argparse, ctypes, json, pathlib, subprocess, sys, time
from process_combat_architecture_pool import read, sha, save, run


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--processing-root',type=pathlib.Path,required=True)
    ap.add_argument('--capture-root',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    ap.add_argument('--driver-pid',type=int,required=True)
    a=ap.parse_args();repo=pathlib.Path(__file__).resolve().parents[1]
    capture=a.capture_root.resolve();processing=a.processing_root.resolve();out=a.out.resolve()
    assert not out.exists()
    receipt=read(capture/'driver-process.json');assert receipt['pid']==a.driver_pid
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32]
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x100000|0x1000,False,a.driver_pid)
    assert handle,'Cannot verify live training process; inspect terminal receipts before retry'
    # Check identity as well as PID before taking a stable process handle.
    pwsh=repo/'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe'
    command=f"Get-CimInstance Win32_Process -Filter 'ProcessId={a.driver_pid}' | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    result=subprocess.run([str(pwsh),'-NoProfile','-Command',command],capture_output=True,text=True,check=True,
                          creationflags=subprocess.CREATE_NO_WINDOW)
    live=json.loads(result.stdout)
    assert 'run_combat_distillation_training.ps1' in live['CommandLine']
    assert str(capture).replace('\\','/').lower() in live['CommandLine'].replace('\\','/').lower()
    stage=capture/'evaluation-continuation.json'
    save(stage,dict(state='waiting_verified_driver',pid=a.driver_pid,out=str(out)))
    try:
        while True:
            status=kernel.WaitForSingleObject(handle,30000)
            if status==0:break
            assert status==258,'Process observation failed'
    finally:kernel.CloseHandle(handle)
    report=read(processing/'report.json')
    assert report['state']=='complete' and len(report['training'])==3
    save(stage,dict(state='preparing_evaluation',processing_report_sha256=sha(processing/'report.json'),out=str(out)))
    run([sys.executable,repo/'scripts/prepare_combat_architecture_evaluation.py',
         '--processing-root',processing,'--out',out,'--compiler',capture/'q2episode.exe',
         '--seed-offset',24,'--count',4],capture/'prepare-evaluation.log')
    protocol=read(out/'protocol.json');assert protocol['total_episodes']==560
    save(stage,dict(state='evaluating',episodes=560,out=str(out)))
    run([pwsh,'-NoProfile','-File',repo/'scripts/run_combat_architecture_evaluation.ps1',
         '-EvaluationRoot',out,'-Port',34700],capture/'evaluation-driver.log')
    quality=read(out/'quality-report.json');assert quality['state']=='complete' and quality['episodes']==560
    save(stage,dict(state='complete',out=str(out),quality_report_sha256=sha(out/'quality-report.json')))
    print(json.dumps(quality['variants']),flush=True)


if __name__=='__main__':main()

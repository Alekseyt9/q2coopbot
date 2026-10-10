"""Wait on a retained evaluation process handle, then audit sealed data only."""
import argparse,pathlib,sys
from process_combat_architecture_pool import read,save,sha,run
from run_combat_spatial_ppo import wait_process


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=pathlib.Path,required=True)
    ap.add_argument('--wait-pid',type=int,required=True); a=ap.parse_args()
    root=a.root.resolve(); repo=pathlib.Path(__file__).resolve().parents[1]
    progress=root/'ownership-analysis-progress.json'
    assert not progress.exists()
    try:
        wait_process(a.wait_pid,progress,'waiting_for_wide_evaluation')
        state=read(root/'progress.json')
        assert state['stage']=='complete' and state['diagnostics_complete']
        assert state['quality_report_sha256']==sha(root/'quality-report.json')
        assert state['diagnostics_acceptance_sha256']==sha(root/'diagnostics-acceptance.json')
        audit=read(root/'sampling-config-audit.json')
        assert audit['state']=='complete' and not audit['repeated_stochastic_execution_configs']
        save(progress,dict(stage='auditing_ownership',promotion=None))
        run([sys.executable,repo/'scripts/report_combat_control_ownership.py','--root',root],root/'control-ownership.log')
        save(progress,dict(stage='auditing_physical_storage',promotion=None))
        run([sys.executable,repo/'scripts/audit_combat_physical_storage.py','--root',root,'--out',root/'storage-final.json'],root/'storage-final.log')
        save(progress,dict(stage='complete',control_ownership_sha256=sha(root/'control-ownership.json'),
            physical_storage_sha256=sha(root/'storage-final.json'),sampling_config_sha256=sha(root/'sampling-config-audit.json'),
            quality_sha256=sha(root/'quality-report.json'),promotion=None))
    except Exception as error:
        save(progress,dict(stage='failed',error=str(error),promotion=None)); raise


if __name__=='__main__': main()

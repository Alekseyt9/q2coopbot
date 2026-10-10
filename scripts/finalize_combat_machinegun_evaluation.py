"""Finalize diagnostics only after sealed core quality results."""
import argparse,pathlib,sys
from process_combat_architecture_pool import read,save,sha,run

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);a=ap.parse_args();root=a.root.resolve()
    repo=pathlib.Path(__file__).resolve().parents[1];protocol=read(root/'protocol.json');quality=read(root/'quality-report.json');proof=read(root/'recovery/verified-members.json')
    assert quality['state']=='complete' and proof['state']=='complete'
    assert quality['protocol_sha256']==proof['protocol_sha256']==sha(root/'protocol.json')
    assert quality['episodes']==protocol['total_episodes']==len(proof['members'])
    progress=dict(stage='complete',episodes=quality['episodes'],quality_report_sha256=sha(root/'quality-report.json'),diagnostics_complete=False,promotion=None)
    save(root/'progress.json',progress)
    reports={
        'report_combat_evaluation_strata.py':'quality-strata.json',
        'report_combat_machinegun_hits.py':'machinegun-hits.json',
        'report_combat_machinegun_aim.py':'machinegun-selected-target-aim.json',
        'report_combat_blaster_hits.py':'blaster-projectile-hits.json',
        'report_combat_selected_target_aim.py':'selected-target-aim.json',
        'report_combat_aim_modes.py':'aim-modes.json',
        'report_combat_first_attack.py':'first-attack.json',
        'report_combat_life_tail.py':'first-life-tail.json',
        'report_combat_movement.py':'first-life-movement.json',
    }
    try:
        for script,target in reports.items():
            run([sys.executable,repo/'scripts'/script,'--root',root],root/(target+'.log'))
            assert read(root/target)['protocol_sha256']==sha(root/'protocol.json')
        save(root/'diagnostics-acceptance.json',dict(state='complete',quality_report_sha256=sha(root/'quality-report.json'),reports={p:sha(root/p) for p in reports.values()},scope='Core sealed quality completed before diagnostic passes; no recapture, source change, training or promotion.'))
        progress.update(diagnostics_complete=True,diagnostics_acceptance_sha256=sha(root/'diagnostics-acceptance.json'));save(root/'progress.json',progress)
    except Exception as error:
        progress['diagnostics_error']=str(error);save(root/'progress.json',progress);raise

if __name__=='__main__':main()

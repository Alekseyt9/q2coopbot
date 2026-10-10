"""Dispatch one predeclared independent test after sealed development selection."""
import argparse,pathlib,sys,shutil
from process_combat_architecture_pool import read,save,sha,run
from run_combat_target_refresh import compile_plan,pool

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preparation',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args()
    repo=pathlib.Path(__file__).resolve().parents[1];prep=a.preparation.resolve();out=a.out.resolve()
    selection=read(prep/'selection-protocol.json');inventory=read(prep/'seed-inventory.json')
    assert selection['state']=='prepared_not_dispatched' and inventory['state']=='audited' and selection['seed_inventory_sha256']==sha(prep/'seed-inventory.json')
    development=pathlib.Path(selection['development_evaluation']);protocol=read(development/'protocol.json');quality=read(development/'quality-report.json');progress=read(development/'progress.json');proof=read(development/'recovery/verified-members.json')
    assert progress['stage']=='complete' and progress['diagnostics_complete'] and quality['state']=='complete'
    assert quality['protocol_sha256']==proof['protocol_sha256']==selection['development_protocol_sha256']==sha(development/'protocol.json')
    assert quality['episodes']==len(proof['members'])==protocol['total_episodes']
    variants={v['variant']:v for v in quality['variants']};candidates=[v for k,v in variants.items() if k.endswith('-after')]
    candidate=min(candidates,key=lambda v:(-v['wins'],v['deaths'],v['mean_received_damage'],v['variant']))
    reference=variants['rules-baseline'];eligible=candidate['wins']>reference['wins'] or candidate['wins']==reference['wins'] and candidate['deaths']<reference['deaths']
    assert not out.exists();out.mkdir(parents=True)
    decision=dict(version='combat_independent_test_decision_v1',candidate=candidate,reference=reference,eligible=eligible,selection_protocol_sha256=sha(prep/'selection-protocol.json'),development_quality_sha256=sha(development/'quality-report.json'),scope='Selection uses development only. Failure of eligibility reserves test conditions; no candidate rotation on test outcomes.')
    save(out/'selection-decision.json',decision)
    if not eligible:
        save(out/'progress.json',dict(stage='test_reserved',reason='Development eligibility not met',decision_sha256=sha(out/'selection-decision.json')))
        print('Independent test remains reserved; development eligibility not met',flush=True);return
    # A second run with another output path is not authorized by this protocol.
    reservation=prep/'dispatch-reservation.json';assert not reservation.exists(),'Independent test already reserved for dispatch'
    save(reservation,dict(state='reserved',output=str(out),candidate=candidate['variant'],decision_sha256=sha(out/'selection-decision.json')))
    try:
        by_variant={e['model']+'-'+e['label']:e for e in protocol['evaluations']}
        chosen=[('selected',candidate['variant']),('parent3','parent3-before'),('firebc','firebc-baseline'),('rules','rules-baseline')]
        plans=[];entries=[]
        registry=read(by_variant[candidate['variant']]['plan'])['registry_path']
        for name,key in chosen:
            entry=by_variant[key];sourceplan=read(entry['plan']);weights=None
            if sourceplan.get('model_path'):
                source=pathlib.Path(sourceplan['model_path']);assert sha(source)==entry['deterministic_weights_sha256']
                weights=out/(name+'-weights.json');shutil.copyfile(source,weights);assert sha(weights)==sha(source) and read(weights)['deterministic']
            plan=compile_plan(repo/'workspace/build/q2episode-spatial-v2.exe',registry,repo,weights,out/name,'test',selection['families'],selection['seed_offset'],selection['count_per_family'])
            compiled=read(plan)
            assert {t['episode']['id']:t['seeds'] for t in compiled['tasks']}=={c['episode']:c['seeds'] for c in inventory['conditions']}
            plans.append(plan);entries.append(dict(model=name,label='test',root=str(out/name/'capture'),plan=str(plan),plan_sha256=sha(plan),source_weights_sha256=sha(weights) if weights else None,deterministic_weights_sha256=sha(weights) if weights else None))
        save(out/'protocol.json',dict(version='combat_independent_test_v1',evaluations=entries,families=selection['families'],episodes_per_model=160,total_episodes=640,slots=16,timescale=2,comparison_reference='rules-test',selection_decision_sha256=sha(out/'selection-decision.json'),seed_inventory_sha256=sha(prep/'seed-inventory.json'),scope='Predeclared single candidate selected solely from development. Reserved registered test distributions,160 conditions per arm, parent3/FireBC/rules controls. No training or candidate reselection on test results; whole campaigns remain separate.'))
        save(out/'progress.json',dict(stage='testing',episodes=640))
        pool(repo,repo/'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe',plans,out/'pool',out/'capture.log')
        manifest=read(next((out/'selected/capture').glob('case-*/s-*/manifest.json')))
        run([sys.executable,repo/'scripts/verify_combat_evaluation_members.py','--root',out,'--source-fingerprint',manifest['source_fingerprint'],'--native-fingerprint',manifest['native_source_fingerprint']],out/'verify.log')
        run([sys.executable,repo/'scripts/report_combat_architecture_evaluation.py','--root',out,'--member-proof',out/'recovery/verified-members.json'],out/'quality.log')
        save(out/'progress.json',dict(stage='quality_complete',episodes=640,quality_report_sha256=sha(out/'quality-report.json'),promotion=None))
        save(reservation,dict(state='captured',output=str(out),candidate=candidate['variant'],quality_report_sha256=sha(out/'quality-report.json')))
    except Exception as error:
        save(out/'progress.json',dict(stage='failed',error=str(error)));raise

if __name__=='__main__':main()

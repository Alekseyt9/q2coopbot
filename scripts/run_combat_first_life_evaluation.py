"""Matched first-life native validation after a sealed CUDA continuation."""
import argparse, os, pathlib, sys
from process_combat_architecture_pool import read, save, sha, run
from run_combat_spatial_ppo import wait_process
from run_combat_target_refresh import compile_plan, pool


def main():
    ap = argparse.ArgumentParser()
    for name in ('capture','processing','out'):
        ap.add_argument('--'+name,type=pathlib.Path,required=True)
    ap.add_argument('--wait-pid',type=int)
    ap.add_argument('--execution-receipt',type=pathlib.Path,help='Sealed recovery execution receipt; original failed capture receipt remains preserved')
    ap.add_argument('--seed-offset',type=int,default=24)
    a = ap.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    capture, processing, out = [p.resolve() for p in (a.capture,a.processing,a.out)]
    assert not out.exists() and a.seed_offset >= 0
    out.mkdir()
    try:
        if a.wait_pid:
            wait_process(a.wait_pid,out/'progress.json',stage='waiting_for_first_life_cuda_update')
        execution = a.execution_receipt.resolve() if a.execution_receipt else capture/'execution.json'
        completed = read(execution)
        assert completed['stage'] == 'complete' and completed['processing_report_sha256'] == sha(processing/'report.json')
        result = read(processing/'report.json')
        assert result['state'] == 'complete' and len(result['training']) == 1
        assert result['protocol']['cuda_finalization'] == 'independent_cases_single_interpreter_v1'
        bindings = read(capture/'models.json')
        assert len(bindings) == 1
        binding = bindings[0]
        update = processing/binding['id']/'update'
        seal, report = read(update/'complete.json'), read(update/'report.json')
        for filename,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
            assert sha(update/filename) == seal[key]
        assert report['device'] == 'cuda' and report['behavior_sha256'] == sha(binding['model'])
        assert report['updates_completed'] == binding['parent_updates_completed']+1
        audit = read(update/'checkpoint-cuda-audit.json')
        assert audit['actor_value_std_exact'] and audit['optimizer_state_exact']
        assert audit['weights_sha256'] == seal['weights_sha256'] and audit['checkpoint_sha256'] == seal['checkpoint_sha256']
        references = read(repo/'workspace/artifacts/target-refresh-v2-20261010/evaluation/protocol.json')
        firebc = next(e for e in references['evaluations'] if e['model'] == 'firebc')
        reference_plan = read(firebc['plan'])
        reference_weights = pathlib.Path(reference_plan['model_path'])
        assert sha(reference_weights) == firebc['deterministic_weights_sha256']
        template = read(binding['plan'])
        families = [t['episode']['id'] for t in template['tasks']]
        assert len(families) == len(set(families)) == 20
        assert all(a.seed_offset+4 <= t['episode']['splits']['validation']['count'] for t in template['tasks']), 'Selection exceeds a registered validation cohort'
        os.environ['GOCACHE'] = str(repo/'workspace/build/go-cache')
        os.environ['GOTOOLCHAIN'] = 'auto'
        compiler = out/'q2episode.exe'
        run(['go','build','-buildvcs=false','-o',compiler,'./cmd/q2episode'],out/'build-compiler.log')
        entries, plans = [], []
        variants = [('first-life-ppo','before',pathlib.Path(binding['model'])),
                    ('first-life-ppo','after',update/'weights.json'),
                    ('firebc','baseline',reference_weights),('rules','baseline',None)]
        for model,label,source in variants:
            weight = None
            if source:
                frozen = read(source)
                frozen.update(deterministic=True,sampling_seed=0)
                weight = out/(model+'-'+label+'-weights.json')
                save(weight,frozen)
            plan = compile_plan(compiler,template['registry_path'],repo,weight,out/(model+'-'+label),'validation',families,a.seed_offset)
            plans.append(plan)
            entries.append(dict(model=model,label=label,root=read(plan)['output_root'],plan=str(plan),
                plan_sha256=sha(plan),source_weights_sha256=sha(source) if source else None,
                deterministic_weights_sha256=sha(weight) if weight else None))
        for plan in plans[1:]:
            for left,right in zip(read(plans[0])['tasks'],read(plan)['tasks']):
                assert left['episode'] == right['episode'] and left['seeds'] == right['seeds']
                assert left.get('instances') == right.get('instances')
        save(out/'protocol.json',dict(version='combat_first_life_update_validation_v1',evaluations=entries,
            families=families,total_episodes=320,episodes_per_model=80,slots=16,timescale=2,
            validation_seed_offset=a.seed_offset,comparison_reference='firebc-baseline',
            processing_report_sha256=sha(processing/'report.json'),
            execution_receipt=str(execution),execution_receipt_sha256=sha(execution),
            training=dict(episodes=80,device='cuda',updates_completed=report['updates_completed'],
                rows=report['rows'],actor_steps=report['actor_steps'],report_sha256=seal['report_sha256']),
            scope='Parent vs fresh CUDA continuation, FireBC and rules on identical validation conditions. All arms use the same verified first-life death/goal stop and original reward. Development validation; no final-test superiority or automatic promotion.'))
        save(out/'progress.json',dict(stage='paired_native_evaluation',episodes=320))
        pool(repo,repo/'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe',plans,out/'pool',out/'evaluation.log')
        manifest = read(next(pathlib.Path(entries[0]['root']).glob('case-*/s-*/manifest.json')))
        run([sys.executable,repo/'scripts/verify_combat_evaluation_members.py','--root',out,
            '--source-fingerprint',manifest['source_fingerprint'],'--native-fingerprint',manifest['native_source_fingerprint']],out/'verify.log')
        run([sys.executable,repo/'scripts/report_combat_architecture_evaluation.py','--root',out,
            '--member-proof',out/'recovery/verified-members.json'],out/'quality.log')
        for tool in ('evaluation_strata','selected_target_aim','blaster_hits','aim_modes','first_attack','life_tail'):
            run([sys.executable,repo/f'scripts/report_combat_{tool}.py','--root',out],out/f'{tool}.log')
        save(out/'progress.json',dict(stage='complete',quality_report_sha256=sha(out/'quality-report.json'),diagnostics_pending=['movement'],promotion='Not assessed'))
        # Movement requires an already sealed native quality report.
        run([sys.executable,repo/'scripts/report_combat_movement.py','--root',out],out/'movement.log')
        save(out/'progress.json',dict(stage='complete',quality_report_sha256=sha(out/'quality-report.json'),diagnostics_complete=True,promotion='Not assessed'))
    except Exception as error:
        save(out/'progress.json',dict(stage='failed',error=str(error),promotion='None; evidence retained'))
        raise


if __name__ == '__main__':
    main()

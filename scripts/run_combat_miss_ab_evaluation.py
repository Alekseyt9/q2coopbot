"""Paired native before/after evaluation of matched miss-reward CUDA updates."""
import argparse, os, pathlib, sys
from process_combat_architecture_pool import read, save, sha, run
from run_combat_spatial_ppo import wait_process
from run_combat_target_refresh import compile_plan, pool


def main():
    ap = argparse.ArgumentParser()
    for name in ('capture','processing','out'):
        ap.add_argument('--'+name,type=pathlib.Path,required=True)
    ap.add_argument('--wait-pid',type=int)
    ap.add_argument('--seed-offset',type=int,default=20)
    a = ap.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    capture, processing, out = [p.resolve() for p in (a.capture,a.processing,a.out)]
    assert not out.exists() and a.seed_offset >= 0
    out.mkdir()
    try:
        if a.wait_pid:
            wait_process(a.wait_pid,out/'progress.json',stage='waiting_for_matched_cuda_updates')
        completed = read(capture/'execution.json')
        assert completed['stage']=='complete' and completed['processing_report_sha256']==sha(processing/'report.json')
        result = read(processing/'report.json')
        assert result['state']=='complete' and result['training_device']=='cuda'
        assert {e['binding'] for e in result['branches']}=={'control','miss'}
        for e in result['branches']:
            assert sha(e['report'])==e['report_sha256'] and read(e['report'])['state']=='complete'
        bindings = {b['id']:b for b in read(capture/'models.json')}
        variants, training = [], {}
        for name in ('control','miss'):
            update = processing/name/name/'update'
            seal, report = read(update/'complete.json'), read(update/'report.json')
            for filename,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
                assert sha(update/filename)==seal[key]
            assert report['device']=='cuda' and report['behavior_sha256']==sha(bindings[name]['model'])
            audit = read(update/'checkpoint-cuda-audit.json')
            assert audit['actor_value_std_exact'] and audit['optimizer_state_exact']
            assert audit['weights_sha256']==seal['weights_sha256'] and audit['checkpoint_sha256']==seal['checkpoint_sha256']
            training[name] = dict(report_sha256=seal['report_sha256'],rows=report['rows'],actor_steps=report['actor_steps'],
                                  reward_config_sha256=read(capture/(name+'-config.json'))['objective_reward_sha256'])
            variants.extend([('miss-ab-'+name,'before',pathlib.Path(bindings[name]['model'])),
                             ('miss-ab-'+name,'after',update/'weights.json')])
        assert sha(bindings['control']['model'])==sha(bindings['miss']['model'])
        old = read(repo/'workspace/artifacts/target-refresh-v2-20261010/evaluation/protocol.json')
        reference = next(e for e in old['evaluations'] if e['model']=='firebc')
        template = read(reference['plan'])
        variants.extend([('firebc','baseline',pathlib.Path(template['model_path'])),('rules','baseline',None)])
        families = [t['episode']['id'] for t in read(bindings['control']['plan'])['tasks']]
        assert len(families)==20
        os.environ['GOCACHE']=str(repo/'workspace/build/go-cache')
        os.environ['GOTOOLCHAIN']='auto'
        compiler=repo/'workspace/build/q2episode-miss-ab-eval-v1.exe'
        run(['go','build','-buildvcs=false','-o',compiler,'./cmd/q2episode'],out/'build-compiler.log')
        entries, plans = [], []
        for model,label,source in variants:
            weight=None
            if source:
                frozen=read(source)
                frozen.update(deterministic=True,sampling_seed=0)
                weight=out/(model+'-'+label+'-weights.json')
                save(weight,frozen)
            plan=compile_plan(compiler,template['registry_path'],repo,weight,out/(model+'-'+label),'validation',families,a.seed_offset)
            plans.append(plan)
            entries.append(dict(model=model,label=label,root=read(plan)['output_root'],plan=str(plan),
                 plan_sha256=sha(plan),source_weights_sha256=sha(source) if source else None,
                 deterministic_weights_sha256=sha(weight) if weight else None))
        save(out/'protocol.json',dict(version='combat_miss_reward_ab_validation_v1',evaluations=entries,families=families,
             total_episodes=480,episodes_per_model=80,slots=16,timescale=2,validation_seed_offset=a.seed_offset,
             comparison_reference='firebc-baseline',comparison_stage='matched_confirmed_miss_objective',training=training,
             processing_report_sha256=sha(processing/'report.json'),
             scope=f'Equal80 fresh training episodes each; identical initial actors/std and critic/Adam reset. Training reward differs only by confirmed Blaster miss cost-.02. All native evaluation variants use common original reward and current guard, validation{a.seed_offset}..{a.seed_offset+3}. Identical before weights repeated to expose native repeat variance. Reused development validation, not final-test superiority or automatic promotion.'))
        save(out/'progress.json',dict(stage='paired_native_evaluation',episodes=480))
        pool(repo,repo/'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe',plans,out/'pool',out/'evaluation.log')
        manifest=read(next(pathlib.Path(entries[0]['root']).glob('case-*/s-*/manifest.json')))
        run([sys.executable,repo/'scripts/verify_combat_evaluation_members.py','--root',out,
             '--source-fingerprint',manifest['source_fingerprint'],'--native-fingerprint',manifest['native_source_fingerprint']],out/'verify.log')
        run([sys.executable,repo/'scripts/report_combat_architecture_evaluation.py','--root',out,
             '--member-proof',out/'recovery/verified-members.json'],out/'quality.log')
        for tool in ('evaluation_strata','selected_target_aim','blaster_hits','aim_modes'):
            run([sys.executable,repo/f'scripts/report_combat_{tool}.py','--root',out],out/f'{tool}.log')
        run([sys.executable,repo/'scripts/audit_combat_miss_reward.py','--root',out],out/'miss-audit.log')
        save(out/'progress.json',dict(stage='complete',quality_report_sha256=sha(out/'quality-report.json'),promotion='Not assessed'))
        run([sys.executable,repo/'scripts/report_combat_movement.py','--root',out],out/'movement.log')
    except Exception as error:
        save(out/'progress.json',dict(stage='failed',error=str(error),promotion='None; evidence retained'))
        raise


if __name__=='__main__':
    main()

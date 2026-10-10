"""Paired development validation after sealed CUDA coarse/fine PPO updates."""
import argparse,pathlib,sys
from process_combat_architecture_pool import read,save,sha,run
from run_combat_target_refresh import compile_plan,pool
from run_combat_spatial_ppo import wait_process

def main():
    ap=argparse.ArgumentParser()
    for name in ('capture','processing','reference','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    ap.add_argument('--wait-pid',type=int)
    a=ap.parse_args();repo=pathlib.Path(__file__).resolve().parents[1]
    capture,processing,reference,out=[p.resolve() for p in (a.capture,a.processing,a.reference,a.out)]
    assert not out.exists();out.mkdir()
    try:
        if a.wait_pid:wait_process(a.wait_pid,out/'progress.json','waiting_for_cuda_ppo')
        execution=read(capture/'execution.json');assert execution['stage']=='complete'
        assert execution['processing_report_sha256']==sha(processing/'report.json')
        report=read(processing/'report.json');assert report['state']=='complete' and len(report['training'])==2
        baseline=read(reference/'protocol.json');assert read(reference/'progress.json')['diagnostics_complete']
        template=read(baseline['evaluations'][0]['plan']);families=[t['episode']['id'] for t in template['tasks']]
        assert len(set(families))==20
        variants=[]
        for binding in read(capture/'models.json'):
            name=binding['id'];before=pathlib.Path(binding['model']);schedule=read(binding['plan'])
            assert sha(before)==schedule['model_sha256']
            update=processing/name/'update';seal=read(update/'complete.json');audit=read(update/'checkpoint-cuda-audit.json')
            for filename,field in [('weights.json','weights_sha256'),('report.json','report_sha256'),('checkpoint.pt','checkpoint_sha256')]:assert sha(update/filename)==seal[field]
            assert audit['device']=='cuda' and audit['actor_value_std_exact'] and audit['optimizer_state_exact']
            assert audit['weights_sha256']==sha(update/'weights.json') and audit['checkpoint_sha256']==sha(update/'checkpoint.pt')
            variants.extend([(name,'before',before),(name,'after',update/'weights.json')])
        firebc=next(e for e in baseline['evaluations'] if e['model']=='firebc');source=pathlib.Path(read(firebc['plan'])['model_path'])
        assert sha(source)==firebc['deterministic_weights_sha256']
        variants.extend([('firebc','baseline',source),('rules','baseline',None)])
        plans=[];entries=[]
        for name,label,source in variants:
            weights=None
            if source:
                model=read(source);model['deterministic']=True;weights=out/(name+'-'+label+'-weights.json');save(weights,model)
            plan=compile_plan(repo/'workspace/build/q2episode-spatial-v2.exe',template['registry_path'],repo,weights,out/(name+'-'+label),'validation',families,24)
            compiled=read(plan)
            assert {t['episode']['id']:t['seeds'] for t in compiled['tasks']}=={t['episode']['id']:t['seeds'] for t in template['tasks']}
            assert [t['episode'] for t in compiled['tasks']]==[t['episode'] for t in template['tasks']]
            plans.append(plan);entries.append(dict(model=name,label=label,root=str(out/(name+'-'+label)/'capture'),plan=str(plan),plan_sha256=sha(plan),source_weights_sha256=sha(source) if source else None,deterministic_weights_sha256=sha(weights) if weights else None))
        save(out/'protocol.json',dict(version='combat_machinegun_ppo_development_v1',evaluations=entries,families=families,episodes_per_model=80,total_episodes=480,slots=16,timescale=2,comparison_reference='rules-baseline',processing_report_sha256=sha(processing/'report.json'),reference_protocol_sha256=sha(reference/'protocol.json'),scope='Two development-selected BC actors before/after fresh CUDA PPO, FireBC/rules,20 common families and reused validation24..27. New executions with identical registered conditions and first-life stops. No independent test, whole-map acceptance or promotion claim.'))
        save(out/'progress.json',dict(stage='evaluating',episodes=480))
        pool(repo,repo/'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe',plans,out/'pool',out/'evaluation.log')
        manifest=read(next(pathlib.Path(entries[0]['root']).glob('case-*/s-*/manifest.json')))
        run([sys.executable,repo/'scripts/verify_combat_evaluation_members.py','--root',out,'--source-fingerprint',manifest['source_fingerprint'],'--native-fingerprint',manifest['native_source_fingerprint']],out/'verify.log')
        run([sys.executable,repo/'scripts/report_combat_architecture_evaluation.py','--root',out,'--member-proof',out/'recovery/verified-members.json'],out/'quality.log')
        save(out/'progress.json',dict(stage='quality_complete',episodes=480,quality_report_sha256=sha(out/'quality-report.json'),promotion=None))
        run([sys.executable,repo/'scripts/finalize_combat_machinegun_evaluation.py','--root',out],out/'diagnostics.log')
        print('Sealed480 development fights and nine diagnostics',flush=True)
    except Exception as error:
        save(out/'progress.json',dict(stage='failed',error=str(error),promotion=None));raise

if __name__=='__main__':main()

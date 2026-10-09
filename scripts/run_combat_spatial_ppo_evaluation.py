"""Paired native evaluation of two spatial BC/PPO branches, no promotion."""
import argparse,os,pathlib,sys
from process_combat_architecture_pool import read,save,sha,run
from run_combat_target_refresh import compile_plan,pool
from run_combat_spatial_ppo import wait_process

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--capture-root',type=pathlib.Path,required=True);ap.add_argument('--processing-root',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);ap.add_argument('--wait-pid',type=int);ap.add_argument('--seed-offset',type=int,default=24);a=ap.parse_args()
    repo=pathlib.Path(__file__).resolve().parents[1];capture=a.capture_root.resolve();processing=a.processing_root.resolve();out=a.out.resolve();assert a.seed_offset>=0 and not out.exists();out.mkdir()
    try:
        if a.wait_pid:wait_process(a.wait_pid,out/'progress.json',stage='waiting_for_ppo')
        execution=read(capture/'execution.json');assert execution['stage']=='complete','Upstream processing is not complete: '+execution['stage']
        assert execution['processing_report_sha256']==sha(processing/'report.json'),'Upstream report hash differs'
        updates=read(processing/'report.json');assert updates['state']=='complete' and len(updates['training'])==2
        assert updates['protocol']['models_sha256']==sha(capture/'models.json') and updates['protocol']['config_sha256']==sha(capture/'config.json')
        bindings=read(capture/'models.json');assert {b['id'] for b in bindings}=={'instant','postmove'}
        variants=[];training={}
        for binding in bindings:
            name=binding['id'];directory=processing/name/'update';seal=read(directory/'complete.json')
            for filename,field in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:assert sha(directory/filename)==seal[field]
            report=read(directory/'report.json');assert report['device']=='cuda' and report['behavior_sha256']==sha(binding['model'])
            training[name]=dict(report_sha256=sha(directory/'report.json'),actor_steps=report['actor_steps'],rows=report['rows'])
            variants.extend([('spatial-'+name,'before',pathlib.Path(binding['model'])),('spatial-'+name,'after',directory/'weights.json')])
        legacy=read(repo/'workspace/artifacts/target-refresh-v2-20261010/evaluation/protocol.json');reference=next(e for e in legacy['evaluations'] if e['model']=='firebc');template=read(reference['plan'])
        variants.extend([('firebc','baseline',pathlib.Path(template['model_path'])),('rules','baseline',None)])
        family_plan=read(bindings[0]['plan']);families=[t['episode']['id'] for t in family_plan['tasks']];assert len(families)==len(set(families))==20
        os.environ['GOCACHE']=str(repo/'workspace/build/go-cache');os.environ['GOTOOLCHAIN']='auto'
        plans=[];entries=[]
        for name,label,source in variants:
            weight=None
            if source:
                model=read(source);model.update(deterministic=True,sampling_seed=0);weight=out/(name+'-'+label+'-weights.json');save(weight,model)
            branch=out/(name+'-'+label)
            plan=compile_plan(repo/'workspace/build/q2episode-precision-v1.exe',template['registry_path'],repo,weight,branch,'validation',families,a.seed_offset);plans.append(plan)
            entries.append(dict(model=name,label=label,root=str(branch/'capture'),plan=str(plan),plan_sha256=sha(plan),source_weights_sha256=sha(source) if source else None,deterministic_weights_sha256=sha(weight) if weight else None))
        save(out/'protocol.json',dict(version='combat_spatial_ppo_validation_v1',evaluations=entries,families=families,total_episodes=480,episodes_per_model=80,slots=16,timescale=2,validation_seed_offset=a.seed_offset,comparison_reference='firebc-baseline',comparison_stage='spatial_own_policy_cuda_ppo',training=training,processing_report_sha256=sha(processing/'report.json'),scope='Two spatial BC parents before/after fresh native-reward PPO updating whole actor/value/std. Paired validation conditions reused for development across20 families, not untouched final test. FireBC/rules controls; no automatic promotion.'))
        save(out/'progress.json',dict(stage='paired_native_evaluation',episodes=480))
        pool(repo,repo/'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe',plans,out/'pool',out/'evaluation.log')
        manifests=sorted(pathlib.Path(entries[0]['root']).glob('case-*/s-*/manifest.json'));assert len(manifests)==80;manifest=read(manifests[0])
        run([sys.executable,repo/'scripts/verify_combat_evaluation_members.py','--root',out,'--source-fingerprint',manifest['source_fingerprint'],'--native-fingerprint',manifest['native_source_fingerprint']],out/'verify.log')
        run([sys.executable,repo/'scripts/report_combat_architecture_evaluation.py','--root',out,'--member-proof',out/'recovery/verified-members.json'],out/'report.log')
        for tool in ('evaluation_strata','selected_target_aim','blaster_hits','aim_modes'):
            run([sys.executable,repo/('scripts/report_combat_'+tool+'.py'),'--root',out],out/(tool+'.log'))
        save(out/'progress.json',dict(stage='complete',quality_report_sha256=sha(out/'quality-report.json'),promotion='Not assessed; inspect paired outcomes'))
    except Exception as error:
        save(out/'progress.json',dict(stage='failed',error=str(error),promotion='None; partial artifacts preserved'))
        raise

if __name__=='__main__':main()

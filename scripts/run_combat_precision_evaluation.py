"""Paired native validation of coarse/fine aim, with 16 independent slots."""
import argparse,pathlib,os
from process_combat_architecture_pool import read,save,sha,run
from run_combat_target_refresh import compile_plan,pool

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-root',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);ap.add_argument('--comparison-model',type=pathlib.Path);ap.add_argument('--name',choices=['precision','spatial'],default='precision');ap.add_argument('--comparison-name',choices=['precision-eye','precision-muzzle','spatial-postmove'],default='precision-eye');ap.add_argument('--family-plan',type=pathlib.Path);ap.add_argument('--seed-offset',type=int,default=28);a=ap.parse_args()
    assert a.seed_offset>=0
    repo=pathlib.Path(__file__).resolve().parents[1];out=a.out.resolve();out.mkdir(exist_ok=False)
    os.environ['GOCACHE']=str(repo/'workspace/build/go-cache');os.environ['GOTOOLCHAIN']='auto'
    python=pathlib.Path(__import__('sys').executable);pwsh=repo/'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe';compiler=repo/'workspace/build/q2episode-precision-v1.exe'
    reference=read(repo/'workspace/artifacts/target-refresh-v2-20261010/evaluation/protocol.json')
    legacy=next(e for e in reference['evaluations'] if e['model']=='firebc');template=read(legacy['plan']);registry=template['registry_path'];families=reference['families']
    if a.family_plan:
        family_plan=read(a.family_plan);assert family_plan['registry_path']==registry
        families=list(dict.fromkeys(t['episode']['id'] for t in family_plan['tasks']))
    assert families and len(families)==len(set(families));episodes=4*len(families)
    plans=[];entries=[]
    variants=[(a.name,'before',a.model_root/'before-weights.json'),(a.name,'after',a.model_root/'weights.json'),('firebc','baseline',pathlib.Path(template['model_path'])),('rules','baseline',None)]
    if a.comparison_model:variants.append((a.comparison_name,'baseline',a.comparison_model))
    for name,label,source in variants:
        weight=None
        if source:
            frozen=read(source);frozen.update(deterministic=True,sampling_seed=0)
            weight=out/(name+'-'+label+'-weights.json');save(weight,frozen)
        branch=out/(name+'-'+label)
        plan=compile_plan(compiler,registry,repo,weight,branch,'validation',families,a.seed_offset);plans.append(plan)
        entries.append(dict(model=name,label=label,root=str(branch/'capture'),plan=str(plan),plan_sha256=sha(plan),source_weights_sha256=sha(source) if source else None,deterministic_weights_sha256=sha(weight) if weight else None))
    training=read(a.model_root/'report.json')
    scope='Shared spatial fine/mode branch trained; old45/encoder frozen.' if a.name=='spatial' else 'Fine/mode rows only; old45/encoder frozen.'
    scope+=' Validation conditions are development checks, not final test. Before81 deterministic coarse reproduces strong45 parent. Blaster geometric query bootstrap; machinegun labels unknown. Optional comparison variant uses the same seeds. Final test deferred.'
    save(out/'protocol.json',dict(version='combat_precision_validation_v1',evaluations=entries,families=families,total_episodes=episodes*len(variants),episodes_per_model=episodes,validation_seed_offset=a.seed_offset,slots=16,timescale=2,comparison_reference='firebc-baseline',training_report_sha256=sha(a.model_root/'report.json'),target_query_version=training.get('target_query_version','observed_target_aim_query_v1'),comparison_variant=a.comparison_name if a.comparison_model else None,family_plan_sha256=sha(a.family_plan) if a.family_plan else None,scope=scope))
    save(out/'progress.json',dict(stage='paired_native_evaluation',episodes=episodes*len(variants)))
    pool(repo,pwsh,plans,out/'pool',out/'evaluation.log')
    manifests=sorted((out/(a.name+'-before')/'capture').glob('case-*/s-*/manifest.json'));assert len(manifests)==episodes
    first=read(manifests[0])
    run([python,repo/'scripts/verify_combat_evaluation_members.py','--root',out,'--source-fingerprint',first['source_fingerprint'],'--native-fingerprint',first['native_source_fingerprint']],out/'verify.log')
    run([python,repo/'scripts/report_combat_architecture_evaluation.py','--root',out,'--member-proof',out/'recovery/verified-members.json'],out/'report.log')
    run([python,repo/'scripts/report_combat_evaluation_strata.py','--root',out],out/'strata.log')
    run([python,repo/'scripts/report_combat_selected_target_aim.py','--root',out],out/'selected-target.log')
    run([python,repo/'scripts/report_combat_blaster_hits.py','--root',out],out/'blaster-hits.log')
    run([python,repo/'scripts/report_combat_aim_modes.py','--root',out],out/'aim-modes.log')
    save(out/'progress.json',dict(stage='complete',quality_report_sha256=sha(out/'quality-report.json')))

if __name__=='__main__':main()

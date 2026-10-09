"""Paired native validation of coarse/fine aim, with 16 independent slots."""
import argparse,pathlib,os
from process_combat_architecture_pool import read,save,sha,run
from run_combat_target_refresh import compile_plan,pool

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-root',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args()
    repo=pathlib.Path(__file__).resolve().parents[1];out=a.out.resolve();out.mkdir(exist_ok=False)
    os.environ['GOCACHE']=str(repo/'workspace/build/go-cache');os.environ['GOTOOLCHAIN']='auto'
    python=pathlib.Path(__import__('sys').executable);pwsh=repo/'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe';compiler=repo/'workspace/build/q2episode-precision-v1.exe'
    reference=read(repo/'workspace/artifacts/target-refresh-v2-20261010/evaluation/protocol.json')
    legacy=next(e for e in reference['evaluations'] if e['model']=='firebc');template=read(legacy['plan']);registry=template['registry_path'];families=reference['families']
    plans=[];entries=[]
    for name,label,source in [('precision','before',a.model_root/'before-weights.json'),('precision','after',a.model_root/'weights.json'),('firebc','baseline',pathlib.Path(template['model_path'])),('rules','baseline',None)]:
        weight=None
        if source:
            frozen=read(source);frozen.update(deterministic=True,sampling_seed=0)
            weight=out/(name+'-'+label+'-weights.json');save(weight,frozen)
        branch=out/(name+'-'+label)
        plan=compile_plan(compiler,registry,repo,weight,branch,'validation',families,28);plans.append(plan)
        entries.append(dict(model=name,label=label,root=str(branch/'capture'),plan=str(plan),plan_sha256=sha(plan),source_weights_sha256=sha(source) if source else None,deterministic_weights_sha256=sha(weight) if weight else None))
    save(out/'protocol.json',dict(version='combat_precision_validation_v1',evaluations=entries,families=families,total_episodes=64,episodes_per_model=16,slots=16,timescale=2,comparison_reference='firebc-baseline',scope='Fine/mode rows only; old45/encoder frozen. Same validation28 conditions reused for development. Before81 deterministic coarse reproduces strong45 parent. Blaster geometric query bootstrap; machinegun labels unknown. Final test deferred.'))
    save(out/'progress.json',dict(stage='paired_native_evaluation',episodes=64))
    pool(repo,pwsh,plans,out/'pool',out/'evaluation.log')
    first=read(out/'precision-before/capture/case-0-learned/s-700028/manifest.json')
    run([python,repo/'scripts/verify_combat_evaluation_members.py','--root',out,'--source-fingerprint',first['source_fingerprint'],'--native-fingerprint',first['native_source_fingerprint']],out/'verify.log')
    run([python,repo/'scripts/report_combat_architecture_evaluation.py','--root',out,'--member-proof',out/'recovery/verified-members.json'],out/'report.log')
    run([python,repo/'scripts/report_combat_selected_target_aim.py','--root',out],out/'selected-target.log')
    save(out/'progress.json',dict(stage='complete',quality_report_sha256=sha(out/'quality-report.json')))

if __name__=='__main__':main()

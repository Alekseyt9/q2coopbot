"""Fresh registered own-policy corpus for native conditional aim labels."""
import argparse,pathlib,sys,os
from process_combat_architecture_pool import read,save,sha,run
from run_combat_target_refresh import compile_plan,pool

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=pathlib.Path,required=True);ap.add_argument('--model',type=pathlib.Path,required=True);a=ap.parse_args()
    repo=pathlib.Path(__file__).resolve().parents[1];out=a.out.resolve();model=a.model.resolve();assert not out.exists();out.mkdir(parents=True)
    try:
        template=read(repo/'workspace/artifacts/first-life-update4-capture-20261010/control/plan.json')
        families=[t['episode']['id'] for t in template['tasks']];assert len(families)==20
        compiler=repo/'workspace/build/q2episode-spatial-v2.exe';plans=[];entries=[]
        for split,offset,chosen in [('train',136,families),('validation',28,families[:4])]:
            folder=out/split;plan=compile_plan(compiler,template['registry_path'],repo,model,folder,split,chosen,offset)
            plans.append(plan);entries.append(dict(model='parent3',label=split,root=str(folder/'capture'),plan=str(plan),plan_sha256=sha(plan),source_weights_sha256=sha(model),deterministic_weights_sha256=sha(model)))
        save(out/'protocol.json',dict(version='native_conditional_machinegun_corpus_v1',evaluations=entries,families=families,total_episodes=96,slots=16,timescale=2,train_offset=136,validation_offset=28,split_scope='80 fresh own-policy train136..139 plus16 reused development validation28..31; validation never enters training. Final test deferred.'))
        save(out/'progress.json',dict(stage='capturing',episodes=96))
        pool(repo,repo/'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe',plans,out/'pool',out/'capture.log')
        manifest=read(next((out/'train/capture').glob('case-*/s-*/manifest.json')))
        run([sys.executable,repo/'scripts/verify_combat_evaluation_members.py','--root',out,'--source-fingerprint',manifest['source_fingerprint'],'--native-fingerprint',manifest['native_source_fingerprint']],out/'verify.log')
        members=[]
        for entry in entries:
            for path in pathlib.Path(entry['root']).glob('case-*/s-*/report.json'):
                result=read(path)['results'][0];steps=pathlib.Path(result['root'])/'dataset/steps.jsonl'
                members.append(dict(seed=result['seed'],split=entry['label'],steps=str(steps),steps_sha256=sha(steps),server=str(pathlib.Path(result['root'])/'server.log'),server_sha256=sha(pathlib.Path(result['root'])/'server.log'),report=str(path),report_sha256=sha(path)))
        assert len(members)==96 and len({m['seed'] for m in members})==96
        save(out/'corpus-spec.json',dict(members=members,scope='Frozen first-life native matched inputs; server events reserved for offline labels. Not PPO reward reuse.'))
        run([repo/'workspace/build/q2target-data-spatial-v2.exe','--spec',out/'corpus-spec.json','--out',out/'client-queries','--center-muzzle','--post-move-labels'],out/'feature-export.log')
        save(out/'progress.json',dict(stage='capture_and_client_features_complete',episodes=96,corpus_spec_sha256=sha(out/'corpus-spec.json')))
    except Exception as error:
        save(out/'progress.json',dict(stage='failed',error=str(error)));raise

if __name__=='__main__':main()

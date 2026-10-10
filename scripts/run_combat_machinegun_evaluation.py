"""Paired common-corpus coarse/fine pilot against frozen priors and controls."""
import argparse,pathlib,sys
from process_combat_architecture_pool import read,save,sha,run
from run_combat_target_refresh import compile_plan,pool

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',type=pathlib.Path,required=True);ap.add_argument('--training',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True)
    ap.add_argument('--full-registry',action='store_true');ap.add_argument('--after-only',action='store_true');a=ap.parse_args()
    repo=pathlib.Path(__file__).resolve().parents[1];out=a.out.resolve();assert not out.exists();training=read(a.training/'report.json');assert training['state']=='complete' and training['device']=='cuda';out.mkdir(parents=True)
    try:
        template=read(repo/'workspace/artifacts/first-life-update4-capture-20261010/control/plan.json');families=[t['episode']['id'] for t in (template['tasks'] if a.full_registry else template['tasks'][:4])]
        entries=[];plans=[];variants=[]
        for entry in training['models']:
            name=entry['model']
            for label,source,digest in [('before',a.models/name/'weights.json',entry['parent_sha256']),('after',a.training/name/'weights.json',entry['weights_sha256'])]:
                if a.after_only and label=='before' and name!='parent3':continue
                assert sha(source)==digest;variants.append((name,label,source))
        prior=read(repo/'workspace/artifacts/first-life-update4-eval-v3-20261010/protocol.json');firebc=next(e for e in prior['evaluations'] if e['model']=='firebc');source=pathlib.Path(read(firebc['plan'])['model_path']);assert sha(source)==firebc['deterministic_weights_sha256']
        variants.extend([('firebc','baseline',source),('rules','baseline',None)])
        for name,label,source in variants:
            folder=out/(name+'-'+label);weights=None
            if source:
                weights=out/(name+'-'+label+'-weights.json');model=read(source);model['deterministic']=True;save(weights,model)
            plan=compile_plan(repo/'workspace/build/q2episode-spatial-v2.exe',template['registry_path'],repo,weights,folder,'validation',families,24)
            plans.append(plan);entries.append(dict(model=name,label=label,root=str(folder/'capture'),plan=str(plan),plan_sha256=sha(plan),source_weights_sha256=sha(source) if source else None,deterministic_weights_sha256=sha(weights) if weights else None))
        episodes=4*len(families)
        scope=('All nine trained candidates, retained PPO before control and FireBC/rules on all20 registered families, including base1/base2 campaign geometry. ' if a.full_registry and a.after_only else 'Eight architectures and retained PPO before/after equal CUDA common-corpus bootstrap, FireBC/rules controls. ')
        scope+='Reused development validation24..27, identical reward and first-life stops. Original architecture histories differ; no final-test superiority or PPO continuation claim.'
        save(out/'protocol.json',dict(version='common_corpus_machinegun_full_validation_v1' if a.full_registry else 'common_corpus_machinegun_pilot_v1',evaluations=entries,families=families,episodes_per_model=episodes,total_episodes=len(entries)*episodes,slots=16,timescale=2,comparison_reference='rules-baseline',training_report_sha256=sha(a.training/'report.json'),scope=scope))
        save(out/'progress.json',dict(stage='evaluating',episodes=len(entries)*episodes))
        pool(repo,repo/'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe',plans,out/'pool',out/'evaluation.log')
        manifest=read(next(pathlib.Path(entries[0]['root']).glob('case-*/s-*/manifest.json')))
        run([sys.executable,repo/'scripts/verify_combat_evaluation_members.py','--root',out,'--source-fingerprint',manifest['source_fingerprint'],'--native-fingerprint',manifest['native_source_fingerprint']],out/'verify.log')
        run([sys.executable,repo/'scripts/report_combat_architecture_evaluation.py','--root',out,'--member-proof',out/'recovery/verified-members.json'],out/'quality.log')
        save(out/'progress.json',dict(stage='quality_complete',episodes=len(entries)*episodes,quality_report_sha256=sha(out/'quality-report.json'),promotion=None))
    except Exception as error:
        save(out/'progress.json',dict(stage='failed',error=str(error)));raise

if __name__=='__main__':main()

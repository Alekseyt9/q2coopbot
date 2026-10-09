"""Freeze fresh spatial own-policy plans for the existing 16-slot CUDA PPO path."""
import argparse,pathlib,os
from process_combat_architecture_pool import read,save,sha,run
from run_combat_target_refresh import compile_plan

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=pathlib.Path,required=True);ap.add_argument('--seed-offset',type=int,default=108);a=ap.parse_args()
    repo=pathlib.Path(__file__).resolve().parents[1];out=a.out.resolve();assert a.seed_offset>=108 and not out.exists()
    os.environ['GOCACHE']=str(repo/'workspace/build/go-cache');os.environ['GOTOOLCHAIN']='auto'
    template=read(repo/'workspace/artifacts/target-refresh-v2-20261010/train/plan.json')
    families=[t['episode']['id'] for t in template['tasks']];assert len(families)==len(set(families))==20
    prior_seeds={m['seed'] for m in read(repo/'workspace/artifacts/target-refresh-v2-20261010/corpus-spec.json')['members']}
    out.mkdir();models=[];plans=[]
    for name,parent in [('instant','spatial-muzzle-v1-20261010'),('postmove','spatial-postmove-v1-20261010')]:
        source=repo/'workspace/artifacts'/parent;weights=read(source/'weights.json');report=read(source/'report.json')
        assert report['device']=='cuda' and report['weights_sha256']==sha(source/'weights.json') and weights.get('spatial_aim')
        checkpoint_audit=read(source/'checkpoint-cuda-audit.json');assert checkpoint_audit['state_exact'] and checkpoint_audit['weights_sha256']==sha(source/'weights.json')
        weights.update(deterministic=False,sampling_seed=20261010);frozen=out/(name+'-weights.json');save(frozen,weights)
        plan=compile_plan(repo/'workspace/build/q2episode-precision-v1.exe',template['registry_path'],repo,frozen,out/name,'train',families,a.seed_offset)
        compiled=read(plan);seeds=[seed for t in compiled['tasks'] for seed in t['seeds']]
        assert len(seeds)==len(set(seeds))==80 and not set(seeds)&prior_seeds
        architecture='attention' if weights.get('attention') else 'gru' if weights.get('memory') else 'entity' if weights.get('entity_attention') else 'mlp'
        models.append(dict(id=name,architecture=dict(id='spatial-'+architecture,architecture=architecture),model=str(frozen),plan=str(plan),capture_root=str(out/name/'capture'),parent_sha256=sha(source/'weights.json'),parent_report_sha256=sha(source/'report.json')));plans.append(str(plan))
    save(out/'models.json',models);save(out/'plans.json',plans)
    config=read(repo/'scripts/scenarios/combat-ppo-recoil-v5.json');config.update(actor_lr=.00003,target_kl=.005,seed=20261010)
    save(out/'config.json',config)
    pwsh=repo/'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe'
    driver=out/'dry-run.ps1';driver.write_text("$ErrorActionPreference='Stop'\n$plans=@(Get-Content -LiteralPath $args[0] -Raw|ConvertFrom-Json)\n& $args[1] -Plans $plans -MaxInstances 16 -Port 35500 -OutputRoot $args[2] -DryRun\n",encoding='utf-8')
    run([pwsh,'-NoProfile','-File',driver,out/'plans.json',repo/'scripts/run_registered_combat_episode_pool.ps1',out/'pool'],out/'dry-run.log')
    save(out/'preparation.json',dict(state='prepared',episodes=160,episodes_per_model=80,slots=16,timescale=2,training_device='cuda',train_seed_offset=a.seed_offset,models_sha256=sha(out/'models.json'),config_sha256=sha(out/'config.json'),parent_corpus_seeds_disjoint=True,scope='Prepared stochastic own-policy captures, not executed. Existing pool and process_combat_architecture_pool CUDA trainers. Fresh PPO optimizer: BC checkpoint is not a PPO optimizer. Retention/bank weights disabled for target-aware policy; old anchor/bank only pinned as legacy processing lineage. PPO trains whole actor/value/std including spatial branch. Final test deferred.'))
    print(str(out/'preparation.json'),flush=True)

if __name__=='__main__':main()

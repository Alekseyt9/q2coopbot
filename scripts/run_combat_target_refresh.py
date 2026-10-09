"""Fresh FireBC visitation -> compressed corrective corpus -> CUDA -> paired native."""
import argparse,json,pathlib,os,subprocess,copy,shutil
from process_combat_architecture_pool import read,save,sha,run
from combat_target_head import migrate
from prepare_combat_target_heads import build,forward
from train_combat_bc import torch

def compile_plan(compiler,registry,root,model,folder,split,families,offset):
    folder.mkdir();plan=folder/'plan.json'
    command=[compiler,'--registry',registry,'--episodes',','.join(families),'--split',split,
        '--mode','learned' if model else 'rules','--count',4,'--seed-offset',offset,
        '--root',root,'--out',plan,'--artifacts',folder/'capture']
    if model:command+=['--model',model]
    run(command,folder/'compile.log');return plan

def pool(repo,pwsh,plans,out,log):
    driver=log.with_suffix('.ps1')
    # JSON argument file keeps paths out of generated shell expressions.
    listpath=driver.with_suffix('.json');save(listpath,[str(p) for p in plans])
    driver.write_text("$ErrorActionPreference='Stop'\n$plans=@(Get-Content -LiteralPath $args[0] -Raw|ConvertFrom-Json)\n& $args[1] -Plans $plans -MaxInstances 16 -Port 35500 -OutputRoot $args[2]\n",encoding='utf-8')
    run([pwsh,'-NoProfile','-File',driver,listpath,repo/'scripts/run_registered_combat_episode_pool.ps1',out],log)
    report=read(out/'report.json');assert report['state']=='complete' and report['source_unchanged'] and not any(j['error'] for j in report['jobs'])
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args()
    repo=pathlib.Path(__file__).resolve().parents[1];out=a.out.resolve();assert not out.exists() and torch.cuda.is_available();out.mkdir()
    os.environ['GOCACHE']=str(repo/'workspace/build/go-cache');os.environ['GOTOOLCHAIN']='auto'
    python=pathlib.Path(__import__('sys').executable);pwsh=repo/'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe';compiler=repo/'workspace/build/q2episode-target-v2.exe';exporter=repo/'workspace/build/q2target-data-v1.exe'
    # The strong, separately measured FireBC baseline; no rejected m0..m7 prior.
    controls=read(repo/'workspace/artifacts/coupling-eval-v1-20261009/protocol.json');control=next(e for e in controls['evaluations'] if e['model']=='firebc');template=read(control['plan']);source=pathlib.Path(template['model_path'])
    assert sha(source)==control['deterministic_weights_sha256'];old=read(source);new=migrate(old);new['deterministic']=False;new['sampling_seed']=20261010
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    torch.manual_seed(20261010);x=torch.randn(2,8,845,device='cuda')*.1;z=torch.cat((x,torch.zeros(2,8,9,device='cuda')),-1);z[:,:,845]=1
    with torch.no_grad():
        errors={name:float((forward(build(old,name),x)-forward(build(new,name),z)[...,:len(old[name][-1]['bias'])]).abs().max()) for name in ('actor','value')}
    assert max(errors.values())<3e-5
    models=out/'models';models.mkdir();folder=models/'firebc';folder.mkdir();weights=folder/'weights.json';save(weights,new)
    save(out/'migration-audit.json',dict(device='cuda',parent_sha256=sha(source),weights_sha256=sha(weights),maximum_error=errors))
    registry=template['registry_path'];families=[t['episode']['id'] for t in template['tasks']];assert len(families)==20
    # New training seeds beyond the old FireBC rounds; validation never joins train.
    train=compile_plan(compiler,registry,repo,weights,out/'train','train',families,104)
    valid=compile_plan(compiler,registry,repo,weights,out/'validation','validation',families[:4],32)
    save(out/'protocol.json',dict(version='combat_target_refresh_v1',parent_sha256=sha(source),migration_sha256=sha(weights),train_seed_offset=104,validation_seed_offset=32,train_episodes=80,validation_episodes=16,slots=16,timescale=2,training_device='cuda',intent_columns=True,final_test_deferred=True))
    save(out/'progress.json',dict(stage='capturing_own_policy',episodes=96))
    closed=pool(repo,pwsh,[train,valid],out/'collect-pool',out/'collect.log')
    members=[];fingerprints=set();native=set()
    for job in closed['jobs']:
        path=pathlib.Path(job['root']);report=read(path/'report.json');manifest=read(path/'manifest.json');result=report['results'][0]
        assert report['capture_complete'] and report['provenance_valid'] and len(report['results'])==1
        assert result['seed']==job['seed'] and all(result[k] for k in ('seed_confirmed','capture_valid','dispatch_valid','frame_budget_valid'))
        assert manifest['model_weights_sha256'].lower()==sha(weights);fingerprints.add(manifest['source_fingerprint']);native.add(manifest['native_source_fingerprint'])
        trace=pathlib.Path(result['root']);dataset=trace/'dataset';proof=read(dataset/'report.json');assert proof['synchronous_confirmed'] and proof['observed_reset_confirmed'] and proof['command_proof']['accepted']
        steps=dataset/'steps.jsonl';members.append(dict(seed=job['seed'],split='train' if job['plan_index']==0 else 'validation',steps=str(steps),steps_sha256=sha(steps),report=str(path/'report.json'),report_sha256=sha(path/'report.json'),manifest_sha256=sha(path/'manifest.json'),trace_sha256=sha(trace/'bot.jsonl'),server_sha256=sha(trace/'server.log')))
    assert len(fingerprints)==len(native)==1 and len(members)==96
    spec=out/'corpus-spec.json';save(spec,dict(members=members,scope='Frozen closed native command proof; inputs remain client observations only'))
    save(out/'progress.json',dict(stage='exporting_corrective_queries',episodes=96))
    data=out/'data';run([exporter,'--spec',spec,'--out',data],out/'export.log')
    save(out/'progress.json',dict(stage='cuda_training',episodes=96))
    run([python,repo/'scripts/train_combat_target_bc.py','--models',models,'--names','firebc','--data',data,'--epochs',50,'--train-intent'],out/'train.log')
    training=read(models/'target-bc-report.json');assert training['state']=='complete' and training['device']=='cuda'
    candidate=folder/'target-bc/weights.json';report=read(folder/'target-bc/report.json');assert report['weights_sha256']==sha(candidate) and report['old_feature_columns_and_common_rows_preserved'] and report['intent_columns_trained']
    checkpoint=torch.load(folder/'target-bc/checkpoint.pt',map_location='cuda',weights_only=True);actor=build(read(candidate),'actor')
    for key,tensor in actor.state_dict().items():assert torch.equal(tensor,checkpoint['actor'][key]) and torch.isfinite(tensor).all()
    evaluation=out/'evaluation';evaluation.mkdir();plans=[];entries=[]
    for name,label,sourcepath in [('firebc-target','before',weights),('firebc-target','after',candidate),('firebc','baseline',source),('rules','baseline',None)]:
        branch=evaluation/(name+'-'+label);weight=None
        if sourcepath:
            frozen=read(sourcepath);frozen.update(deterministic=True,sampling_seed=0);weight=evaluation/(name+'-'+label+'-weights.json');save(weight,frozen)
        plan=compile_plan(compiler,registry,repo,weight,branch,'validation',families[:4],32);plans.append(plan)
        entries.append(dict(model=name,label=label,root=str(branch/'capture'),plan=str(plan),plan_sha256=sha(plan),source_weights_sha256=sha(sourcepath) if sourcepath else None,deterministic_weights_sha256=sha(weight) if weight else None))
    save(evaluation/'protocol.json',dict(version='combat_target_refresh_validation_v1',evaluations=entries,families=families[:4],total_episodes=64,episodes_per_model=16,slots=16,timescale=2,comparison_reference='firebc-baseline',scope='Strong FireBC V6 control vs migrated target head before/after fresh own-policy corrective BC with learned intent columns. Validation32 reused by BC; not final test. Shared weight rows frozen, intent input can alter shared outputs despite retention penalty.'))
    save(out/'progress.json',dict(stage='paired_native_evaluation',episodes=64))
    pool(repo,pwsh,plans,evaluation/'pool',out/'evaluation.log')
    first=read(evaluation/'firebc-target-before/capture/case-0-learned/s-700032/manifest.json')
    run([python,repo/'scripts/verify_combat_evaluation_members.py','--root',evaluation,'--source-fingerprint',first['source_fingerprint'],'--native-fingerprint',first['native_source_fingerprint']],out/'verify.log')
    run([python,repo/'scripts/report_combat_architecture_evaluation.py','--root',evaluation,'--member-proof',evaluation/'recovery/verified-members.json'],out/'report.log')
    run([python,repo/'scripts/report_combat_selected_target_aim.py','--root',evaluation],out/'selected-target.log')
    save(out/'progress.json',dict(stage='complete',training_report_sha256=sha(models/'target-bc-report.json'),quality_report_sha256=sha(evaluation/'quality-report.json')))

if __name__=='__main__':main()

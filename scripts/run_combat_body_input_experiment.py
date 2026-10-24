"""Four matched-budget body/ablation PPO rounds sharing one 16-slot queue."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from evaluate_registered_combat_models import read,save,sha
from combat_execution_contract import LEGACY
from combat_body_training_scope import SCOPE
from run_combat_diverse_registry_ppo import process_alive


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','spec','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();repo=Path(__file__).resolve().parents[1];root=a.out.resolve()
    protocol=read(a.protocol);assert protocol['version']==SCOPE and protocol['rounds']==4 and protocol['cases_per_update']==24
    configs={str(p.resolve()):sha(p) for p in (a.protocol,a.spec)}
    root.mkdir(exist_ok=True)
    if a.resume:
        run=read(root/'run.json');state=read(root/'progress.json')
        assert not process_alive(state.get('runner_pid')) and not process_alive(state.get('child_pid'))
        assert run['configs']==configs
    else:
        assert not (root/'run.json').exists()
        paths=set()
        for directory,ext in ((repo/'scripts',('.py','.ps1')),(repo/'internal',('.go',)),(repo/'cmd',('.go',)),(repo.parent/'yquake2/src',('.c','.h'))):
            paths.update(p.resolve() for p in directory.rglob('*') if p.is_file() and p.suffix in ext)
        run=dict(configs=configs,source_files={str(p):sha(p) for p in sorted(paths)},max_instances=16,
            native_train_budget=192,cuda_train_budget=192,native_eval_budget=40,cuda_eval_budget=32,promotion=False)
        save(root/'run.json',run)
    env=dict(os.environ,GOCACHE=str(repo/'workspace/build/gocache'),GOTOOLCHAIN='auto')

    def unchanged():
        assert all(sha(p)==h for p,h in configs.items()),'Frozen protocol changed'
        assert all(sha(p)==h for p,h in run['source_files'].items()),'Source changed'

    def closed(target,filename):
        receipt=read(target/filename)
        assert receipt['state']==('prepared' if filename=='prepared.json' else 'complete'),str(target)
        if 'report_sha256' in receipt:assert sha(target/'report.json')==receipt['report_sha256']
        return receipt

    def step(label,command,target,filename='complete.json'):
        unchanged()
        if (target/filename).exists():
            closed(target,filename)
            save(root/'progress.json',dict(stage='reuse-'+label,runner_pid=os.getpid(),child_pid=None))
            return
        assert not target.exists(),'Partial output preserved: '+str(target)
        with (root/(label+'.log')).open('w',encoding='utf-8') as log:
            child=subprocess.Popen(list(map(str,command)),cwd=repo,env=env,stdout=log,stderr=log,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            save(root/'progress.json',dict(stage=label,runner_pid=os.getpid(),child_pid=child.pid))
            result=child.wait()
        assert result==0,str(root/(label+'.log'))
        closed(target,filename);unchanged()

    py=lambda script,*args:[sys.executable,'scripts/'+script,*args]
    arms=[('ablated',LEGACY),('observed',LEGACY)]
    sources={};history=list(protocol['historical_plans']);rounds=[]
    try:
        for arm,contract in arms:
            target=root/'initial'/arm
            step('initial-'+arm,py('initialize_combat_body_training_cuda.py','--source',protocol['initial_sources'][arm]['path'],
                '--corpus',protocol['parity_corpus'],'--protocol',a.protocol.resolve(),'--mode',arm,'--out',target),target)
            report=read(target/'report.json')
            assert report['device']=='cuda' and report['source_weights_sha256']==protocol['initial_sources'][arm]['weights_sha256']
            assert all(report[k] for k in ('source_training_tensors_exact','source_optimizers_exact','source_rng_exact','checkpoint_roundtrip_exact'))
            assert report['protocol_sha256']==sha(a.protocol) and report['shared_input_bias'] and report['disposable_resume_adam_step_exact']
            assert report['feature_version']==protocol['features_by_arm'][arm]
            assert report['disposable_gradients']['shared_bias']>0
            sources[arm]=target
        shell=shutil.which('pwsh');assert shell
        literal=lambda p:"'"+str(p).replace("'","''")+"'"
        for number in range(1,5):
            folder=root/f'round-{number:02d}';folder.mkdir(exist_ok=True)
            plans=[]
            for index,(arm,contract) in enumerate(arms):
                branch=folder/arm;branch.mkdir(exist_ok=True)
                history_path=branch/'history.json';value=dict(historical_plans=history[:])
                if history_path.exists():assert read(history_path)==value
                else:save(history_path,value)
                corpus=branch/'corpus'
                step(f'r{number:02d}-prepare-'+arm,py('collect_combat_registry_curriculum.py','--source',sources[arm],
                    '--registry',protocol['registry'],'--episodes',','.join(protocol['episodes']),'--execution-contract',contract,
                    '--history-spec',history_path,'--train-count','1','--train-offset',str(protocol['train_offset']+2*(number-1)+index),
                    '--allow-groups','--ppo-ready','--train-only','--prepare-only','--out',corpus),corpus,'prepared.json')
                plan=corpus/'train-plan.json';plans.append(plan);history.append(str(plan))
            pool=folder/'pool'
            command='& '+literal(repo/'scripts/run_registered_combat_pool.ps1')+' -Plans @('+','.join(map(literal,plans))+') -MaxInstances 16 -Port '+str(protocol['port'])+' -OutputRoot '+literal(pool)
            step(f'r{number:02d}-native-pool',[shell,'-NoProfile','-Command',command],pool,'report.json')
            native=read(pool/'report.json');assert native['source_unchanged'] and native['usable_captures']==len(native['jobs'])==48
            for index,(arm,contract) in enumerate(arms):
                branch=folder/arm;corpus=branch/'corpus'
                # Prepared output exists deliberately. Only finish a fresh
                # processing subset; never retry a partially processed one.
                if not (corpus/'complete.json').exists():
                    assert not (corpus/'processing-train').exists() and not (corpus/'pool').exists()
                    with (root/f'r{number:02d}-finalize-{arm}.log').open('w',encoding='utf-8') as log:
                        child=subprocess.Popen(list(map(str,py('finalize_combat_execution_corpus.py','--corpus',corpus,'--pool',pool,'--plan-index',index))),
                            cwd=repo,env=env,stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                        save(root/'progress.json',dict(stage=f'r{number:02d}-finalize-'+arm,runner_pid=os.getpid(),child_pid=child.pid))
                        result=child.wait()
                    assert result==0,str(root/f'r{number:02d}-finalize-{arm}.log')
                c=closed(corpus,'complete.json');assert c['native_cases']==c['cuda_cases']==24 and c['selected_plan_index']==index
                unchanged()
                native_credit,advantage,update=(branch/n for n in ('native-credit','advantage-credit','update'))
                step(f'r{number:02d}-native-credit-'+arm,py('audit_combat_blocked_ground_native_credit.py','--corpus',corpus,'--out',native_credit),native_credit)
                step(f'r{number:02d}-advantage-'+arm,py('audit_combat_advantage_credit.py','--corpus',corpus,'--native-audit',native_credit,'--out',advantage),advantage)
                step(f'r{number:02d}-train-'+arm,py('train_combat_body_input_cuda.py','--source',sources[arm],'--corpus',corpus,
                    '--native-credit',native_credit,'--advantage-credit',advantage,'--protocol',a.protocol.resolve(),'--out',update),update)
                trained=read(update/'report.json')
                assert trained['scope_updates']==number and trained['combat_execution_contract']==contract
                assert trained['feature_version']==protocol['features_by_arm'][arm] and trained['shared_input_bias']
                assert trained['disposable_critic_resume_adam_step_exact']
                assert trained['behavior_sha256']==read(sources[arm]/'complete.json')['weights_sha256']
                assert trained['checkpoint_roundtrip_exact'] and trained['disposable_resume_adam_step_exact']
                rounds.append(dict(round=number,arm=arm,contract=contract,rows=trained['rows'],actor_steps=trained['actor_steps'],
                    critic_steps=trained['critic_steps'],weights_sha256=trained['weights_sha256']))
                sources[arm]=update
            save(root/'rounds.json',dict(rounds=rounds))
        evaluation=root/'evaluation'
        step('evaluation',py('evaluate_registered_combat_models.py','--spec',a.spec.resolve(),'--out',evaluation),evaluation)
        quality=root/'native-quality'
        step('native-quality',py('audit_registered_native_action_quality.py','--evaluation',evaluation,'--out',quality),quality)
        result=root/'result'
        step('result',py('report_combat_body_input_screen.py','--workflow',root,'--protocol',a.protocol.resolve(),
            '--spec',a.spec.resolve(),'--out',result),result)
        unchanged()
        save(root/'complete.json',dict(state='complete',version='combat_body_input_experiment_v2',
            protocol_sha256=sha(a.protocol),spec_sha256=sha(a.spec),native_train_cases=192,cuda_train_cases=192,
            native_eval_cases=40,cuda_eval_cases=32,result_seal_sha256=sha(result/'complete.json'),
            source_unchanged=True,promotion=False))
        save(root/'progress.json',dict(stage='complete',runner_pid=os.getpid(),child_pid=None))
    except BaseException as error:
        save(root/'progress.json',dict(stage='failed',runner_pid=os.getpid(),error=str(error)))
        raise


if __name__=='__main__':main()

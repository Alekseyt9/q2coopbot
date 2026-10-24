"""Four matched-budget typed/neutral projectile PPO rounds sharing one 16-slot queue."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from evaluate_registered_combat_models import read,save,sha
from combat_execution_contract import LEGACY
from combat_projectile_training_scope import SCOPE
from run_combat_diverse_registry_ppo import process_alive


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','spec','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--offline-recovery',type=Path)
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
    recovery_digest=None
    if a.offline_recovery:
        assert a.resume
        recovery_digest=sha(a.offline_recovery)
        recovery=read(a.offline_recovery)
        assert recovery['version']=='combat_enemy_projectile_offline_class_recovery_v3'
        assert recovery['original_run_sha256']==sha(root/'run.json')
        proof=read(recovery['pre_fit_proof'])
        assert sha(recovery['pre_fit_proof'])==recovery['pre_fit_proof_sha256']
        assert proof['original_run_sha256']==sha(root/'run.json') and proof['no_update_directories']
        assert proof['unassigned_events']==43 and proof['class_mismatch_events']==42
        assert len(proof['verified_post_learning_goal_events'])==1
        failure=read(recovery['failure_progress'])
        assert sha(recovery['failure_progress'])==recovery['failure_progress_sha256']
        assert failure['stage']=='failed' and failure['error']=='Pre-fit exposure failed; fixed data preserved'
        old_exposure=read(recovery['original_exposure_report'])
        assert sha(recovery['original_exposure_report'])==recovery['original_exposure_report_sha256']
        assert not old_exposure['checks']['no_infighting'] and old_exposure['counts']['native_infighting_events']==1
        assert all(value for key,value in old_exposure['checks'].items() if key!='no_infighting')
        expected={str((repo/p).resolve()) for p in ('scripts/run_combat_enemy_projectile_training.py',
            'scripts/audit_combat_blocked_ground_native_credit.py','internal/learningenv/target_commitment_reward.go',
            'internal/learningenv/target_commitment_reward_test.go','cmd/q2ppo-data/main.go',
            'scripts/process_registered_combat_pool_cuda.py','scripts/finalize_combat_execution_corpus.py',
            'scripts/audit_combat_projectile_training_exposure.py')}
        assert set(recovery['replacements'])==expected
        run['source_files']=dict(run['source_files'])
        for path,change in recovery['replacements'].items():
            assert run['source_files'][path]==change['before_sha256']==sha(change['before_snapshot'])
            assert sha(path)==change['after_sha256']
            run['source_files'][path]=change['after_sha256']
        for path,digest in recovery['additional_sources'].items():
            assert path not in run['source_files'] and sha(path)==digest
            run['source_files'][path]=digest
        assert sha(recovery['archive_manifest'])==recovery['archive_manifest_sha256']
        archive=read(recovery['archive_manifest'])
        for path,digest in archive['retained_files'].items():assert sha(path)==digest,path
        pool=read(root/'round-01/pool/report.json')
        assert sha(root/'round-01/pool/report.json')==recovery['pool_report_sha256']
        assert pool['source_unchanged'] and pool['usable_captures']==len(pool['jobs'])==48
        assert sha(recovery['replay_exporter'])==recovery['replay_exporter_sha256']
        marker=root/('offline-recovery-'+recovery_digest[:12]+'-applied.json')
        if marker.exists():assert read(marker)['recovery_sha256']==recovery_digest
        else:
            assert not any(root.glob('round-*/**/update')),'Cannot first admit an offline repair after fitting'
            save(marker,dict(recovery_sha256=recovery_digest,pre_fit=True,new_native_cases=0))
    env=dict(os.environ,GOCACHE=str(repo/'workspace/build/gocache'),GOTOOLCHAIN='auto')

    def unchanged():
        if recovery_digest is not None:
            assert sha(a.offline_recovery)==recovery_digest
            assert sha(recovery['replay_exporter'])==recovery['replay_exporter_sha256']
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
    arms=[('typed',LEGACY),('neutral',LEGACY)]
    sources={};history=list(protocol['historical_plans']);rounds=[]
    try:
        mechanism=read(protocol['enemy_lifetime_validation'])
        assert sha(protocol['enemy_lifetime_validation'])==protocol['enemy_lifetime_validation_sha256']
        assert mechanism['parity_passed'] and mechanism['counts']['enemy_velocity_known']>0
        parity=read(protocol['parity_audit'])
        assert sha(protocol['parity_audit'])==protocol['parity_report_sha256'] and parity['gate_passed']
        assert parity['native_cases']==parity['cuda_cases']==12 and all(parity['checks'].values())
        for path,digest in parity['inputs'].items():assert sha(path)==digest,path
        pair=read(protocol['pair_proof'])
        assert sha(protocol['pair_proof'])==protocol['pair_proof_sha256']
        assert all(pair[k] for k in ('network_tensors_exact','actor_optimizers_exact','critic_optimizers_exact','rng_exact'))
        for arm,contract in arms:
            initial=protocol['initial_sources'][arm];target=Path(initial['path'])
            seal=closed(target,'complete.json')
            assert sha(target/'complete.json')==initial['seal_sha256']
            for filename,key in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):
                assert sha(target/filename)==seal[key]==initial[key]
            report=read(target/'report.json')
            assert report['device']=='cuda' and report['version']==protocol['version'] and report['scope_updates']==0
            assert report['source_weights_sha256']==protocol['root_source_weights_sha256']
            assert report['protocol_sha256']==protocol['initialization_protocol_sha256']
            assert report['feature_version']==protocol['features_by_arm'][arm]
            assert all(report[k] for k in ('source_training_tensors_exact','inherited_seven_actor_groups_exact',
                'critic_old_moment_prefix_exact','critic_step_counters_preserved','checkpoint_roundtrip_exact',
                'disposable_actor_resume_adam_step_exact','disposable_critic_resume_adam_step_exact','source_rng_exact'))
            assert report['gradients']['projectile']>0 and report['gradients']['shared_bias']>0
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
                        command=py('finalize_combat_execution_corpus.py','--corpus',corpus,'--pool',pool,'--plan-index',index)
                        if recovery_digest is not None and number==1:command+=['--replay-exporter',recovery['replay_exporter']]
                        child=subprocess.Popen(list(map(str,command)),
                            cwd=repo,env=env,stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                        save(root/'progress.json',dict(stage=f'r{number:02d}-finalize-'+arm,runner_pid=os.getpid(),child_pid=child.pid))
                        result=child.wait()
                    assert result==0,str(root/f'r{number:02d}-finalize-{arm}.log')
                c=closed(corpus,'complete.json');assert c['native_cases']==c['cuda_cases']==24 and c['selected_plan_index']==index
                unchanged()
                native_credit,advantage,update=(branch/n for n in ('native-credit','advantage-credit','update'))
                step(f'r{number:02d}-native-credit-'+arm,py('audit_combat_blocked_ground_native_credit.py','--corpus',corpus,'--out',native_credit),native_credit)
                step(f'r{number:02d}-advantage-'+arm,py('audit_combat_advantage_credit.py','--corpus',corpus,'--native-audit',native_credit,'--out',advantage),advantage)
                exposure=branch/'exposure'
                step(f'r{number:02d}-exposure-'+arm,py('audit_combat_projectile_training_exposure.py','--corpus',corpus,
                    '--protocol',a.protocol.resolve(),'--round',number,'--out',exposure),exposure)
                assert read(exposure/'report.json')['gate_passed'],'Pre-fit exposure failed; fixed data preserved'
                step(f'r{number:02d}-train-'+arm,py('train_combat_projectile_types_cuda.py','--source',sources[arm],'--corpus',corpus,
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
        step('result',py('report_combat_enemy_projectile_screen.py','--workflow',root,'--protocol',a.protocol.resolve(),
            '--spec',a.spec.resolve(),'--out',result),result)
        unchanged()
        save(root/'complete.json',dict(state='complete',version='combat_enemy_projectile_training_workflow_v1',
            protocol_sha256=sha(a.protocol),spec_sha256=sha(a.spec),native_train_cases=192,cuda_train_cases=192,
            native_eval_cases=40,cuda_eval_cases=32,result_seal_sha256=sha(result/'complete.json'),
            source_unchanged=True,original_source_unchanged=recovery_digest is None,
            offline_recovery_sha256=recovery_digest,promotion=False))
        save(root/'progress.json',dict(stage='complete',runner_pid=os.getpid(),child_pid=None))
    except BaseException as error:
        save(root/'progress.json',dict(stage='failed',runner_pid=os.getpid(),error=str(error)))
        raise


if __name__=='__main__':main()

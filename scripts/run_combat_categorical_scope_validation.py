"""Validate a prepared categorical checkpoint through fresh continuation and fixed quality."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from evaluate_registered_combat_models import read, save, sha
from run_combat_diverse_registry_ppo import process_alive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'spec', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = args.out.resolve()
    protocol, spec = read(args.protocol), read(args.spec)
    assert protocol['version'] == 'combat_categorical_scope_ppo_v1'
    assert spec['experiment_protocol_sha256'] == sha(args.protocol)
    os.environ['GOCACHE'] = str(repo / 'workspace/build/gocache')
    os.environ['GOTOOLCHAIN'] = 'auto'
    configs = {str(path.resolve()):sha(path) for path in (args.protocol,args.spec)}
    if args.resume:
        run = read(root/'run.json')
        state = read(root/'progress.json')
        assert not process_alive(state.get('runner_pid')) and not process_alive(state.get('child_pid'))
        assert run['configs'] == configs
    else:
        assert root.is_dir() and (root/'update/complete.json').exists() and not (root/'run.json').exists()
        paths = set()
        for directory,suffixes in ((repo/'scripts',('.py','.ps1')),(repo/'internal',('.go',)),(repo/'cmd',('.go',)),(repo.parent/'yquake2/src',('.c','.h'))):
            paths.update(path.resolve() for path in directory.rglob('*') if path.is_file() and path.suffix in suffixes)
        run = dict(configs=configs,source_files={str(path):sha(path) for path in sorted(paths)},
                   new_native_training_cases=24,native_evaluation_cases=32,max_instances=16,timescale=2,
                   quality_endpoint='Prepared first update only; second update verifies fresh on-policy continuation')
        save(root/'run.json',run)

    def unchanged():
        assert all(sha(path)==digest for path,digest in configs.items())
        assert all(sha(path)==digest for path,digest in run['source_files'].items())

    def stage(label,child_pid=None):
        save(root/'progress.json',dict(stage=label,runner_pid=os.getpid(),child_pid=child_pid))

    def closed(directory):
        receipt = read(directory/'complete.json')
        assert receipt['state']=='complete'
        if 'report_sha256' in receipt:
            assert sha(directory/'report.json')==receipt['report_sha256']
        return receipt

    def step(label,command,target):
        unchanged()
        if (target/'complete.json').exists():
            closed(target)
            stage('reuse-'+label)
            return
        assert not target.exists(),'Partial output preserved: '+str(target)
        with (root/(label+'.log')).open('w',encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable,*map(str,command)],cwd=repo,stdout=log,stderr=log,
                                     creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            stage(label,child.pid)
            result = child.wait()
        assert result==0,str(root/(label+'.log'))
        closed(target)
        unchanged()

    try:
        initial = root/'update'
        seal = closed(initial)
        candidate = read(initial/'report.json')
        assert candidate['version']==protocol['version'] and candidate['scope_updates']==1
        assert candidate['behavior_sha256']==protocol['initial_source_weights_sha256']
        assert candidate['rollout_sha256']==protocol['initial_rollout_sha256']
        assert candidate['experiment_protocol_sha256']==sha(args.protocol)
        assert all(candidate[key] for key in ('frozen_coordinates_exact','matched_control_critic_exact','checkpoint_roundtrip_exact','disposable_resume_adam_step_exact','rng_preserved'))
        probe_path=Path(protocol['probe_report'])
        assert sha(probe_path)==protocol['probe_report_sha256']
        probe=read(probe_path)
        assert probe['mechanical_gate_passed'] and probe['device']=='cuda' and not probe['weights_written']
        assert all(candidate[name] for name in ('frozen_motor_exact','gaussian_means_exact','frozen_inherited_decision_optimizer_exact'))
        assert candidate['movement_conditional_kl']==candidate['aim_conditional_kl']==0
        continuation = root/'continuation'
        continuation.mkdir(exist_ok=True)
        history_path = continuation/'history.json'
        history = dict(historical_plans=protocol['historical_plans'])
        if history_path.exists():
            assert read(history_path)==history
        else:
            save(history_path,history)
        corpus = continuation/'corpus'
        step('collect-continuation',['scripts/collect_combat_registry_curriculum.py','--source',initial,'--registry',protocol['registry'],
            '--episodes',','.join(protocol['episodes']),'--history-spec',history_path,'--train-count','1',
            '--train-offset',str(protocol['continuation_train_offset']),'--port','48000','--allow-groups','--ppo-ready','--train-only','--out',corpus],corpus)
        assert closed(corpus)['native_cases']==closed(corpus)['cuda_cases']==24
        native, advantage, update = (continuation/name for name in ('native-credit','advantage-credit','update'))
        step('native-credit',['scripts/audit_combat_blocked_ground_native_credit.py','--corpus',corpus,'--out',native],native)
        step('advantage-credit',['scripts/audit_combat_advantage_credit.py','--corpus',corpus,'--native-audit',native,'--out',advantage],advantage)
        step('train-continuation',['scripts/train_combat_categorical_scope_cuda.py','--source',initial,'--corpus',corpus,
            '--native-credit',native,'--advantage-credit',advantage,'--protocol',args.protocol,'--out',update],update)
        report = read(update/'report.json')
        assert report['scope_updates']==2 and report['behavior_sha256']==seal['weights_sha256']
        assert report['checkpoint_roundtrip_exact'] and report['disposable_resume_adam_step_exact']
        assert sha(initial/'weights.json')==seal['weights_sha256'],'Quality endpoint changed during continuation'
        evaluation = root/'evaluation'
        step('evaluation',['scripts/evaluate_registered_combat_models.py','--spec',args.spec,'--out',evaluation],evaluation)
        assert closed(evaluation)['native_cases']==32 and closed(evaluation)['cuda_cases']==24
        quality, damage = root/'native-quality',root/'damage'
        step('native-quality',['scripts/audit_registered_native_action_quality.py','--evaluation',evaluation,'--out',quality],quality)
        step('damage',['scripts/audit_combat_damage_distribution.py','--evaluation',evaluation,'--out',damage],damage)
        save(root/'complete.json',dict(state='complete',version='combat_categorical_scope_validation_v1',
            native_training_cases=24,cuda_training_cases=24,native_evaluation_cases=32,cuda_evaluation_cases=24,
            reused_prior_train_cases=24,first_update_seal_sha256=sha(initial/'complete.json'),
            continuation_seal_sha256=sha(update/'complete.json'),evaluation_seal_sha256=sha(evaluation/'complete.json'),
            quality_weights_sha256=seal['weights_sha256'],protocol_sha256=sha(args.protocol),promotion=False))
        step('result',['scripts/report_combat_categorical_scope_screen.py','--workflow',root,'--spec',args.spec,'--out',root/'result'],root/'result')
        stage('complete')
    except BaseException as error:
        save(root/'progress.json',dict(stage='failed',runner_pid=os.getpid(),error=str(error)))
        raise


if __name__=='__main__':
    main()

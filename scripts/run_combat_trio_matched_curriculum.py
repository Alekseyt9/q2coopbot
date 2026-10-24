"""Align prior update counts, then compare matched own-policy trio curricula."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from evaluate_registered_combat_models import read,save,sha
from run_combat_diverse_registry_ppo import process_alive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','spec','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1];root=args.out.resolve()
    study,spec=read(args.protocol),read(args.spec)
    assert study['version']=='combat_trio_matched_curriculum_v1'
    assert study['native_training_cases']==80 and study['cases_per_update']==16
    assert spec['experiment_protocol_sha256']==sha(args.protocol)
    os.environ['GOCACHE']=str(repo/'workspace/build/gocache');os.environ['GOTOOLCHAIN']='auto'
    configs={str(path.resolve()):sha(path) for path in (args.protocol,args.spec,Path(study['full_contract']),Path(study['decision_contract']))}
    assert sha(study['full_contract'])==study['full_contract_sha256'] and sha(study['decision_contract'])==study['decision_contract_sha256']
    if args.resume:
        run=read(root/'run.json');status=read(root/'progress.json')
        assert not process_alive(status.get('runner_pid')) and not process_alive(status.get('child_pid'))
        assert run['configs']==configs
    else:
        assert not root.exists();root.mkdir(parents=True)
        paths=set()
        for folder,suffixes in ((repo/'scripts',('.py','.ps1')),(repo/'internal',('.go',)),(repo/'cmd',('.go',)),(repo.parent/'yquake2/src',('.c','.h'))):
            paths.update(path.resolve() for path in folder.rglob('*') if path.is_file() and path.suffix in suffixes)
        run=dict(configs=configs,source_files={str(path):sha(path) for path in sorted(paths)},native_training_cases=80,
                 native_evaluation_cases=32,max_instances=16,timescale=2,quality_endpoint='Final four-update endpoints only')
        save(root/'run.json',run)

    def stage(label,child_pid=None):save(root/'progress.json',dict(stage=label,runner_pid=os.getpid(),child_pid=child_pid))
    def unchanged():
        assert all(sha(path)==digest for path,digest in configs.items())
        assert all(sha(path)==digest for path,digest in run['source_files'].items())
    def closed(directory):
        receipt=read(directory/'complete.json');assert receipt['state']=='complete'
        if 'report_sha256' in receipt:assert sha(directory/'report.json')==receipt['report_sha256']
        return receipt
    def step(label,command,target):
        unchanged()
        if (target/'complete.json').exists():closed(target);stage('reuse-'+label);return
        assert not target.exists(),'Partial output preserved: '+str(target)
        with (root/(label+'.log')).open('w',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,*map(str,command)],cwd=repo,stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            stage(label,child.pid);code=child.wait()
        assert code==0,str(root/(label+'.log'))
        closed(target);unchanged()

    history=set(study['historical_plans'])
    def phase(label,source,episodes,offset,number,kind,paired_reference=None):
        directory=root/label;directory.mkdir(exist_ok=True)
        history_path=directory/'history.json';expected=dict(historical_plans=sorted(history))
        if history_path.exists():assert read(history_path)==expected
        else:save(history_path,expected)
        seal=closed(source)
        assert sha(source/'weights.json')==seal['weights_sha256'] and sha(source/'checkpoint.pt')==seal['checkpoint_sha256']
        corpus=directory/'corpus'
        command=['scripts/collect_combat_blocked_ground_curriculum.py','--source',source,'--registry',study['registry'],
            '--episodes',','.join(episodes),'--history-spec',history_path,'--train-count','2','--train-offset',str(offset),
            '--port','48000','--allow-groups','--ppo-ready','--train-only','--out',corpus]
        if paired_reference:command+=['--paired-reference',paired_reference]
        step(label+'-collect',command,corpus)
        receipt=closed(corpus);assert receipt['native_cases']==receipt['cuda_cases']==16
        protocol=read(corpus/'protocol.json');assert sha(corpus/'protocol.json')==receipt['protocol_sha256']
        assignment=protocol['assignments'][0];plan=Path(assignment['plan'])
        assert sha(plan)==assignment['plan_sha256'] and read(plan)['model_sha256']==seal['weights_sha256']
        if paired_reference:assert read(corpus/'paired-train-proof.json')['state']=='matched'
        history.add(str(plan.resolve()))
        native,advantage,update=(directory/name for name in ('native-credit','advantage-credit','update'))
        step(label+'-native',['scripts/audit_combat_blocked_ground_native_credit.py','--corpus',corpus,'--out',native],native)
        step(label+'-advantage',['scripts/audit_combat_advantage_credit.py','--corpus',corpus,'--native-audit',native,'--out',advantage],advantage)
        if kind=='full':
            command=['scripts/train_combat_trio_full_ppo_cuda.py','--source',source,'--corpus',corpus,'--credit-audit',native,
                '--advantage-audit',advantage,'--protocol',study['full_contract'],'--out',update,'--arm','G','--branch','cost',
                '--round',str(number),'--boundary-acceptance',study['boundary_acceptance']]
        else:
            command=['scripts/train_combat_decision_scope_cuda.py','--source',source,'--corpus',corpus,'--native-credit',native,
                '--advantage-credit',advantage,'--protocol',study['decision_contract'],'--out',update]
        step(label+'-training',command,update)
        report=read(update/'report.json');assert report['device']=='cuda' and report['behavior_sha256']==seal['weights_sha256']
        if kind=='full':assert report['round']==number and report['checkpoint_optimizer_roundtrip_exact']
        else:assert report['scope_updates']==number and report['checkpoint_roundtrip_exact'] and report['disposable_resume_adam_step_exact']
        return update.resolve(),plan.resolve()

    try:
        reference=Path(study['alignment_reference']);assert sha(reference)==study['alignment_reference_sha256']
        full,_=phase('full-round-01',Path(study['full_source']),study['alignment_episodes'],study['alignment_offset'],1,'full',reference)
        decisions=Path(study['decision_source'])
        migration=read(decisions/'report.json');assert migration['weights_bytes_exact'] and migration['all_neural_optimizer_rng_history_exact']
        for index,offset in enumerate(study['train_offsets'],start=2):
            full,paired_plan=phase(f'full-round-{index:02d}',full,study['episodes'],offset,index,'full')
            decisions,_=phase(f'decision-round-{index+1:02d}',decisions,study['episodes'],offset,index+1,'decisions',paired_plan)
        # Metadata deserialization on CUDA; no CPU neural computation.
        from train_combat_bc import torch,training_devices
        training_devices()
        parent_path=Path(next(v['update'] for v in spec['variants'] if v['label']=='parent'))
        parent_seal=closed(parent_path)
        assert sha(parent_path/'checkpoint.pt')==parent_seal['checkpoint_sha256']
        parent=torch.load(parent_path/'checkpoint.pt',map_location='cuda',weights_only=True)
        counts={}
        for label,directory in (('full',full),('decisions',decisions)):
            seal=closed(directory);assert sha(directory/'checkpoint.pt')==seal['checkpoint_sha256']
            checkpoint=torch.load(directory/'checkpoint.pt',map_location='cuda',weights_only=True)
            assert checkpoint['updates_completed']==parent['updates_completed']+4
            assert checkpoint['consumed_rollouts'][:len(parent['consumed_rollouts'])]==parent['consumed_rollouts']
            assert len(checkpoint['consumed_rollouts'])==len(parent['consumed_rollouts'])+4
            counts[label]=dict(updates_completed=checkpoint['updates_completed'],new_updates=4,new_consumed_rollouts=4)
        save(root/'matched-update-counts.json',dict(state='matched',parent_updates_completed=parent['updates_completed'],counts=counts))
        evaluation=root/'evaluation'
        step('evaluation',['scripts/evaluate_registered_combat_models.py','--spec',args.spec,'--out',evaluation],evaluation)
        assert closed(evaluation)['native_cases']==32 and closed(evaluation)['cuda_cases']==24
        step('native-quality',['scripts/audit_registered_native_action_quality.py','--evaluation',evaluation,'--out',root/'native-quality'],root/'native-quality')
        step('damage',['scripts/audit_combat_damage_distribution.py','--evaluation',evaluation,'--out',root/'damage'],root/'damage')
        save(root/'complete.json',dict(state='complete',version=study['version'],native_training_cases=80,cuda_training_cases=80,
             native_evaluation_cases=32,cuda_evaluation_cases=24,protocol_sha256=sha(args.protocol),
             final_weights={label:sha(path/'weights.json') for label,path in (('full',full),('decisions',decisions))},
             evaluation_seal_sha256=sha(evaluation/'complete.json'),matched_updates_from_parent=4,promotion=False))
        stage('complete')
    except BaseException as error:
        save(root/'progress.json',dict(stage='failed',runner_pid=os.getpid(),error=str(error)))
        raise


if __name__=='__main__':main()

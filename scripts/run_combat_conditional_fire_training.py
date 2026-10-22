"""Two predeclared CUDA fits and a fresh paired native mechanism screen."""
import argparse
import pathlib
import subprocess
import sys
from evaluate_registered_combat_models import read,save,sha
from run_registered_combat_ppo_pilot import lock


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=pathlib.Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();path=args.protocol.resolve();root=path.parent;p=read(path)
    repo=pathlib.Path(__file__).resolve().parents[1]
    assert p['version']=='combat_conditional_fire_training_v1' and not p['promotion']
    assert p['cuda_only'] and p['max_instances']==16 and p['timescale']==2
    assert p['new_training_cases']==32 and p['max_actor_steps']==10 and p['value_steps']==40
    assert p['branches']==[dict(label='full',scope='all'),dict(label='fire-only',scope='conditional-fire')]
    import torch
    assert torch.cuda.is_available()
    with lock(root/'training-execution.lock'):
        corpus=root/'corpus';closed=read(corpus/'complete.json')
        assert closed['state']=='complete' and closed['native_cases']==closed['cuda_cases']==32
        gate=pathlib.Path(p['native_gate']);assert read(gate)['state']=='complete' and read(gate)['semantic_episode_parity']
        strict=gate.parent/'full-trace-parity.json';assert read(strict)['strict_full_trace_parity']
        source=pathlib.Path(p['source']);pins={str(v):sha(v) for v in (path,root/'evaluation-spec.json',corpus/'complete.json',gate,strict,source/'complete.json')}
        for name in ('run_combat_conditional_fire_training.py','audit_combat_conditional_fire_fit_cuda.py','ppo_recurrent.py','combat_conditional_fire.py','combat_sequence_retention.py','combat_weapon_state_residual.py'):
            file=repo/'scripts'/name;pins[str(file)]=sha(file)
        def verify():
            for name,digest in pins.items():assert sha(pathlib.Path(name))==digest,'Pinned input changed: '+name
        def run(label,script,arguments,seal):
            verify();save(root/'progress.json',dict(stage=label,completed=completed,promotion=False))
            if seal.exists():
                assert args.resume,'Existing phase needs --resume'
            else:
                with (root/(label+'.log')).open('w',encoding='utf-8') as log:
                    result=subprocess.run([sys.executable,str(repo/'scripts'/script),*map(str,arguments)],cwd=repo,
                        stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                assert result.returncode==0,'Inspect '+str(root/(label+'.log'))
            value=read(seal);assert value.get('state','complete')=='complete'
            if 'checkpoint_sha256' in value:
                for filename,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
                    assert sha(seal.parent/filename)==value[key]
            verify();return value
        completed=[]
        try:
            config=read(corpus/'training-config.json');assert config['actor_steps']==10 and config['value_steps']==40
            retention=root/'retention.json'
            run('retention','prepare_combat_sequence_retention.py',[
                '--pool',corpus/'pool','--plan-index','0','--processing',corpus/'processing-train','--out',retention],retention)
            for item in p['branches']:
                label,scope=item['label'],item['scope'];update=root/(label+'-update');boundary=root/(label+'-initial.pt')
                run(label+'-fit','ppo_recurrent.py',[
                    '--model',source/'weights.json','--resume',source/'checkpoint.pt',
                    '--data',corpus/'processing-train/merged','--config',corpus/'training-config.json',
                    '--anchor-model',p['anchor'],'--retention-bank',p['bank'],'--retention-weight','0','--bank-weight','0',
                    '--sequence-retention-spec',retention,'--sequence-retention-weight',p['sequence_retention_weight'],'--fork-sequence-retention',
                    '--actor-head-scope',scope,'--conditional-fire-lr',p['conditional_fire_lr'],
                    '--fork-head-scope','--initial-boundary',boundary,'--out',update],update/'complete.json')
                audit=root/(label+'-audit.json')
                run(label+'-audit','audit_combat_conditional_fire_fit_cuda.py',[
                    '--source',source,'--update',update,'--boundary',boundary,'--corpus',corpus,'--scope',scope,'--out',audit],audit)
                completed.append(dict(label=label,scope=scope,update_complete_sha256=sha(update/'complete.json'),audit_sha256=sha(audit)))
            save(root/'training-complete.json',dict(state='complete',completed=completed,new_training_cases=32,device='cuda',promotion=False))
            run('evaluation','evaluate_registered_combat_models.py',['--spec',root/'evaluation-spec.json','--out',root/'evaluation'],root/'evaluation/complete.json')
            save(root/'complete.json',dict(state='complete',protocol_sha256=sha(path),training_complete_sha256=sha(root/'training-complete.json'),
                evaluation_complete_sha256=sha(root/'evaluation/complete.json'),promotion=False))
            save(root/'progress.json',dict(stage='complete',completed=completed,promotion=False))
        except BaseException as error:
            save(root/'progress.json',dict(stage='failed',completed=completed,error=str(error),promotion=False));raise


if __name__=='__main__':main()

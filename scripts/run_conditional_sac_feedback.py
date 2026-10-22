"""Finite equal-environment-budget SAC/PPO own-policy feedback experiment.

Resume accepts sealed phases only. Incomplete captures/updates are preserved for
diagnosis; never silently restarted. One native pool is active at a time.
"""
import argparse
import pathlib
import subprocess
import sys
from process_combat_architecture_pool import read,sha
from run_registered_combat_ppo_pilot import lock,save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=pathlib.Path,required=True)
    ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    path=a.protocol.resolve();root=path.parent;protocol=read(path);repo=pathlib.Path(__file__).resolve().parents[1]
    assert protocol['version']=='conditional_sac_feedback_v1' and protocol['rounds']==2
    assert protocol['cuda_only'] and protocol['max_instances']==16 and protocol['timescale']==2
    assert protocol['new_native_training_budget']==128 and not protocol['promotion']
    assert not protocol['selection_on_intermediate_evaluation']
    import torch
    assert torch.cuda.is_available()
    with lock(root/'execution.lock'):
        assert a.resume or not (root/'progress.json').exists()
        pins={str(path):sha(path)}
        for item in protocol['inputs']:pins[item['path']]=item['sha256']
        snapshot=root/'execution-source.json'
        if not snapshot.exists():
            assert not a.resume
            files=set((repo/'internal').rglob('*.go'))|set((repo/'cmd').rglob('*.go'))
            files|=set((repo/'scripts').glob('*.py'))|set((repo/'scripts').rglob('*.ps1'))
            native=repo.parent/'yquake2/src'
            files|=set(native.rglob('*.c'))|set(native.rglob('*.h'))
            save(snapshot,dict(state='complete',files=[dict(path=str(p.resolve()),sha256=sha(p)) for p in sorted(files)]))
        pins[str(snapshot)]=sha(snapshot)
        for item in read(snapshot)['files']:pins[item['path']]=item['sha256']
        def verify():
            for p,h in pins.items():assert sha(p)==h,'Pinned input changed: '+p
        verify();completed=[];seen=set();sources={k:pathlib.Path(v) for k,v in protocol['sources'].items()}
        replay=[pathlib.Path(protocol['initial_replay'])]
        def run(label,script,arguments,output,seal='complete.json'):
            verify();save(root/'progress.json',dict(stage=label,completed=completed))
            if not output.exists():
                with (root/(label+'.log')).open('w',encoding='utf-8') as f:
                    result=subprocess.run([sys.executable,str(repo/'scripts'/script),*map(str,arguments)],cwd=repo,
                        stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                assert result.returncode==0,'Inspect '+str(root/(label+'.log'))
            else:assert a.resume,'Unexpected output; no automatic reuse'
            receipt=output/seal if output.is_dir() else output
            assert receipt.exists(),'Incomplete phase preserved: '+str(output)
            value=read(receipt)
            if 'state' in value:assert value['state']=='complete'
            if seal=='complete.json' and 'weights_sha256' in value:
                for filename,key in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):assert sha(output/filename)==value[key]
            verify();return value
        try:
            for number in range(1,3):
                for algorithm in ('sac','ppo'):
                    source=sources[algorithm];label=f'round-{number}-{algorithm}'
                    corpus=root/(label+'-corpus');port=protocol['ports'][number-1][algorithm]
                    run(label+'-collect','collect_combat_aim_feedback_corpus.py',[
                        '--source',source,'--registry',protocol['registry'],'--episodes',','.join(protocol['episodes']),
                        '--train-offset',protocol['offsets'][number-1][algorithm],'--port',port,
                        '--allow-groups','--ppo-ready','--train-only','--out',corpus],corpus)
                    binding=read(corpus/'protocol.json')
                    assert pathlib.Path(binding['source'])==source and binding['train_only']
                    assert len(binding['assignments'])==1
                    seeds=set(binding['assignments'][0]['seeds']);assert len(seeds)==32 and not seeds.intersection(seen);seen.update(seeds)
                    update=root/(label+'-update')
                    if algorithm=='sac':
                        cache=root/(label+'-replay')
                        run(label+'-replay','prepare_combat_replay_cuda.py',['--source',source,'--corpus',corpus,'--out',cache],cache)
                        replay.append(cache)
                        run(label+'-fit','train_combat_conditional_sac_cuda.py',['--source',source,'--replay',*replay,
                            '--resume','--warmup','0','--steps','256','--out',update],update)
                        audit=root/(label+'-audit.json')
                        run(label+'-audit','audit_combat_sac_continuation_cuda.py',['--source',source,'--update',update,
                            '--fresh-replay',cache,'--out',audit],audit)
                    else:
                        retention=root/(label+'-retention.json')
                        run(label+'-retention','prepare_combat_sequence_retention.py',['--pool',corpus/'pool',
                            '--plan-index','0','--processing',corpus/'processing-train','--out',retention],retention)
                        run(label+'-fit','ppo_recurrent.py',['--model',source/'weights.json','--resume',source/'checkpoint.pt',
                            '--data',corpus/'processing-train/merged','--config',corpus/'training-config.json',
                            '--anchor-model',protocol['anchor'],'--retention-bank',protocol['bank'],'--retention-weight','0','--bank-weight','0',
                            '--sequence-retention-spec',retention,'--sequence-retention-weight','0','--fork-sequence-retention',
                            '--actor-head-scope','conditional-aim','--weapon-state-lr','.0003','--out',update],update)
                    sources[algorithm]=update
                    completed.append(dict(round=number,algorithm=algorithm,source=str(source),update=str(update),
                        update_complete_sha256=sha(update/'complete.json'),corpus=str(corpus),corpus_complete_sha256=sha(corpus/'complete.json'),seeds=sorted(seeds)))
                audit=root/f'round-{number}-pair-audit.json'
                run(f'round-{number}-pair-audit','audit_conditional_aim_pair_cuda.py',['--source',protocol['initial_source'],
                    '--ppo',sources['ppo'],'--sac',sources['sac'],'--replay',protocol['initial_replay'],'--out',audit],audit)
            assert len(seen)==128
            training=root/'training-complete.json'
            value=dict(state='complete',new_native_cases=128,algorithm_total_training_cases=96,device='cuda',
                completed=completed,sources={k:str(v) for k,v in sources.items()},promotion=False)
            if training.exists():assert a.resume and read(training)==value
            else:save(training,value)
            evaluation=root/'evaluation';bundle=root/'bundle';spec=root/'evaluation-spec.json'
            run('evaluation','evaluate_registered_combat_models.py',['--spec',spec,'--out',evaluation],evaluation)
            run('reports','report_combat_hypothesis_bundle.py',['--evaluation',evaluation,'--spec',spec,'--out',bundle],bundle)
            outcomes=bundle/'machinegun-outcomes.json'
            run('native-outcomes','report_registered_machinegun_outcomes.py',['--evaluation',evaluation,'--spec',spec,
                '--aim',bundle/'machinegun-aim.json','--out',outcomes],outcomes)
            save(root/'complete.json',dict(state='complete',version=protocol['version'],new_native_training_cases=128,
                native_evaluation_cases=32,cuda_evaluation_cases=24,training_performed=True,promotion=False,
                protocol_sha256=sha(path),training_complete_sha256=sha(training),bundle_complete_sha256=sha(bundle/'complete.json')))
            save(root/'progress.json',dict(stage='complete',completed=completed))
            print('Conditional SAC/PPO feedback complete',flush=True)
        except BaseException as error:
            save(root/'progress.json',dict(stage='failed',completed=completed,error=str(error)));raise


if __name__=='__main__':main()

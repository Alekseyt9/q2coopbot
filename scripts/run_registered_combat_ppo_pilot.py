"""Finite objective-matched own-policy rounds using existing collector/CUDA fitter.

Resume reuses sealed phases only. An incomplete phase is kept for diagnosis and
never automatically recollected or overwritten. The OS lock prevents duplicates.
"""
import argparse
import contextlib
import json
import os
import pathlib
import subprocess
import sys
from process_combat_architecture_pool import read,sha


def save(path,value):
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


@contextlib.contextmanager
def lock(path):
    with path.open('a+b') as stream:
        stream.seek(0,2)
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=pathlib.Path,required=True)
    ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    protocol_path=args.protocol.resolve();root=protocol_path.parent;protocol=read(protocol_path)
    assert protocol['version']=='combat_broad_ppo_pilot_v1' and not protocol['promotion']
    assert protocol['cuda_only'] and protocol['max_instances']==16 and protocol['timescale']==2
    assert not protocol['selection_on_intermediate_validation'] and not protocol['objective_changed']
    rounds=protocol['rounds'];assert 1<=len(rounds)<=16
    assert [r['round'] for r in rounds]==list(range(1,len(rounds)+1))
    assert all(r['cases']==4*len(r['episodes']) and len(set(r['episodes']))==len(r['episodes']) for r in rounds)
    assert sum(r['cases'] for r in rounds)==protocol['native_training_budget']
    assert len({e for r in rounds for e in r['episodes']})==protocol['unique_families']
    repo=pathlib.Path(__file__).resolve().parents[1]
    import torch
    assert torch.cuda.is_available(), 'CUDA required; no CPU fallback'
    with lock(root/'execution.lock'):
        assert args.resume or not (root/'progress.json').exists(), 'Existing execution: inspect and use --resume for sealed phases'
        pins={str(protocol_path):sha(protocol_path)}
        for item in protocol['inputs']:pins[item['path']]=item['sha256']
        def verify():
            for path,digest in pins.items():assert sha(path)==digest,'Pinned input changed: '+path
        verify()
        source_pin=root/'execution-source.json'
        if not source_pin.exists():
            assert not args.resume, 'No source snapshot for resumed pilot'
            paths=set((repo/'internal').rglob('*.go'))|set((repo/'cmd').rglob('*.go'))
            paths|=set((repo/'scripts').glob('*.py'))|set((repo/'scripts').rglob('*.ps1'))
            native=repo.parent/'yquake2/src'
            if native.exists():paths|=set(native.rglob('*.c'))|set(native.rglob('*.h'))
            save(source_pin,dict(state='complete',files=[dict(path=str(p.resolve()),sha256=sha(p)) for p in sorted(paths)]))
        snapshot=read(source_pin)
        assert snapshot['state']=='complete'
        pins[str(source_pin)]=sha(source_pin)
        for item in snapshot['files']:pins[item['path']]=item['sha256']
        verify()
        source=pathlib.Path(protocol['source']);assert sha(source/'complete.json')==protocol['source_complete_sha256']
        initial=torch.load(source/'checkpoint.pt',map_location='cuda',weights_only=True)
        assert initial['config']['actor_steps']==protocol['actor_steps_per_round']
        assert initial['config']['value_steps']==protocol['value_steps_per_round'];del initial
        completed=[];seen=set();current=None
        def run(label,command):
            with (root/(label+'.log')).open('w',encoding='utf-8') as output:
                process=subprocess.run([sys.executable,*map(str,command)],cwd=repo,stdout=output,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            assert process.returncode==0,'Inspect '+str(root/(label+'.log'))
        try:
            for current in rounds:
                number=current['round'];corpus=pathlib.Path(current['corpus']);fit=pathlib.Path(current['fit'])
                verify()
                save(root/'progress.json',dict(stage='collecting',round=number,total_rounds=len(rounds),completed=completed,source=str(source)))
                if not corpus.exists():
                    run(f'round-{number}-collect',[repo/'scripts/collect_combat_aim_feedback_corpus.py',
                        '--source',source,'--registry',protocol['registry'],'--episodes',','.join(current['episodes']),
                        '--train-offset',current['train_offset_minimum'],'--port',current['port'],
                        '--allow-groups','--ppo-ready','--train-only','--out',corpus])
                else:assert args.resume, 'Unexpected existing corpus; no automatic reuse'
                assert (corpus/'complete.json').exists(),'Incomplete corpus preserved; diagnose before resuming'
                closed=read(corpus/'complete.json');assert closed['state']=='complete' and not closed['training_performed']
                data=read(corpus/'protocol.json');assert sha(corpus/'protocol.json')==closed['protocol_sha256']
                assert data['source']==str(source.resolve()) and data['train_only'] and data['ppo_training_objective']
                assert len(data['assignments'])==1
                assignment=data['assignments'][0];seeds=set(assignment['seeds'])
                assert assignment['split']=='train' and len(seeds)==current['cases'] and not seen.intersection(seeds)
                assert assignment['offset']>=current['train_offset_minimum'];seen.update(seeds)
                assert set(t['episode']['id'] for t in read(assignment['plan'])['tasks'])==set(current['episodes'])
                save(root/'progress.json',dict(stage='cuda-fitting',round=number,total_rounds=len(rounds),completed=completed,source=str(source)))
                if not fit.exists():
                    run(f'round-{number}-fit',[repo/'scripts/fit_registered_combat_ppo_cuda.py',
                        '--source',source,'--corpus',corpus,'--anchor',protocol['anchor'],'--bank',protocol['bank'],
                        '--sequence-retention-weight',protocol['sequence_retention_weight'],'--out',fit])
                else:assert args.resume, 'Unexpected existing fit; no automatic reuse'
                assert (fit/'complete.json').exists(),'Incomplete fit preserved; inspect its checkpoint before resuming'
                result=read(fit/'complete.json');assert result['state']=='complete' and result['device']=='cuda'
                assert pathlib.Path(result['source']).resolve()==source.resolve() and pathlib.Path(result['corpus']).resolve()==corpus.resolve()
                assert result['training_cases']==current['cases'] and not result['promotion']
                for item in result['inputs']:assert sha(item['path'])==item['sha256']
                update=pathlib.Path(result['update']);receipt=read(update/'complete.json')
                assert sha(update/'complete.json')==result['update_complete_sha256']
                for name,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:assert sha(update/name)==receipt[key]
                completed.append(dict(round=number,source=str(source),source_weights_sha256=result['source_weights_sha256'],
                    corpus=str(corpus),corpus_complete_sha256=sha(corpus/'complete.json'),fit=str(fit),fit_complete_sha256=sha(fit/'complete.json'),
                    update=str(update),accepted_actor_steps=result['accepted_actor_steps'],rows=result['rows'],seeds=sorted(seeds)))
                source=update
                verify()
                save(root/'progress.json',dict(stage='round-complete',round=number,total_rounds=len(rounds),completed=completed))
            complete=dict(state='complete',version='combat_broad_ppo_training_complete_v1',phase='training_only',
                protocol_sha256=pins[str(protocol_path)],execution_source_sha256=pins[str(source_pin)],device='cuda',rounds=completed,
                native_cases=len(seen),accepted_actor_steps=sum(r['accepted_actor_steps'] for r in completed),
                final_update=str(source),final_update_complete_sha256=sha(source/'complete.json'),
                training_performed=True,promotion=False,quality_unproven=True)
            if (root/'complete.json').exists():assert read(root/'complete.json')==complete
            else:save(root/'complete.json',complete)
            save(root/'progress.json',dict(stage='complete',rounds=len(completed),native_cases=len(seen),accepted_actor_steps=complete['accepted_actor_steps']))
            print(json.dumps({k:v for k,v in complete.items() if k!='rounds'}))
        except BaseException as error:
            save(root/'progress.json',dict(stage='failed',round=current['round'] if current else None,completed=completed,error=str(error)))
            raise


if __name__=='__main__':main()

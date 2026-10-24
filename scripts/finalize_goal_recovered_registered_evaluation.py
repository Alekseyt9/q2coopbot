"""Finish uniform goal-boundary reprocessing of preserved raw captures; CUDA checks only."""
import argparse,pathlib,subprocess,sys
from process_combat_architecture_pool import read,save,sha


def main():
    ap=argparse.ArgumentParser()
    for name in ('evaluation','spec'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();root=a.evaluation.resolve();repo=pathlib.Path(__file__).resolve().parents[1]
    assert not (root/'complete.json').exists()
    assert read(root/'progress.json')['stage']==('failed' if a.resume else 'native-recovered')
    recovery_path=root/'recovery-native-complete.json';recovery=read(recovery_path);assert recovery['version']=='combat_goal_pool_reprocessing_v1' and recovery['new_games']==0
    spec=read(a.spec);fixture_path=root/'fixture-proof.json';fixture=read(fixture_path)
    pool=root/'pool';pool_path=pool/'report.json';report=read(pool_path);digest=sha(pool_path)
    assert recovery['state']==report['state']=='complete' and recovery['pool_sha256']==digest
    assert report['source_unchanged'] and report['usable_captures']==len(report['jobs'])==fixture['native_cases']
    assert all(not j['error'] for j in report['jobs'])
    assert sha(a.spec)==fixture['spec_sha256'] and sha(spec['registry'])==fixture['registry_sha256']
    labels=[v['label'] for v in spec['variants']];assert len(labels)==len(set(labels))==len(report['plans'])
    pins={recovery_path:sha(recovery_path),pool_path:digest,a.spec:sha(a.spec),fixture_path:sha(fixture_path),pathlib.Path(spec['registry']):fixture['registry_sha256']}
    scripts=['audit_registered_damage_timing.py','compare_registered_combat_pools.py',
        'audit_registered_combat_ownership.py','audit_registered_target_choices.py',
        'audit_registered_weapon_choices.py','audit_registered_execution_actions.py',
        'process_registered_combat_pool_cuda.py','summarize_registered_action_efficiency.py']
    for name in scripts:pins[repo/'scripts'/name]=sha(repo/'scripts'/name)
    for index,binding in enumerate(fixture['plans']):
        p=pathlib.Path(binding['path']);assert sha(p)==binding['sha256'];pins[p]=binding['sha256']
        assert sha(root/(labels[index]+'-plan.json'))==binding['sha256']==report['plans'][index]['sha256']
    for binding in fixture['models']:
        update=pathlib.Path(binding['update']);seal_path=update/'complete.json';seal=read(seal_path)
        assert sha(seal_path)==binding['update_seal_sha256'];pins[seal_path]=binding['update_seal_sha256']
        for filename,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
            p=update/filename;assert sha(p)==seal[key];pins[p]=seal[key]
        p=pathlib.Path(binding['model']);assert sha(p)==binding['model_sha256'];pins[p]=binding['model_sha256']
    for binding in fixture['readiness']:
        p=pathlib.Path(binding['path']);assert sha(p)==binding['sha256'];pins[p]=binding['sha256']
    def run(label,script,*args):
        with (root/(label+'.log')).open('w',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,str(repo/'scripts'/script),*map(str,args)],cwd=repo,
                stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert result.returncode==0,str(root/(label+'.log'))
    processed=[]
    try:
        save(root/'progress.json',dict(stage='damage-timing',training_performed=False))
        if a.resume:
            assert read(root/'damage-timing.json')['pool_sha256']==digest
        else:run('damage-timing',scripts[0],'--pool',pool,'--out',root/'damage-timing.json')
        comparisons=[]
        for c in spec['comparisons']:
            save(root/'progress.json',dict(stage='paired-comparison',label=c['name'],training_performed=False))
            target=root/('quality-'+c['name']+'.json')
            if a.resume:
                original=read(root/'native-complete.json')
                assert original['state']=='complete' and original['pool_sha256']==digest
                binding=next(b for b in original['comparisons'] if b['name']==c['name'])
                assert sha(target)==binding['sha256'] and binding['before']==c['before'] and binding['after']==c['after']
            else:
                run('quality-'+c['name'],scripts[1],'--before',pool,'--before-plan-index',labels.index(c['before']),
                    '--after',pool,'--after-plan-index',labels.index(c['after']),'--out',target)
            comparisons.append(dict(**c,sha256=sha(target)))
        if not a.resume:
            save(root/'native-complete.json',dict(state='complete',native_cases=len(report['jobs']),pool_sha256=digest,
                comparisons=comparisons,recovery_sha256=pins[recovery_path],training_performed=False,promotion=False))
        behavior=[]
        for index,v in enumerate(spec['variants']):
            if v['mode']!='learned':continue
            label=v['label'];save(root/'progress.json',dict(stage='auditing',label=label,processed=processed,training_performed=False))
            for script,prefix in zip(scripts[2:6],['ownership','behavior','weapon','execution']):
                target=root/(prefix+'-'+label+'.json')
                if a.resume and target.exists():
                    reused=read(target)
                    assert reused['pool_sha256']==digest and reused['selected_plan_index']==index
                    assert reused['script_sha256']==sha(repo/'scripts'/script)
                    assert reused.get('state','complete')=='complete' and len(reused['cases'])==fixture['cases_per_variant']
                else:run(prefix+'-'+label,script,'--pool',pool,'--plan-index',index,'--out',target)
            behavior+=['--audit',label+'='+str(root/('behavior-'+label+'.json'))]
            if v['cuda_replay']:
                directory=root/('processing-'+label)
                if not directory.exists():run('processing-'+label,scripts[6],'--pool',pool,'--plan-index',index,'--out',directory)
                else:assert a.resume and (directory/'report.json').exists(),'Partial CUDA batch needs explicit recovery'
                meta=read(directory/'report.json')
                assert meta['state']=='complete' and meta['device']=='cuda' and not meta['training_performed']
                assert meta['pool_sha256']==digest and meta['selected_plan_index']==index
                for task in read(directory/'cuda-request.json')['tasks']:
                    output=pathlib.Path(task['out']);data=pathlib.Path(task['data']);info=read(output/'report.json')
                    assert task['native_report_sha256']==sha(data/'report.json')
                    assert task['model_sha256']==info['model_sha256']==meta['model_sha256']
                    assert sha(output/'rollout.jsonl')==info['rollout_sha256']
                    assert sha(output/'sequence.jsonl')==info['sequence_sha256']
                    assert sha(output/'cuda-verification.json')==info['cuda_verification_sha256']
                    assert read(output/'cuda-verification.json')['device']=='cuda'
                expected=fixture['cases_per_variant'];assert len(meta['cases'])==expected
                processed.append(dict(label=label,cases=expected,rows=meta['rows'],report_sha256=sha(directory/'report.json')))
        run('action-efficiency',scripts[7],*behavior,'--damage',root/'damage-timing.json','--out',root/'action-efficiency.json')
        for p,d in pins.items():assert sha(p)==d,str(p)
        save(root/'complete.json',dict(state='complete',native_cases=len(report['jobs']),cuda_cases=sum(p['cases'] for p in processed),
            spec_sha256=pins[a.spec],pool_sha256=digest,comparisons=comparisons,processed=processed,readiness=fixture['readiness'],
            recovery_sha256=pins[recovery_path],source_pins={str(p):d for p,d in pins.items()},script_sha256=sha(pathlib.Path(__file__)),
            training_performed=False,promotion=False,
            scope='All original32raw captures retained, uniformly reprocessed through fixed goal-boundary finalization; no game rerun. Native/CUDA verification only, no fitting or promotion.'))
        save(root/'progress.json',dict(stage='complete',native_cases=len(report['jobs']),processed=processed,training_performed=False))
        print('Recovered evaluation verified',flush=True)
    except BaseException as error:
        save(root/'progress.json',dict(stage='failed',error=str(error),processed=processed,training_performed=False));raise


if __name__=='__main__':main()

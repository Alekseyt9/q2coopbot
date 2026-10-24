"""Resume exact prepared evaluation plans after a proven pre-native port collision."""
import argparse
import os
import pathlib
import shutil
import subprocess
import sys

from evaluate_registered_combat_models import read, save, sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation','spec','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    ap.add_argument('--port',type=int,required=True)
    a=ap.parse_args();repo=pathlib.Path(__file__).resolve().parents[1]
    root,out=a.evaluation.resolve(),a.out.resolve();assert not out.exists()
    spec_path=a.spec.resolve();spec_sha=sha(spec_path);spec=read(spec_path)
    state=read(root/'progress.json');assert state['stage']=='failed'
    log=root/'native-pool.log';assert 'Pool port range occupied' in log.read_text(encoding='utf-8-sig')
    assert not (root/'pool').exists() and not (root/'native-complete.json').exists() and not (root/'complete.json').exists()
    assert 1024<=a.port<=65520 and a.port!=spec['port']
    fixture=read(root/'fixture-proof.json');assert fixture['state']=='matched' and fixture['spec_sha256']==spec_sha
    original_run=root.parent/'run.json';run_receipt=read(original_run);fence=run_receipt['source_files']
    if isinstance(fence,list):fence={item['path']:item['sha256'] for item in fence}
    own_sha=sha(__file__)
    def unchanged():
        assert sha(__file__)==own_sha and sha(spec_path)==spec_sha
        assert all(sha(p)==h for p,h in fence.items()),'Original source fence changed'
    unchanged()
    plans=[]
    for binding in fixture['plans']:
        path=pathlib.Path(binding['path']);assert sha(path)==binding['sha256'];plan=read(path)
        assert not pathlib.Path(plan['output_root']).exists(),'Capture output exists; not a pre-native collision'
        plans.append(path)
    variants=spec['variants'];labels=[v['label'] for v in variants]
    assert [p.stem.removesuffix('-plan') for p in plans]==labels
    count,total=fixture['cases_per_variant'],fixture['native_cases'];assert total==count*len(variants)
    bindings,readiness=fixture['models'],fixture['readiness'];registry=pathlib.Path(spec['registry']);registry_sha=fixture['registry_sha256']
    assert sha(registry)==registry_sha
    for item in readiness:assert sha(item['path'])==item['sha256']
    for item in bindings:
        assert sha(item['model'])==item['model_sha256'] and sha(pathlib.Path(item['update'])/'complete.json')==item['update_seal_sha256']
    out.mkdir(parents=True)
    shutil.copyfile(root/'progress.json',out/'previous-evaluation-progress.json')
    save(out/'recovery.json',dict(version='registered_evaluation_pre_native_port_recovery_v1',
        original_spec_sha256=spec_sha,original_fixture_sha256=sha(root/'fixture-proof.json'),
        original_failure_log_sha256=sha(log),original_run_sha256=sha(original_run),
        declared_port=spec['port'],actual_port=a.port,native_cases_before_recovery=0,
        plans=fixture['plans'],models=bindings,script_sha256=own_sha,
        scope='Exact prepared plans and capture roots retained. No games or captures existed; only occupied UDP range replaced. Other services untouched.'))
    env=dict(os.environ,GOCACHE=str(repo/'workspace/build/gocache'),GOTOOLCHAIN='auto')
    def run(label,command):
        unchanged()
        with (out/(label+'.log')).open('w',encoding='utf-8') as output:
            child=subprocess.Popen(list(map(str,command)),cwd=repo,env=env,stdout=output,stderr=output,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            save(out/'progress.json',dict(stage=label,runner_pid=os.getpid(),child_pid=child.pid))
            result=child.wait()
        assert result==0,str(out/(label+'.log'));unchanged()
    def python(label,script,*arguments):run(label,[sys.executable,repo/'scripts'/script,*arguments])
    pool=root/'pool';shell=shutil.which('pwsh');assert shell
    literal=lambda p:"'"+str(p).replace("'","''")+"'"
    command='& '+literal(repo/'scripts/run_registered_combat_pool.ps1')+' -Plans @('+','.join(map(literal,plans))+') -MaxInstances 16 -Port '+str(a.port)+' -OutputRoot '+literal(pool)
    try:
        run('native-pool',[shell,'-NoProfile','-Command',command])
        report = read(pool/'report.json'); digest = sha(pool/'report.json')
        assert report['state'] == 'complete' and report['source_unchanged'] and report['usable_captures'] == len(report['jobs']) == total
        assert all(not j['error'] for j in report['jobs'])
        python('damage-timing', 'audit_registered_damage_timing.py', '--pool', pool, '--out', root/'damage-timing.json')
        comparisons = []
        for comparison in spec['comparisons']:
            label = comparison['name']; target = root/('quality-'+label+'.json')
            python('quality-'+label, 'compare_registered_combat_pools.py', '--before', pool, '--before-plan-index', labels.index(comparison['before']),
                   '--after', pool, '--after-plan-index', labels.index(comparison['after']), '--out', target)
            comparisons.append(dict(**comparison, sha256=sha(target)))
        save(root/'native-complete.json', dict(state='complete', native_cases=total, pool_sha256=digest, comparisons=comparisons,
                                               training_performed=False, promotion=False))
        processed = []; behavior = []
        for index, variant in enumerate(variants):
            if variant['mode'] != 'learned': continue
            label = variant['label']; save(root/'progress.json', dict(stage='auditing', label=label, processed=processed))
            for script, prefix in [('audit_registered_combat_ownership.py', 'ownership'), ('audit_registered_target_choices.py', 'behavior'),
                                   ('audit_registered_weapon_choices.py', 'weapon'), ('audit_registered_execution_actions.py', 'execution')]:
                python(prefix+'-'+label, script, '--pool', pool, '--plan-index', index, '--out', root/(prefix+'-'+label+'.json'))
            behavior += ['--audit', label+'='+str(root/('behavior-'+label+'.json'))]
            if variant['cuda_replay']:
                processing = root/('processing-'+label)
                python('processing-'+label, 'process_registered_combat_pool_cuda.py', '--pool', pool, '--plan-index', index, '--out', processing)
                meta = read(processing/'report.json')
                assert meta['state'] == 'complete' and meta['device'] == 'cuda' and not meta['training_performed'] and len(meta['cases']) == count
                processed.append(dict(label=label, cases=count, rows=meta['rows'], report_sha256=sha(processing/'report.json')))
        python('action-efficiency', 'summarize_registered_action_efficiency.py', *behavior, '--damage', root/'damage-timing.json', '--out', root/'action-efficiency.json')
        assert sha(pool/'report.json') == digest and sha(spec_path) == spec_sha and sha(registry) == registry_sha
        for item in readiness: assert sha(item['path']) == item['sha256']
        for item in bindings:
            assert sha(item['model']) == item['model_sha256'] and sha(pathlib.Path(item['update'])/'complete.json') == item['update_seal_sha256']
        save(root/'complete.json', dict(state='complete', native_cases=total, cuda_cases=sum(p['cases'] for p in processed),
             spec_sha256=spec_sha, pool_sha256=digest, comparisons=comparisons, processed=processed, readiness=readiness,
             script_sha256=sha(pathlib.Path(__file__)), operational_recovery=str((out/'recovery.json').resolve()), operational_recovery_sha256=sha(out/'recovery.json'), training_performed=False, promotion=False,
             scope='Exact already-frozen fixtures resumed after a pre-native port collision. Only UDP port changed, recorded in operational recovery. No new seed, resampling, fitting, promotion or quality acceptance.'))
        save(root/'progress.json', dict(stage='complete', native_cases=total, processed=processed))
        unchanged()
        save(out/'complete.json',dict(state='complete',evaluation_seal_sha256=sha(root/'complete.json'),
            recovery_sha256=sha(out/'recovery.json'),source_unchanged=True,training_performed=False,promotion=False))
        save(out/'progress.json',dict(stage='complete',runner_pid=os.getpid(),child_pid=None))
    except BaseException as error:
        save(out/'progress.json',dict(stage='failed',runner_pid=os.getpid(),error=str(error),training_performed=False))
        raise


if __name__=='__main__':main()

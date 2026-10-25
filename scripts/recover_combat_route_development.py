"""Recover interrupted development screen, preserving every valid closed capture."""
import argparse
import concurrent.futures
import copy
from pathlib import Path
import shutil
import subprocess
import sys
from evaluate_registered_combat_models import read,save,sha
from process_combat_architecture_pool import compact_closed_exports
from process_coop_learning_pool_cuda import verify_memory_export


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--screen',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--resume-native',action='store_true',help='Reuse completed retry pool after offline reader repair')
    a=ap.parse_args();source=a.screen.resolve();out=a.out.resolve();repo=Path(__file__).resolve().parents[1]
    assert not out.exists() or a.resume_native
    spec=read(source/'spec.json');assert spec['version']=='combat_memory_route_development_screen_v1'
    for path,digest in spec['inputs'].items():assert sha(path)==digest,'Original input changed: '+path
    # Do not mistake stale slot files for live work. Caller must have verified
    # processes are absent; closed member receipts are independently checked.
    jobs={}
    for p in (source/'pool/jobs').glob('*-result.json'):
        j=read(p);assert not j['error'];r=read(Path(j['root'])/'report.json')
        assert r['capture_complete'] and r['provenance_valid']
        jobs[j['plan_index']]=j
    missing=[i for i in range(24) if i not in jobs]
    assert missing and jobs
    if a.resume_native:
        assert not (out/'pool/report.json').exists(), 'Use separate processing revision after merge'
        previous=read(out/'recovery-protocol.json')
        assert previous['preserved_cases']==sorted(jobs) and previous['repeated_unfinished_cases']==missing
        save(out/'failed-attempt-before-reader-repair.json',read(out/'progress.json'))
    else:
        out.mkdir(parents=True)
    def run(label,command):
        with (out/(label+'.log')).open('w',encoding='utf-8') as f:
            p=subprocess.run(list(map(str,command)),cwd=repo,stdout=f,stderr=subprocess.STDOUT,
                             creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert p.returncode==0,str(out/(label+'.log'))
    def stage(name):save(out/'progress.json',dict(stage=name))
    try:
        plans=[]
        for i in missing:
            plan=read(source/f'plan-{i:02d}.json')
            plan['output_root']=str(out/f'captures-{i:02d}')
            target=out/f'plan-{i:02d}.json';save(target,plan);plans.append(target)
        save(out/'recovery-protocol.json',dict(original=str(source),original_spec_sha256=sha(source/'spec.json'),
            preserved_cases=sorted(jobs),repeated_unfinished_cases=missing,new_native_games=len(missing),
            original_partial_directories_preserved=True,selection='Only missing terminal member receipts, no selection by quality.'))
        q=lambda p:"'"+str(p).replace("'","''")+"'"
        command='& '+q(repo/'scripts/run_registered_combat_pool.ps1')+' -Plans @('+','.join(map(q,plans))+')'
        command+=' -MaxInstances 16 -Port 49700 -OutputRoot '+q(out/'retry-pool')
        stage('recover-missing-native')
        if not a.resume_native:
            run('retry-pool',[shutil.which('pwsh'),'-NoProfile','-Command',command])
        retry=read(out/'retry-pool/report.json')
        assert retry['state']=='complete' and retry['source_unchanged'] and retry['usable_captures']==len(missing)
        for j in retry['jobs']:
            global_index=missing[j['plan_index']]
            jobs[global_index]={**j,'plan_index':global_index,'job':global_index}
        assert len(jobs)==24
        fingerprints=set();native_fingerprints=set()
        for i,j in jobs.items():
            d=Path(j['root']);r=read(d/'report.json');m=read(d/'manifest.json')
            assert r['capture_complete'] and r['provenance_valid'] and len(r['results'])==1
            result=r['results'][0];v=spec['variants'][i]
            assert result['capture_valid'] and result['dispatch_valid'] and result['seed_confirmed']
            assert result['seed']==v['seed'] and result['generated_fixture']==v['instance']
            assert sha(d/'q2combat-export.exe')==m['exporter_sha256'].lower()
            fingerprints.add(m['source_fingerprint']);native_fingerprints.add(m['native_source_fingerprint'])
        assert len(fingerprints)==len(native_fingerprints)==1,'Recovered native implementation differs'
        save(out/'spec.json',spec)
        (out/'pool').mkdir()
        save(out/'pool/report.json',dict(state='complete',source_unchanged=True,usable_captures=24,jobs=list(jobs.values()),
            recovery_protocol_sha256=sha(out/'recovery-protocol.json'),preserved_native_cases=len(jobs)-len(missing),
            new_native_cases=len(missing),scope='Recovered per-episode native receipts; original closed captures retained.'))
        exporter=out/'q2ppo-data.exe'
        run('build-exporter',['go','build','-o',exporter,'./cmd/q2ppo-data'])
        def export(j):
            i=j['plan_index'];v=spec['variants'][i];d=out/f'case-{i:02d}';d.mkdir()
            capture=Path(j['root']);result=read(capture/'report.json')['results'][0]
            run(f'export-{i:02d}',[exporter,'--batch',capture,'--model',v['model'],'--cuda-only','--out',d/'native'])
            verify_memory_export(Path(result['root'])/'bot.jsonl',d/'native/replay-0')
            compact_closed_exports(d/'native')
            return dict(model=v['model'],model_sha256=sha(v['model']),data=str(d/'native'),out=str(d/'cuda'),native_report_sha256=sha(d/'native/report.json'))
        stage('native-export')
        learned=[j for i,j in sorted(jobs.items()) if spec['variants'][i]['mode']=='learned']
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as workers:tasks=list(workers.map(export,learned))
        save(out/'cuda-request.json',dict(version='combat_cuda_finalize_batch_v1',compact_finalized_streams=True,tasks=tasks))
        stage('cuda-verification')
        run('cuda-batch',[sys.executable,repo/'scripts/finalize_combat_cuda_batch.py','--request',out/'cuda-request.json','--receipt-root',out/'cuda-batch'])
        assert read(out/'cuda-batch/report.json')['state']=='complete'
        for path,digest in spec['inputs'].items():assert sha(path)==digest
        stage('acceptance')
        run('acceptance',[sys.executable,repo/'scripts/audit_combat_memory_route_development.py','--screen',out,'--out',out/'acceptance.json'])
        stage('complete');print(out/'acceptance.json',flush=True)
    except BaseException as error:
        save(out/'progress.json',dict(stage='failed',error=str(error)));raise


if __name__=='__main__':main()

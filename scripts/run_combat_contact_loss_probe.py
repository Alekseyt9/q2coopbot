"""Run prepared contact-loss diagnostic variants through existing native pool and CUDA replay."""
import argparse
import concurrent.futures
import os
from pathlib import Path
import shutil
import subprocess
import sys
from evaluate_registered_combat_models import read,save,sha
from process_combat_architecture_pool import compact_closed_exports
from process_coop_learning_pool_cuda import verify_memory_export


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepared',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--port',type=int,default=49200)
    a=ap.parse_args();repo=Path(__file__).resolve().parents[1];out=a.out.resolve()
    assert not out.exists() and shutil.disk_usage('F:/').free>=24*1024**3
    pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    seal=read(pin(a.prepared/'complete.json'));prepared=read(pin(a.prepared/'report.json',seal['report_sha256']))
    assert prepared['version']=='combat_contact_loss_probe_preparation_v1' and prepared['device']=='cuda'
    assert prepared['neural_parameters_exact'] and prepared['zero_likelihood_exact'] and len(prepared['variants'])==24
    for p,h in prepared['inputs'].items():pin(p,h)
    for p in (repo/'scripts').glob('*.py'):pin(p)
    variants=prepared['variants']
    for v in variants:pin(v['model'],v['model_sha256'])
    out.mkdir();os.environ['GOCACHE']=str(repo/'workspace/build/gocache')
    def run(label,command):
        with (out/(label+'.log')).open('w',encoding='utf-8') as log:
            r=subprocess.run(list(map(str,command)),cwd=repo,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert r.returncode==0,str(out/(label+'.log'))
    def stage(stage):save(out/'progress.json',dict(stage=stage))
    try:
        planner,exporter=out/'q2episode.exe',out/'q2ppo-data.exe'
        run('build-planner',['go','build','-o',planner,'./cmd/q2episode'])
        run('build-exporter',['go','build','-o',exporter,'./cmd/q2ppo-data'])
        plans=[]
        for i,v in enumerate(variants):
            case=v['context'];original=read(pin(case['source_plan']));registry=pin(original['registry_path'],original['registry_sha256'])
            ep=original['tasks'][0]['episode'];offset=case['seed']-ep['splits']['train']['start']
            path=out/f'plan-{i:02d}.json'
            run(f'plan-{i:02d}',[planner,'--registry',registry,'--root',repo,'--episodes',case['episode'],'--split','train',
                '--mode','learned','--model',v['model'],'--count',1,'--seed-offset',offset,'--checked-wall-slide',
                '--out',path,'--artifacts',out/f'captures-{i:02d}'])
            p=read(path);assert p['tasks'][0]['instances']==[case['instance']] and p['tasks'][0]['seeds']==[case['seed']]
            plans.append(pin(path))
        spec=dict(version='combat_contact_loss_native_probe_v1',prepared=str(a.prepared.resolve()),prepared_report_sha256=seal['report_sha256'],
            variants=variants,inputs=pins,independent_worlds=3,policy_future_streams_per_world=2,native_engine_rng_forked=False,
            pool=8,timescale=2,diagnostic_only=True,training_performed=False,promotion=False)
        save(out/'spec.json',spec)
        snap=out/'source-snapshot';records=[]
        files=[p for folder in ('cmd','internal') for p in (repo/folder).rglob('*.go')]
        files += [p for p in (repo/'scripts').iterdir() if p.suffix in ('.py','.ps1')]
        native=repo.parent/'yquake2';sources=[(p,Path('q2coopbot-src')/p.relative_to(repo)) for p in files]
        sources += [(p,Path('yquake2')/p.relative_to(native)) for p in (native/'src').rglob('*') if p.is_file() and p.suffix in ('.c','.h')]
        for source,relative in sources:
            target=snap/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
            assert sha(source)==sha(target);records.append(dict(source=str(source),copy=str(relative),sha256=sha(target)))
        save(snap/'manifest.json',dict(files=records))
        quote=lambda p:"'"+str(p).replace("'","''")+"'"
        command='& '+quote(repo/'scripts/run_registered_combat_pool.ps1')+' -Plans @('+','.join(map(quote,plans))+')'
        command+=' -MaxInstances 8 -Port '+str(a.port)+' -OutputRoot '+quote(out/'pool')
        stage('native-capture');run('native-pool',[shutil.which('pwsh'),'-NoProfile','-Command',command])
        pool=read(out/'pool/report.json');assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==24
        assert len(pool['jobs'])==24 and all(not j['error'] for j in pool['jobs'])
        def export(job):
            i=job['plan_index'];v=variants[i];case=v['context'];capture=Path(job['root']);d=out/f'case-{i:02d}';d.mkdir()
            report=read(capture/'report.json');result=report['results'][0];mf=read(capture/'manifest.json')
            assert report['capture_complete'] and report['provenance_valid'] and result['capture_valid'] and result['dispatch_valid']
            assert result['seed_confirmed'] and result['seed']==case['seed'] and result['generated_fixture']==case['instance']
            assert mf['model_weights_sha256'].lower()==v['model_sha256']
            run(f'export-{i:02d}',[exporter,'--batch',capture,'--model',v['model'],'--cuda-only','--out',d/'native'])
            verify_memory_export(Path(result['root'])/'bot.jsonl',d/'native/replay-0');compact_closed_exports(d/'native')
            return dict(model=v['model'],model_sha256=v['model_sha256'],data=str(d/'native'),out=str(d/'cuda'),native_report_sha256=sha(d/'native/report.json'))
        stage('native-export')
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as workers:
            tasks=list(workers.map(export,sorted(pool['jobs'],key=lambda j:j['plan_index'])))
        save(out/'cuda-request.json',dict(version='combat_cuda_finalize_batch_v1',compact_finalized_streams=True,tasks=tasks))
        stage('cuda-verification');run('cuda-batch',[sys.executable,repo/'scripts/finalize_combat_cuda_batch.py','--request',out/'cuda-request.json','--receipt-root',out/'cuda-batch'])
        batch=read(out/'cuda-batch/report.json');assert batch['state']=='complete' and len(batch['tasks'])==24
        assert all(sha(p)==h for p,h in pins.items())
        save(out/'complete.json',dict(state='complete',native_cases=24,cuda_cases=24,spec_sha256=sha(out/'spec.json'),
            pool_sha256=sha(out/'pool/report.json'),cuda_receipt_sha256=sha(out/'cuda-batch/report.json'),diagnostic_only=True,training_performed=False,promotion=False))
        stage('acceptance');run('acceptance',[sys.executable,repo/'scripts/audit_combat_contact_loss_probe.py','--probe',out,'--out',out/'acceptance'])
        stage('complete');print(out/'acceptance/report.json')
    except BaseException as error:
        save(out/'progress.json',dict(stage='failed',error=str(error)));raise


if __name__=='__main__':main()

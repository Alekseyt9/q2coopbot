"""Run sealed TRAIN movement interventions; never merge them into PPO data."""
import argparse
import copy
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
    for name in ('prepared','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--port',type=int,default=45100)
    ap.add_argument('--fresh-reference',action='store_true',help='Add an ordinary same-policy control under the current frozen-world clock')
    a=ap.parse_args(); repo=Path(__file__).resolve().parents[1]; out=a.out.resolve()
    assert not out.exists() and shutil.disk_usage('F:/').free>=32*1024**3
    pins={}

    def pin(path,expected=None):
        path=Path(path).resolve(); h=sha(path); assert expected is None or h==expected,str(path)
        pins[str(path)]=h;return path

    seal=read(pin(a.prepared/'complete.json')); prepared=read(pin(a.prepared/'report.json',seal['report_sha256']))
    assert prepared['version'] in ('combat_movement_probe_preparation_v1','combat_continuation_probe_preparation_v1') and prepared['device']=='cuda'
    continuation=prepared['version']=='combat_continuation_probe_preparation_v1'
    assert not continuation or not a.fresh_reference,'Continuation preparations supply their own forked and unmodified references'
    assert prepared['diagnostic_only'] and not prepared['training_performed'] and not prepared['validation_used']
    context=prepared['context']; assert context['split']=='train'
    variants=copy.deepcopy(prepared['variants']); assert 2<=len(variants)<=6 and variants[0]['label']=='control'
    if a.fresh_reference:
        ordinary=Path(prepared['source'])/'weights.json'
        assert sha(ordinary)==prepared['source_weights_sha256']
        variants.append(dict(label='ordinary_reference',index=len(variants),model=str(ordinary.resolve()),
            model_sha256=sha(ordinary),mean_shift=None,frame=context['frame']))
    for path,h in prepared['inputs'].items():pin(path,h)
    registry=pin(prepared['registry'])
    for v in variants:pin(v['model'],v['model_sha256'])
    out.mkdir(parents=True)
    os.environ['GOCACHE']=str(repo/'workspace/build/gocache');os.environ['GOTOOLCHAIN']='auto'

    def run(label,command):
        with (out/(label+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(list(map(str,command)),cwd=repo,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert result.returncode==0,str(out/(label+'.log'))

    try:
        planner=out/'q2episode.exe';run('build-planner',['go','build','-o',planner,'./cmd/q2episode'])
        plans=[]
        for i,v in enumerate(variants):
            path=out/f'plan-{i:02d}.json'
            run(f'plan-{i:02d}',[planner,'--registry',registry,'--root',repo,'--episodes',context['episode'],'--split','train',
                '--mode','learned','--model',v['model'],'--count',1,'--seed-offset',context['offset'],'--checked-wall-slide',
                '--out',path,'--artifacts',out/f'captures-{i:02d}'])
            p=read(path);assert len(p['tasks'])==1 and p['tasks'][0]['instances']==[context['instance']]
            assert p['tasks'][0]['seeds']==[context['seed']] and p['tasks'][0]['split']=='train'
            plans.append(path)
        save(out/'spec.json',dict(version='combat_native_continuation_probe_v1' if continuation else 'combat_native_movement_probe_v1',prepared=str(a.prepared.resolve()),
            prepared_report_sha256=seal['report_sha256'],context=context,variants=variants,inputs=pins,
            diagnostic_only=True,ordinary_training_eligible=False,training_performed=False,quality_confirmation=False,
            max_instances=16,timescale=2,seed_reuse_intentional=True,fresh_reference=a.fresh_reference,promotion=False))
        shell=shutil.which('pwsh');assert shell
        quote=lambda p:"'"+str(p).replace("'","''")+"'"
        command='& '+quote(repo/'scripts/run_registered_combat_pool.ps1')+' -Plans @('+','.join(map(quote,plans))+') -MaxInstances 16 -Port '+str(a.port)+' -OutputRoot '+quote(out/'pool')
        save(out/'progress.json',dict(stage='native-movement-smoke',cases=len(variants)))
        run('native-pool',[shell,'-NoProfile','-Command',command])
        pool=read(out/'pool/report.json')
        assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==len(pool['jobs'])==len(variants)
        assert all(not j['error'] for j in pool['jobs'])
        exporter=out/'q2ppo-data.exe';run('build-exporter',['go','build','-o',exporter,'./cmd/q2ppo-data'])
        save(out/'progress.json',dict(stage='native-export',cases=len(variants)))

        def export(job):
            index=job['plan_index'];v=variants[index];capture=Path(job['root']);directory=out/f'case-{index:02d}';directory.mkdir()
            r=read(pin(capture/'report.json'));m=read(pin(capture/'manifest.json')); result=r['results'][0]
            assert r['capture_complete'] and r['provenance_valid'] and result['capture_valid'] and result['dispatch_valid']
            assert result['seed_confirmed'] and result['seed']==context['seed'] and result['generated_fixture']==context['instance']
            assert m['model_weights_sha256'].lower()==v['model_sha256']
            run(f'export-{index:02d}',[exporter,'--batch',capture,'--model',v['model'],'--cuda-only','--out',directory/'native'])
            verify_memory_export(Path(result['root'])/'bot.jsonl',directory/'native/replay-0')
            compact_closed_exports(directory/'native')
            return dict(model=v['model'],model_sha256=v['model_sha256'],data=str(directory/'native'),out=str(directory/'cuda'),
                native_report_sha256=sha(directory/'native/report.json'))

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as workers:
            tasks=list(workers.map(export,sorted(pool['jobs'],key=lambda j:j['plan_index'])))
        request=out/'cuda-request.json';save(request,dict(version='combat_cuda_finalize_batch_v1',compact_finalized_streams=True,tasks=tasks))
        save(out/'progress.json',dict(stage='cuda-movement-smoke',cases=len(variants)))
        run('cuda-batch',[sys.executable,repo/'scripts/finalize_combat_cuda_batch.py','--request',request,'--receipt-root',out/'cuda-batch'])
        receipt=read(out/'cuda-batch/report.json');assert receipt['state']=='complete' and receipt['device']=='cuda' and len(receipt['tasks'])==len(variants)
        for path,h in pins.items():assert sha(path)==h,path
        save(out/'complete.json',dict(state='complete',version='combat_native_continuation_probe_v1' if continuation else 'combat_native_movement_probe_v1',native_cases=len(variants),cuda_cases=len(variants),
            spec_sha256=sha(out/'spec.json'),pool_sha256=sha(out/'pool/report.json'),cuda_receipt_sha256=sha(out/'cuda-batch/report.json'),
            diagnostic_only=True,ordinary_training_eligible=False,training_performed=False,promotion=False,
            scope='Technical native/CUDA capture only; baseline/prefix/draw acceptance and consequence labels remain separate.'))
        save(out/'progress.json',dict(stage='complete',native_cases=len(variants),cuda_cases=len(variants)))
        print(out/'complete.json')
    except BaseException as error:
        save(out/'progress.json',dict(stage='failed',error=str(error),exception=type(error).__name__));raise


if __name__=='__main__':main()

"""Run a sealed 25-head TRAIN diagnostic grid; no fitting or quality claims."""
import argparse
import concurrent.futures
import os
from pathlib import Path
import shutil
import subprocess
import sys
from evaluate_registered_combat_models import read, save, sha
from process_combat_architecture_pool import compact_closed_exports
from process_coop_learning_pool_cuda import verify_memory_export


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('prepared','out'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--port',type=int,default=44800)
    ap.add_argument('--planner',type=Path);a=ap.parse_args()
    repo=Path(__file__).resolve().parents[1];out=a.out.resolve();assert not out.exists()
    assert shutil.disk_usage('F:/').free>=32*1024**3
    os.environ['GOCACHE']=str(repo/'workspace/build/gocache');os.environ['GOTOOLCHAIN']='auto';pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    seal=read(pin(a.prepared/'complete.json'));prepared=read(pin(a.prepared/'report.json',seal['report_sha256']))
    one_frame=prepared['version']=='combat_one_frame_aim_preparation_v1'
    paired=one_frame or prepared['version']=='combat_aim_pair_context_preparation_v1'
    assert prepared['state']=='complete' and prepared['device']=='cuda' and (paired or prepared['diagnostic_only'])
    assert not prepared['training_performed'] and not prepared['validation_used']
    if paired:
        if one_frame:assert len(prepared['contexts'])==1 and len(prepared['variants']) in (2,5) and a.planner
        else:assert len(prepared['contexts']) in (4,12) and len(prepared['variants'])==5*len(prepared['contexts'])
        assert len({v['context_id'] for v in prepared['variants']})==len(prepared['contexts'])
    else:assert [v['category'] for v in prepared['variants']]==list(range(25))
    case_count=len(prepared['variants'])
    for p,h in prepared['inputs'].items():pin(p,h)
    reference=Path(prepared['baseline'] if paired else prepared['reference']);closed=read(pin(reference/'complete.json'))
    refspec=read(pin(reference/'spec.json',closed['spec_sha256']));registry=pin(refspec['registry'])
    original=read(pin(reference/('baseline-plan.json' if paired else 'source-plan.json')))
    assert all(t['split']=='train' for t in original['tasks'])
    for v in prepared['variants']:pin(v['model'],v['model_sha256'])
    out.mkdir(parents=True)
    def run(label,args):
        with (out/(label+'.log')).open('w',encoding='utf-8') as stream:
            r=subprocess.run(list(map(str,args)),cwd=repo,stdout=stream,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert r.returncode==0,str(out/(label+'.log'))
    try:
        save(out/'progress.json',dict(stage='planning'));plans=[]
        planner=a.planner.resolve() if a.planner else repo/'workspace/build/q2episode-joint-aim.exe';assert planner.is_file();pin(planner)
        for k,v in enumerate(prepared['variants']):
            fixture=[v['instance']] if paired else original['tasks'][0]['instances']
            episode=v['episode'] if paired else prepared['episode'];offset=v['offset'] if paired else prepared['offset']
            seed=v['seed'] if paired else prepared['seed'];plan=out/f'plan-{k:02d}.json'
            run(f'plan-{k:02d}',[planner,'--registry',registry,'--root',repo,'--episodes',episode,
                '--split','train','--mode','learned','--model',v['model'],'--count',1,'--seed-offset',offset,
                '--checked-wall-slide','--out',plan,'--artifacts',out/f'captures-{k:02d}'])
            value=read(plan);assert len(value['tasks'])==1 and value['tasks'][0]['instances']==fixture
            assert value['tasks'][0]['split']=='train' and value['tasks'][0]['seeds']==[seed]
            plans.append(plan)
        version='combat_native_aim_pairs_probe_v1' if paired else 'combat_joint_aim_grid_probe_v1'
        save(out/'spec.json',dict(version=version,split='train',prepared=str(a.prepared.resolve()),
            prepared_report_sha256=seal['report_sha256'],variants=prepared['variants'],seed=prepared.get('seed'),frame=prepared.get('frame'),
            reference=str(reference.resolve()),max_instances=16,timescale=2,inputs=pins,training_performed=False,
            quality_confirmation=False,promotion=False,seed_reuse_intentional=True))
        shell=shutil.which('pwsh');assert shell
        quote=lambda p:"'"+str(p).replace("'","''")+"'"
        command='& '+quote(repo/'scripts/run_registered_combat_pool.ps1')+' -Plans @('+','.join(map(quote,plans))+') -MaxInstances 16 -Port '+str(a.port)+' -OutputRoot '+quote(out/'pool')
        save(out/'progress.json',dict(stage='native-grid',cases=case_count))
        run('native-pool',[shell,'-NoProfile','-Command',command])
        pool=read(out/'pool/report.json');assert pool['state']=='complete' and pool['source_unchanged']
        assert pool['usable_captures']==len(pool['jobs'])==case_count and all(not j['error'] for j in pool['jobs'])
        pin(out/'pool/report.json');assert {j['plan_index'] for j in pool['jobs']}==set(range(case_count))
        exporter=out/'q2ppo-data.exe';run('build-exporter',['go','build','-o',exporter,'./cmd/q2ppo-data'])
        save(out/'progress.json',dict(stage='native-export',cases=case_count))
        def export(job):
            k=job['plan_index'];v=prepared['variants'][k];folder=Path(job['root']);directory=out/f'case-{k:02d}'
            directory.mkdir();capture=read(pin(folder/'report.json'));manifest=read(pin(folder/'manifest.json'))
            assert capture['capture_complete'] and capture['provenance_valid'] and len(capture['results'])==1
            result=capture['results'][0]
            assert result['capture_valid'] and result['dispatch_valid'] and result['seed_confirmed']
            assert result['seed']==v.get('seed',prepared.get('seed')) and result['generated_fixture']['split']=='train'
            assert manifest['model_weights_sha256'].lower()==v['model_sha256']
            run(f'export-{k:02d}',[exporter,'--batch',folder,'--model',v['model'],'--cuda-only','--out',directory/'native'])
            verify_memory_export(Path(result['root'])/'bot.jsonl',directory/'native/replay-0')
            compact_closed_exports(directory/'native')
            return dict(model=v['model'],model_sha256=v['model_sha256'],data=str(directory/'native'),out=str(directory/'cuda'),
                native_report_sha256=sha(directory/'native/report.json'))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as workers:
            tasks=list(workers.map(export,sorted(pool['jobs'],key=lambda j:j['plan_index'])))
        receipts=[]
        for offset in range(0,case_count,8):
            group=tasks[offset:offset+8];index=len(receipts);request=out/f'cuda-request-{index}.json';receipt=out/f'cuda-batch-{index}'
            save(request,dict(version='combat_cuda_finalize_batch_v1',compact_finalized_streams=True,tasks=group))
            save(out/'progress.json',dict(stage='cuda-grid',first_category=offset,cases=len(group)))
            run(f'cuda-batch-{index}',[sys.executable,repo/'scripts/finalize_combat_cuda_batch.py','--request',request,'--receipt-root',receipt])
            value=read(receipt/'report.json');assert value['state']=='complete' and value['device']=='cuda'
            assert value['request_sha256']==sha(request) and len(value['tasks'])==len(group)
            receipts.append(dict(report=str(receipt/'report.json'),report_sha256=sha(receipt/'report.json')))
        for p,h in pins.items():assert sha(Path(p))==h,p
        save(out/'complete.json',dict(state='complete',version=version,native_cases=case_count,cuda_cases=case_count,
            spec_sha256=sha(out/'spec.json'),pool_sha256=sha(out/'pool/report.json'),cuda_receipts=receipts,
            training_performed=False,quality_confirmation=False,promotion=False,
            scope='Technical captures and CUDA replay complete. Common-state acceptance and native shot analysis remain separate.'))
        save(out/'progress.json',dict(stage='complete',native_cases=case_count,cuda_cases=case_count));print(out/'complete.json')
    except BaseException as e:
        save(out/'progress.json',dict(stage='failed',error=str(e),exception=type(e).__name__));raise


if __name__=='__main__':main()

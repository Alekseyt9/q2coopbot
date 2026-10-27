"""Fresh registered TRAIN collection for the CUDA movement-option optimizer."""
import argparse
import concurrent.futures
import os
from pathlib import Path
import shutil
import sys

from process_combat_architecture_pool import read, save, sha
from run_combat_full_joint_series import command, now


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'curriculum', 'engine', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--seed-offset', type=int, required=True)
    p.add_argument('--port', type=int, default=55400)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    a.out = a.out.resolve(); a.model = a.model.resolve()
    assert not a.out.exists() and a.out.is_relative_to(repo/'workspace/artifacts')
    assert shutil.disk_usage('F:/').free > 32*1024**3
    model = read(a.model)
    assert model['movement_options']['enabled']
    assert model['action_distribution'] == 'combat_joint_aim_movement_options_v3'
    assert model['combat_execution_contract'] == 'stock_weapons_fire_v1'
    curriculum = read(a.curriculum)
    ids = curriculum['train_ids']; assert len(ids) == len(set(ids)) == 53
    assert a.seed_offset > 0
    env = dict(os.environ, GOCACHE=str(repo/'workspace/build/gocache'),
        Q2COOPBOT_DIAGNOSTIC_ENGINE_ROOT=str(a.engine.resolve()))
    a.out.mkdir(parents=True)
    pins = {str(a.model): sha(a.model), str(a.curriculum.resolve()): sha(a.curriculum)}
    for directory, pattern in (('scripts','*.py'),('scripts','*.ps1'),('cmd','*.go'),('internal','*.go')):
        for path in (repo/directory).rglob(pattern): pins[str(path)] = sha(path)
    save(a.out/'protocol.json', dict(version='combat_movement_options_train_collection_v1',
        model=str(a.model), model_sha256=sha(a.model), split='train', train_ids=ids,
        seed_offset=a.seed_offset, max_instances=16, timescale=2, inputs=pins,
        training_performed=False, promotion=False))
    save(a.out/'runner.json', dict(pid=os.getpid(), started_utc=now(), executable=sys.executable))
    def progress(stage):
        save(a.out/'progress.json', dict(stage=stage, utc=now(), runner_pid=os.getpid()))
        print(stage, flush=True)
    try:
        progress('building')
        for name in ('q2episode','q2ppo-data'):
            command(['go','build','-o',a.out/(name+'.exe'),'./cmd/'+name],a.out,name+'-build',env)
        progress('planning')
        command([a.out/'q2episode.exe','--root',repo,'--episodes',','.join(ids),
            '--mode','learned','--model',a.model,'--execution-contract','stock_weapons_fire_v1',
            '--split','train','--count','1','--seed-offset',a.seed_offset,'--checked-wall-slide',
            '--out',a.out/'plan.json','--artifacts',a.out/'captures'],a.out,'plan',env)
        plan=read(a.out/'plan.json')
        assert len(plan['tasks']) == 53 and all(t['split']=='train' for t in plan['tasks'])
        seeds=[i['engine_seed'] for t in plan['tasks'] for i in t['instances']]
        assert len(seeds)==len(set(seeds))==53
        progress('collecting')
        command([shutil.which('pwsh.exe'),'-NoProfile','-File',repo/'scripts/run_registered_combat_pool.ps1',
            '-Plans',a.out/'plan.json','-MaxInstances','16','-Port',a.port,'-OutputRoot',a.out/'pool'],a.out,'collect',env)
        pool=read(a.out/'pool/report.json')
        assert pool['state']=='complete' and pool['source_unchanged']
        assert len(pool['jobs'])==53 and all(not j['error'] for j in pool['jobs'])
        progress('exporting')
        def export(job):
            index=job['task_index']; native=a.out/f'export-{index:02d}-native'
            cuda=a.out/f'export-{index:02d}-cuda'
            command([a.out/'q2ppo-data.exe','--batch',job['root'],'--model',a.model,
                '--cuda-only','--out',native],a.out,f'export-{index:02d}',env)
            return dict(seed=job['seed'],capture_root=job['root'],data=str(cuda)), dict(
                model=str(a.model),model_sha256=sha(a.model),data=str(native),out=str(cuda),
                native_report_sha256=sha(native/'report.json'))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results=list(executor.map(export,sorted(pool['jobs'],key=lambda j:j['task_index'])))
        save(a.out/'cuda-request.json',dict(version='combat_cuda_finalize_batch_v1',
            compact_finalized_streams=True,tasks=[r[1] for r in results]))
        progress('cuda-replay')
        command([sys.executable,repo/'scripts/finalize_combat_cuda_batch.py','--request',a.out/'cuda-request.json',
            '--receipt-root',a.out/'cuda-batch'],a.out,'cuda',env)
        assert all(sha(Path(path))==digest for path,digest in pins.items()), 'Collection sources changed'
        save(a.out/'train-request.json',dict(version='combat_movement_options_train_request_v1',split='train',
            plan=str(a.out/'plan.json'),plan_sha256=sha(a.out/'plan.json'),
            pool_report=str(a.out/'pool/report.json'),pool_report_sha256=sha(a.out/'pool/report.json'),
            cases=[r[0] for r in results]))
        save(a.out/'complete.json',dict(state='complete',native_cases=53,cuda_cases=53,
            train_request_sha256=sha(a.out/'train-request.json'),training_performed=False,promotion=False))
        progress('complete')
    except BaseException as error:
        save(a.out/'failure.json',dict(state='failed',error=str(error),utc=now()))
        raise


if __name__ == '__main__': main()

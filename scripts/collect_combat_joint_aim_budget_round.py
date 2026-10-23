"""Collect B/G/S own-policy TRAIN with matched fresh seeds in one native queue."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from evaluate_registered_combat_models import read, save, sha


def historical_plans(base):
    paths = set(base.glob('*plan*.json')) | set(base.glob('*/*plan*.json'))
    for pattern in ('*/round-*/*plan*.json', '*/preflight/*plan*.json', '*/corpus/*plan*.json',
                    '*/evaluation/*plan*.json', '*/screen/*plan*.json'):
        paths.update(base.glob(pattern))
    # Older comparison specs already pin deeper per-variant plans. Retain their
    # history without walking millions of runtime asset names.
    for p in sorted(set(base.glob('*spec*.json')) | set(base.glob('*/*spec*.json'))):
        value = read(p)
        if isinstance(value, dict) and isinstance(value.get('historical_plans'), list):
            paths.update(Path(name).resolve() for name in value['historical_plans'] if isinstance(name, str))
    result = []
    historical_plans.missing_references = []
    for p in sorted(paths):
        if not p.is_file():
            historical_plans.missing_references.append(str(p))
            continue
        value = read(p)
        if isinstance(value, dict) and isinstance(value.get('tasks'), list): result.append((p, value))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('previous', 'registry', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--round', type=int, choices=(2, 3, 4), required=True)
    ap.add_argument('--port', type=int, default=44600)
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    previous, registry, out = (p.resolve() for p in (a.previous, a.registry, a.out))
    assert not out.exists() and shutil.disk_usage('F:/').free >= 32*1024**3
    os.environ['GOCACHE'] = str(repo/'workspace/build/gocache'); os.environ['GOTOOLCHAIN'] = 'auto'
    from train_combat_bc import torch, training_devices
    training_devices()
    descriptors = {read(registry.parent/f)['id']: read(registry.parent/f) for f in read(registry)['files']}
    names = sorted(descriptors); assert len(names) == 8
    assert all(e['status']=='runnable' and e['ppo_trainable'] and e['timescale']==2 for e in descriptors.values())
    history = historical_plans(repo/'workspace/artifacts')
    used = {s for _, plan in history for task in plan['tasks'] for s in task.get('seeds', [])}
    offset = 0
    while True:
        assert all(offset+4 <= e['splits']['train']['count'] for e in descriptors.values())
        seeds = {e['splits']['train']['start']+offset+i for e in descriptors.values() for i in range(4)}
        if not seeds & used: break
        offset += 4
    assert len(seeds) == 32
    sources = {}; configs = {}; pins = {str(registry):sha(registry)}
    for arm in ('B', 'G', 'S'):
        source = previous/arm; seal = read(source/'complete.json')
        for name, key in (('weights.json','weights_sha256'), ('checkpoint.pt','checkpoint_sha256'), ('report.json','report_sha256')):
            p = source/name; assert sha(p)==seal[key]; pins[str(p)] = seal[key]
        report = read(source/'report.json')
        assert report['version']=='combat_joint_aim_near_budget_v1' and report['arm']==arm and report['round']==a.round-1
        assert report['device']=='cuda' and not read(source/'weights.json')['deterministic']
        cp = torch.load(source/'checkpoint.pt', map_location='cuda', weights_only=True)
        assert cp['weights_sha256']==seal['weights_sha256'] and cp['joint_aim_budget']['round']==a.round-1 and cp['joint_aim_budget']['arm']==arm
        config = cp['config']; expected = config['objective_reward_sha256']
        assert all(sha(repo/e['recipe']['reward_config'])==expected for e in descriptors.values())
        for e in descriptors.values(): pins[str((repo/e['recipe']['reward_config']).resolve())] = expected
        configs[arm] = config; sources[arm] = source
        del cp
    assert configs['B']==configs['G']==configs['S']
    out.mkdir(parents=True)

    def run(label, command):
        with (out/(label+'.log')).open('w',encoding='utf-8') as log:
            result = subprocess.run(list(map(str, command)), cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert result.returncode==0, str(out/(label+'.log'))

    try:
        save(out/'progress.json',dict(stage='preparing', round=a.round, cases=96))
        planner = out/'q2episode.exe'; run('build-planner',['go','build','-o',planner,'./cmd/q2episode'])
        plans = []; protocols = {}
        for arm in ('B','G','S'):
            corpus = out/arm; corpus.mkdir()
            config_path = corpus/'training-config.json'; save(config_path,configs[arm]); pins[str(config_path)] = sha(config_path)
            plan = out/(arm+'-plan.json')
            run('plan-'+arm,[planner,'--registry',registry,'--root',repo,'--episodes',','.join(names),'--split','train',
                '--mode','learned','--model',sources[arm]/'weights.json','--count',4,'--seed-offset',offset,
                '--checked-wall-slide','--out',plan,'--artifacts',corpus/'captures-train'])
            frozen = read(plan); actual = [s for t in frozen['tasks'] for s in t['seeds']]
            assert len(actual)==len(set(actual))==32 and set(actual)==seeds
            assert all(t['split']=='train' and t['modes']==['learned'] for t in frozen['tasks'])
            assert frozen['model_sha256']==sha(sources[arm]/'weights.json') and not frozen.get('diagnostic_nominal_aim')
            protocols[arm] = dict(version='combat_aim_feedback_corpus_v1', state='prepared',source=str(sources[arm]),
                assignments=[dict(label='train',split='train',offset=offset,count=4,seeds=actual,plan=str(plan),plan_sha256=sha(plan))],
                train_only=True, ppo_training_objective=dict(config=str(config_path),config_sha256=sha(config_path),
                    reward_sha256=configs[arm]['objective_reward_sha256'],checkpoint_sha256=sha(sources[arm]/'checkpoint.pt')),
                max_instances=16,timescale=2,training_performed=False, round=a.round,arm=arm,
                history=[dict(path=str(p),sha256=sha(p)) for p,_ in history],
                unavailable_historical_references=historical_plans.missing_references,
                shared_pool=str(out/'pool'),paired_conditions_across_arms=True,
                scope='Same fresh TRAIN fixture seeds across independent B/G/S models; every trajectory belongs only to its behavior policy.')
            save(corpus/'protocol.json',protocols[arm]); plans.append(plan)
        save(out/'protocol.json',dict(version='combat_joint_aim_budget_round_collection_v1',state='prepared',round=a.round,
            arms=list(sources),sources={k:str(v) for k,v in sources.items()},plans=[str(p) for p in plans],
            inputs=pins,offset=offset,seeds=sorted(seeds),unique_seed_conditions=32,native_cases=96,max_instances=16,timescale=2,
            minimum_free_gib=32,training_performed=False,paired_seeds=True))
        shell = shutil.which('pwsh'); assert shell
        literal = lambda p: "'"+str(p).replace("'","''")+"'"
        command = '& '+literal(repo/'scripts/run_registered_combat_pool.ps1')+' -Plans @('+','.join(map(literal,plans))+') -MaxInstances 16 -Port '+str(a.port)+' -OutputRoot '+literal(out/'pool')
        assert shutil.disk_usage('F:/').free>=32*1024**3
        save(out/'progress.json',dict(stage='native-collection',round=a.round,cases=96))
        run('native-pool',[shell,'-NoProfile','-Command',command])
        pool = read(out/'pool/report.json')
        assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==len(pool['jobs'])==96
        for index, arm in enumerate(('B','G','S')):
            corpus = out/arm
            save(out/'progress.json',dict(stage='cuda-replay',round=a.round,arm=arm))
            run('processing-'+arm,[sys.executable,repo/'scripts/process_registered_combat_pool_cuda.py',
                '--pool',out/'pool','--plan-index',index,'--out',corpus/'processing-train'])
            report = read(corpus/'processing-train/report.json')
            assert report['state']=='complete' and report['device']=='cuda' and not report['training_performed'] and len(report['cases'])==32
            assert {c['seed'] for c in report['cases']}==seeds
            # This immutable report describes the shared 96-job pool. Per-corpus
            # processing manifests explicitly select its own 32 job IDs.
            (corpus/'pool').mkdir(); shutil.copy2(out/'pool/report.json',corpus/'pool/report.json')
            save(corpus/'complete.json',dict(state='complete',version='combat_aim_feedback_corpus_v1',native_cases=32,cuda_cases=32,
                protocol_sha256=sha(corpus/'protocol.json'),pool_sha256=sha(corpus/'pool/report.json'),
                processed=[dict(label='train',cases=32,rows=report['rows'],report=str(corpus/'processing-train/report.json'),sha256=sha(corpus/'processing-train/report.json'))],
                training_performed=False,promotion=False,shared_native_pool_cases=96))
            save(out/'progress.json',dict(stage='credit-audit',round=a.round,arm=arm))
            run('native-credit-'+arm,[sys.executable,repo/'scripts/audit_combat_native_credit.py','--corpus',corpus,'--out',corpus/'native-credit'])
            run('advantage-credit-'+arm,[sys.executable,repo/'scripts/audit_combat_advantage_credit.py','--corpus',corpus,
                '--native-audit',corpus/'native-credit','--out',corpus/'advantage-credit'])
        for path,digest in pins.items(): assert sha(Path(path))==digest,'Input changed: '+path
        save(out/'complete.json',dict(state='complete',version='combat_joint_aim_budget_round_collection_v1',round=a.round,
            native_cases=96,cuda_cases=96,paired_seed_conditions=32,protocol_sha256=sha(out/'protocol.json'),
            pool_sha256=sha(out/'pool/report.json'),training_performed=False,promotion=False,
            corpora={arm:dict(path=str(out/arm),complete_sha256=sha(out/arm/'complete.json'),native_credit_sha256=sha(out/arm/'native-credit/report.json'),
                advantage_credit_sha256=sha(out/arm/'advantage-credit/report.json')) for arm in ('B','G','S')},script_sha256=sha(__file__)))
        save(out/'progress.json',dict(stage='complete',round=a.round,native_cases=96,cuda_cases=96))
        print(out/'complete.json',flush=True)
    except BaseException as error:
        save(out/'progress.json',dict(stage='failed',round=a.round,error=str(error),exception=type(error).__name__))
        raise


if __name__=='__main__':main()

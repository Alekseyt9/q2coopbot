"""Collect second-round B/G own-policy TRAIN in one dynamic 16-slot pool."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from evaluate_registered_combat_models import read, save, sha
from collect_combat_joint_aim_budget_round import historical_plans
from combat_geometry_ppo import TRAINING_VERSION
from train_combat_bc import torch, training_devices


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('previous', 'registry', 'protocol', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--port', type=int, default=47300)
    a = ap.parse_args(); repo = Path(__file__).resolve().parents[1]
    out, registry, previous = (p.resolve() for p in (a.out, a.registry, a.previous))
    assert not out.exists() and shutil.disk_usage('F:/').free >= 32*1024**3
    training_devices()
    os.environ['GOCACHE'] = str(repo/'workspace/build/gocache'); os.environ['GOTOOLCHAIN'] = 'auto'
    descriptors = {read(registry.parent/f)['id']: read(registry.parent/f) for f in read(registry)['files']}
    names = sorted(descriptors); assert len(names) == 8
    assert all(e['status'] == 'runnable' and e['ppo_trainable'] and e['timescale'] == 2 for e in descriptors.values())
    assert read(a.protocol)['version'] == TRAINING_VERSION
    history = historical_plans(repo/'workspace/artifacts')
    used = {s for _, p in history for t in p['tasks'] for s in t.get('seeds', [])}
    sources = {}; configs = {}; pins = {str(registry): sha(registry), str(a.protocol.resolve()): sha(a.protocol)}
    for arm in ('B', 'G'):
        source = previous/arm; seal = read(source/'complete.json')
        for name, key in (('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'), ('report.json', 'report_sha256')):
            p = source/name; assert sha(p) == seal[key]; pins[str(p)] = seal[key]
        r = read(source/'report.json')
        assert r['version'] == TRAINING_VERSION and r['device'] == 'cuda' and r['arm'] == arm and r['round'] == 1
        assert r['experiment_protocol_sha256'] == sha(a.protocol)
        cp = torch.load(source/'checkpoint.pt', map_location='cuda', weights_only=True)
        g = cp['geometry_aim_training']
        assert cp['weights_sha256'] == seal['weights_sha256'] and g['arm'] == arm and g['updates_completed'] == 1
        assert g['protocol_sha256'] == sha(a.protocol)
        configs[arm] = cp['config']; sources[arm] = source
        expected = configs[arm]['objective_reward_sha256']
        for e in descriptors.values():
            p = (repo/e['recipe']['reward_config']).resolve(); assert sha(p) == expected; pins[str(p)] = expected
        del cp
    assert configs['B'] == configs['G']
    out.mkdir(parents=True)

    def run(label, command):
        with (out/(label+'.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(list(map(str, command)), cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert result.returncode == 0, str(out/(label+'.log'))

    try:
        save(out/'progress.json', dict(stage='preparing', cases=64))
        planner = out/'q2episode.exe'; run('build-planner', ['go', 'build', '-o', planner, './cmd/q2episode'])
        plans = []; all_seeds = set()
        for arm in ('B', 'G'):
            offset = 0
            while True:
                assert all(offset+4 <= e['splits']['train']['count'] for e in descriptors.values())
                seeds = {e['splits']['train']['start']+offset+i for e in descriptors.values() for i in range(4)}
                if not seeds & used: break
                offset += 4
            assert len(seeds) == 32; used.update(seeds); all_seeds.update(seeds)
            corpus = out/arm; corpus.mkdir()
            config_path = corpus/'training-config.json'; save(config_path, configs[arm]); pins[str(config_path)] = sha(config_path)
            plan = out/(arm+'-plan.json')
            run('plan-'+arm, [planner, '--registry', registry, '--root', repo, '--episodes', ','.join(names), '--split', 'train',
                '--mode', 'learned', '--model', sources[arm]/'weights.json', '--count', 4, '--seed-offset', offset,
                '--checked-wall-slide', '--out', plan, '--artifacts', corpus/'captures-train'])
            frozen = read(plan); actual = [s for t in frozen['tasks'] for s in t['seeds']]
            assert len(actual) == len(set(actual)) == 32 and set(actual) == seeds
            assert frozen['model_sha256'] == sha(sources[arm]/'weights.json') and not frozen.get('diagnostic_nominal_aim')
            assert all(t['split'] == 'train' and t['modes'] == ['learned'] for t in frozen['tasks'])
            save(corpus/'protocol.json', dict(version='combat_aim_feedback_corpus_v1', source=str(sources[arm]),
                assignments=[dict(label='train', split='train', offset=offset, count=4, seeds=actual, plan=str(plan), plan_sha256=sha(plan))],
                train_only=True, diagnostic_intervention=False,
                ppo_training_objective=dict(config=str(config_path), config_sha256=sha(config_path),
                    reward_sha256=configs[arm]['objective_reward_sha256'], checkpoint_sha256=sha(sources[arm]/'checkpoint.pt')),
                max_instances=16, timescale=2, training_performed=False, arm=arm, round=2,
                experiment_protocol_sha256=sha(a.protocol), history=[dict(path=str(p), sha256=sha(p)) for p, _ in history],
                unavailable_historical_references=historical_plans.missing_references,
                shared_pool=str(out/'pool'), scope='Fresh disjoint TRAIN seeds per behavior policy; one dynamic queue.'))
            plans.append(plan)
        assert len(all_seeds) == 64
        save(out/'protocol.json', dict(version=TRAINING_VERSION, round=2, plans=[str(p) for p in plans], inputs=pins,
            unique_seeds=sorted(all_seeds), native_cases=64, max_instances=16, timescale=2, experiment_protocol_sha256=sha(a.protocol)))
        shell = shutil.which('pwsh'); assert shell
        literal = lambda p: "'"+str(p).replace("'", "''")+"'"
        command = '& '+literal(repo/'scripts/run_registered_combat_pool.ps1')+' -Plans @('+','.join(map(literal, plans))+') -MaxInstances 16 -Port '+str(a.port)+' -OutputRoot '+literal(out/'pool')
        save(out/'progress.json', dict(stage='native-collection', cases=64))
        run('native-pool', [shell, '-NoProfile', '-Command', command])
        pool = read(out/'pool/report.json')
        assert pool['state'] == 'complete' and pool['source_unchanged'] and pool['usable_captures'] == len(pool['jobs']) == 64
        for index, arm in enumerate(('B', 'G')):
            corpus = out/arm
            save(out/'progress.json', dict(stage='cuda-replay', arm=arm))
            run('processing-'+arm, [sys.executable, repo/'scripts/process_registered_combat_pool_cuda.py',
                '--pool', out/'pool', '--plan-index', index, '--out', corpus/'processing-train'])
            r = read(corpus/'processing-train/report.json')
            assert r['state'] == 'complete' and r['device'] == 'cuda' and not r['training_performed'] and len(r['cases']) == 32
            assert {c['seed'] for c in r['cases']} == set(read(corpus/'protocol.json')['assignments'][0]['seeds'])
            (corpus/'pool').mkdir(); shutil.copy2(out/'pool/report.json', corpus/'pool/report.json')
            save(corpus/'complete.json', dict(state='complete', version='combat_aim_feedback_corpus_v1', native_cases=32, cuda_cases=32,
                protocol_sha256=sha(corpus/'protocol.json'), pool_sha256=sha(corpus/'pool/report.json'), training_performed=False, promotion=False,
                shared_native_pool_cases=64, processed=[dict(label='train', cases=32, rows=r['rows'], report=str(corpus/'processing-train/report.json'), sha256=sha(corpus/'processing-train/report.json'))]))
            save(out/'progress.json', dict(stage='credit-audit', arm=arm))
            run('native-credit-'+arm, [sys.executable, repo/'scripts/audit_combat_native_credit.py', '--corpus', corpus, '--out', corpus/'native-credit'])
            run('advantage-credit-'+arm, [sys.executable, repo/'scripts/audit_combat_advantage_credit.py', '--corpus', corpus,
                '--native-audit', corpus/'native-credit', '--out', corpus/'advantage-credit'])
        for path, digest in pins.items(): assert sha(Path(path)) == digest, 'Input changed: '+path
        save(out/'complete.json', dict(state='complete', version=TRAINING_VERSION, round=2, native_cases=64, cuda_cases=64,
            protocol_sha256=sha(out/'protocol.json'), pool_sha256=sha(out/'pool/report.json'), training_performed=False, promotion=False,
            corpora={arm: dict(path=str(out/arm), complete_sha256=sha(out/arm/'complete.json')) for arm in ('B', 'G')}))
        save(out/'progress.json', dict(stage='complete', native_cases=64, cuda_cases=64))
    except BaseException as error:
        save(out/'progress.json', dict(stage='failed', error=str(error))); raise


if __name__ == '__main__': main()

"""Four known-seed technical replays; excluded from PPO and quality estimates."""
import argparse
import copy
from pathlib import Path
import shutil
import subprocess
import sys
from evaluate_registered_combat_models import read, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('registry', 'source_corpus', 'failed_corpus', 'initializer', 'prior_g', 'out'):
        ap.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    ap.add_argument('--port', type=int, default=48300)
    a = ap.parse_args(); assert not a.out.exists()
    repo = Path(__file__).resolve().parents[1]
    sources = [a.source_corpus.resolve(), a.failed_corpus.resolve()]
    initial_pool, failed_pool = (read(p/'pool/report.json') for p in sources)
    descriptors = {e['id']:e for e in (read(a.registry.parent/f) for f in read(a.registry)['files'])}
    selection = {}
    for job in initial_pool['jobs']:
        capture = read(Path(job['root'])/'report.json')['results'][0]
        kind = 'goal' if capture['goal_stop'] else 'death' if capture['death_stop'] else 'horizon'
        if kind not in selection: selection[kind] = (job, capture, a.initializer.resolve())
    job = next(j for j in failed_pool['jobs'] if j['seed'] == 1174065121)
    capture = read(Path(job['root'])/'report.json')['results'][0]
    assert not capture['goal_stop'] and not capture['death_stop']
    # A separately declared shorter technical horizon, same initial fight,
    # stops before the previously observed visibility timeout. Never TRAIN.
    selection['horizon'] = (job, capture, a.prior_g.resolve())
    selection['unfinished'] = (job, capture, a.prior_g.resolve())
    assert set(selection) == {'goal', 'death', 'horizon', 'unfinished'}
    assert shutil.disk_usage('F:/').free >= 32*1024**3
    a.out.mkdir(parents=True)
    pins = {str(p.resolve()):sha(p) for p in (a.registry, sources[0]/'pool/report.json', sources[1]/'pool/report.json')}

    def run(label, command):
        with (a.out/(label+'.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(list(map(str, command)), cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert result.returncode == 0, str(a.out/(label+'.log'))

    import os
    os.environ['GOCACHE'] = str(repo/'workspace/build/gocache'); os.environ['GOTOOLCHAIN'] = 'auto'
    planner = a.out/'q2episode.exe'; run('build-planner', ['go', 'build', '-o', planner, './cmd/q2episode'])
    # Deserialization of training metadata on CUDA, no neural evaluation here.
    import torch
    assert torch.cuda.is_available()
    cp = torch.load(a.initializer/'checkpoint.pt', map_location='cuda', weights_only=True)
    config = copy.deepcopy(cp['config']); del cp
    plans = []; labels = list(selection)
    for label, (original_job, original_capture, model_root) in selection.items():
        episode = original_capture['generated_fixture']['episode_id']; descriptor = descriptors[episode]
        seed = original_job['seed']; offset = seed-descriptor['splits']['train']['start']
        assert 0 <= offset < descriptor['splits']['train']['count']
        assert sha(repo/descriptor['recipe']['reward_config']) == config['objective_reward_sha256']
        model = model_root/'weights.json'; pins[str(model)] = sha(model)
        target = a.out/label; target.mkdir(); save(target/'training-config.json', config)
        registry = a.registry.resolve()
        if label == 'horizon':
            local = target/'horizon-registry'; local.mkdir()
            shortened = copy.deepcopy(descriptor); shortened['game_frames'] = 150; shortened['revision'] += 1
            save(local/(episode+'.json'), shortened)
            save(local/'index.json', dict(version=1, files=[episode+'.json']))
            registry = (local/'index.json').resolve()
        plan = a.out/(label+'-plan.json')
        run('plan-'+label, [planner, '--registry', registry, '--root', repo, '--episodes', episode,
            '--split', 'train', '--mode', 'learned', '--model', model, '--count', 1, '--seed-offset', offset,
            '--checked-wall-slide', '--out', plan, '--artifacts', target/'captures'])
        p = read(plan); assert len(p['tasks']) == 1 and p['tasks'][0]['seeds'] == [seed]
        save(target/'protocol.json', dict(version='combat_handoff_live_acceptance_v1', train_only=True, diagnostic_intervention=True,
            diagnostic_only=True, ordinary_training_eligible=False, model_sha256=sha(model),
            assignments=[dict(label='train', split='train', seeds=[seed], plan=str(plan.resolve()), plan_sha256=sha(plan))],
            scope='Known TRAIN seed replay for boundary acceptance only; not fresh training or quality evidence.'))
        plans.append(plan)
    save(a.out/'protocol.json', dict(version='combat_handoff_live_acceptance_v1', technical_replays=4, labels=labels,
        historical_seeds_intentionally_reused=True, ordinary_training_eligible=False, training_performed=False,
        inputs=pins, plans=[str(p.resolve()) for p in plans], max_instances=16, timescale=2))
    shell = shutil.which('pwsh'); assert shell
    literal = lambda p: "'"+str(p.resolve()).replace("'", "''")+"'"
    command = '& '+literal(repo/'scripts/run_registered_combat_pool.ps1')+' -Plans @('+','.join(map(literal, plans))+') -MaxInstances 16 -Port '+str(a.port)+' -OutputRoot '+literal(a.out/'pool')
    save(a.out/'progress.json', dict(stage='native-acceptance', cases=4))
    run('native-pool', [shell, '-NoProfile', '-Command', command])
    pool = read(a.out/'pool/report.json')
    assert pool['state'] == 'complete' and pool['source_unchanged'] and pool['usable_captures'] == len(pool['jobs']) == 4
    for index, label in enumerate(labels):
        job = next(j for j in pool['jobs'] if j['plan_index']==index)
        actual = read(Path(job['root'])/'report.json')['results'][0]
        assert actual['generated_fixture'] == selection[label][1]['generated_fixture'], 'Replay physical start differs: '+label
    results = {}
    for index, label in enumerate(labels):
        target = a.out/label
        save(a.out/'progress.json', dict(stage='cuda-and-credit', label=label))
        run('process-'+label, [sys.executable, repo/'scripts/process_registered_combat_pool_cuda.py',
            '--pool', a.out/'pool', '--plan-index', index, '--out', target/'processing-train'])
        processed = read(target/'processing-train/report.json')
        assert processed['state'] == 'complete' and len(processed['cases']) == 1 and processed['device'] == 'cuda'
        (target/'pool').mkdir(); shutil.copy2(a.out/'pool/report.json', target/'pool/report.json')
        save(target/'complete.json', dict(state='complete', native_cases=1, cuda_cases=1, training_performed=False,
            protocol_sha256=sha(target/'protocol.json'), pool_sha256=sha(target/'pool/report.json'),
            diagnostic_only=True, ordinary_training_eligible=False))
        run('credit-'+label, [sys.executable, repo/'scripts/audit_combat_native_credit.py', '--corpus', target, '--out', target/'native-credit'])
        run('gae-'+label, [sys.executable, repo/'scripts/audit_combat_advantage_credit.py', '--corpus', target,
            '--native-audit', target/'native-credit', '--out', target/'advantage-credit'])
        credit = read(target/'native-credit/report.json'); gae = read(target/'advantage-credit/report.json')
        assert not credit['unassigned_native_effects'] and gae['gae_closed_form_max_error'] < 1e-8
        case = credit['cases'][0]; handoffs = credit['handoffs']
        if label == 'goal': assert case['goal'] and not case['death']
        if label == 'death': assert case['death'] and not case['goal']
        if label == 'horizon': assert not case['goal'] and not case['death'] and gae['retained_horizon_bootstraps'] > 0
        if label == 'unfinished':
            assert not case['goal'] and not case['death']
            assert any(not h['confirmed_goal'] and not h['bootstrap_zero'] and h['bootstrap_contract_verified'] for h in handoffs)
        results[label] = dict(seed=case['seed'], rows=case['rows'], goal=case['goal'], death=case['death'],
            handoffs=handoffs, retained_bootstraps=gae['retained_horizon_bootstraps'], credit_sha256=sha(target/'native-credit/report.json'),
            gae_sha256=sha(target/'advantage-credit/report.json'))
    for path, digest in pins.items(): assert sha(path) == digest, path
    report = dict(state='complete', version='combat_handoff_live_acceptance_v1', native_cases=4, cuda_cases=4,
        boundary_types=results, training_performed=False, diagnostic_only=True, ordinary_training_eligible=False, promotion=False,
        inputs=pins, pool_sha256=sha(a.out/'pool/report.json'))
    save(a.out/'report.json', report); save(a.out/'complete.json', dict(state='complete', report_sha256=sha(a.out/'report.json')))
    save(a.out/'progress.json', dict(stage='complete', native_cases=4, cuda_cases=4))
    print(dict(state='complete', labels=labels, native_cases=4, cuda_cases=4))


if __name__ == '__main__': main()

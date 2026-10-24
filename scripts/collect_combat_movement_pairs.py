"""Observable-only selection, one dynamic native pool, CUDA proof, separate pair labels."""
import argparse
import concurrent.futures
import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys

from evaluate_registered_combat_models import read, save, sha
from process_combat_architecture_pool import compact_closed_exports
from process_coop_learning_pool_cuda import verify_memory_export


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('corpus', 'protocol', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--port', type=int, default=46700)
    ap.add_argument('--prepared-index', type=Path, help='Sealed CUDA continuation units; never ordinary training data')
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out, corpus, protocol_path = a.out.resolve(), a.corpus.resolve(), a.protocol.resolve()
    assert not out.exists() and shutil.disk_usage('F:/').free >= 32 * 1024**3
    protocol = read(protocol_path)
    assert protocol['version'] in ('combat_movement_pair_collection_v1','combat_movement_continuation_stability_v1','combat_movement_underfire_pairs_v1','combat_memory_recovery_zero_compatibility_v1','combat_action_inertia_zero_compatibility_v1')
    continuation=protocol['version']=='combat_movement_continuation_stability_v1'
    underfire=protocol['version']=='combat_movement_underfire_pairs_v1'
    zero_compat=protocol['version'] in ('combat_memory_recovery_zero_compatibility_v1','combat_action_inertia_zero_compatibility_v1')
    inertia_zero=protocol['version']=='combat_action_inertia_zero_compatibility_v1'
    prepared_units=continuation or underfire or zero_compat
    assert prepared_units==(a.prepared_index is not None)
    capture_version='combat_native_action_inertia_zero_v1' if inertia_zero else 'combat_native_memory_recovery_zero_v1' if zero_compat else 'combat_native_continuation_pairs_v1' if continuation else 'combat_native_movement_pairs_v1'
    assert protocol['max_instances'] == 16 and protocol['timescale'] == 2
    if not prepared_units:assert protocol['maximum_variants_per_context'] == 6 and protocol['maximum_probe_cases'] == 144
    if underfire:assert protocol['native_cases']==protocol['cuda_cases']==36
    closed = read(corpus/'complete.json')
    assert closed['state'] == 'complete' and closed['native_cases'] == closed['cuda_cases'] == 32
    out.mkdir(parents=True)
    shutil.copy2(protocol_path, out/'collection-protocol.json')
    shutil.copy2(__file__, out/Path(__file__).name)
    os.environ['GOCACHE'] = str(repo/'workspace/build/gocache')
    os.environ['GOTOOLCHAIN'] = 'auto'
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or expected == digest, str(path)
        pins[str(path)] = digest
        return path

    pin(protocol_path)
    pin(corpus/'complete.json')

    def run(label, command):
        with (out/(label+'.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(list(map(str, command)), cwd=repo, stdout=log,
                stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert result.returncode == 0, str(out/(label+'.log'))

    def stage(name, **extra):
        save(out/'progress.json', dict(stage=name, **extra))

    try:
        stage('selecting-contexts')
        if prepared_units:
            is_=read(pin(a.prepared_index/'complete.json'))
            index=read(pin(a.prepared_index/'report.json',is_['report_sha256']))
            assert index['version']==(('combat_action_inertia_units_preparation_v1' if inertia_zero else 'combat_memory_recovery_units_preparation_v1') if zero_compat else 'combat_movement_units_preparation_v1' if underfire else 'combat_continuation_units_preparation_v1') and index['device']=='cuda'
            assert index['protocol_sha256']==sha(protocol_path)
            chosen=[unit['context'] for unit in index['units']]
        else:
            prepare = pin(repo/'scripts/prepare_combat_movement_probe_cuda.py')
            common = [sys.executable, prepare, '--corpus', corpus, '--protocol', protocol_path]
            run('context-index', common + ['--list-only', '--out', out/'context-index'])
            index = read(pin(out/'context-index/report.json'))
            chosen = index['contexts']
            assert len({c['seed'] for c in chosen}) == len(chosen)
        assert chosen and len(chosen) <= 24
        assert all(c['instance']['split'] == 'train' for c in chosen)
        if not prepared_units:
            assert len([c for c in chosen if c['split'] == 'train']) >= protocol['dataset_admission']['minimum_train_contexts'], 'Insufficient observable TRAIN coverage before games'
            assert len([c for c in chosen if c['split'] == 'development']) >= protocol['dataset_admission']['minimum_development_contexts'], 'Insufficient observable development coverage before games'
        else:assert len(chosen)==(4 if underfire or zero_compat else 4*protocol['continuation_replicates']) and all(c['split']=='train' for c in chosen)
        contexts, variants = [], []
        for ci, context in enumerate(chosen):
            stage('cuda-preparing', context=ci, contexts=len(chosen))
            if prepared_units:
                target=Path(index['units'][ci]['prepared'])
            else:
                target = out/'preparations'/context['context_id']
                run(f'prepare-{ci:02d}', common + ['--context-id', context['context_id'], '--out', target])
            seal = read(pin(target/'complete.json'))
            p = read(pin(target/'report.json', seal['report_sha256']))
            assert p['context'] == context and p['device'] == 'cuda'
            assert p['neural_parameters_exact']
            if not prepared_units:assert p['collection_protocol'] and p['ordinary_outside_frame_exact'] and p['zero_control_exact']
            if underfire:assert p['window_smoke'] and p['ordinary_outside_window_exact'] and p['zero_control_exact'] and Path(p['baseline']).resolve()==corpus
            if zero_compat:assert p['zero_source_raw_exact'] and Path(p['baseline']).resolve()==corpus
            assert p['source_weights_sha256'] == protocol['source_weights_sha256']
            minimum_variants=2 if zero_compat or protocol.get('study') in ('combat_underfire_constrained_confirmation_v1','combat_underfire_full_task_replication_v1') else 5
            assert p['variants'][0]['label'] == 'control' and minimum_variants <= len(p['variants']) <= (9 if underfire else 8 if continuation else 6)
            for path, h in p['inputs'].items():
                pin(path, h)
            local = copy.deepcopy(p['variants'])
            for v in local:
                v['plan_index'] = len(variants)
                v['context_index'] = ci
                pin(v['model'], v['model_sha256'])
                variants.append(v)
            contexts.append(dict(prepared=str(target), prepared_report_sha256=seal['report_sha256'],
                context=context, variants=local,unit_id=index['units'][ci]['unit_id'] if prepared_units else context['context_id']))
        assert len(variants) <= (protocol.get('native_cases',96) if prepared_units else protocol['maximum_probe_cases'])
        if prepared_units and protocol.get('native_cases'):assert len(variants)==protocol['native_cases']
        registry = pin(p['registry'])
        planner = out/'q2episode.exe'
        run('build-planner', ['go', 'build', '-o', planner, './cmd/q2episode'])
        plans = []
        stage('planning', cases=len(variants), contexts=len(contexts))
        for v in variants:
            i, context = v['plan_index'], contexts[v['context_index']]['context']
            path = out/f'plan-{i:03d}.json'
            run(f'plan-{i:03d}', [planner, '--registry', registry, '--root', repo,
                '--episodes', context['episode'], '--split', 'train', '--mode', 'learned',
                '--model', v['model'], '--count', 1, '--seed-offset', context['offset'],
                '--checked-wall-slide', '--out', path, '--artifacts', out/f'captures-{i:03d}'])
            plan = read(path)
            assert len(plan['tasks']) == 1 and plan['tasks'][0]['instances'] == [context['instance']]
            assert plan['tasks'][0]['seeds'] == [context['seed']] and plan['tasks'][0]['split'] == 'train'
            plans.append(path)
        save(out/'spec.json', dict(version=capture_version, contexts=contexts,
            corpus=str(corpus), protocol=str(protocol_path), protocol_sha256=sha(protocol_path), inputs=pins,
            diagnostic_only=True, ordinary_training_eligible=False, training_performed=False,
            quality_confirmation=False, max_instances=16, timescale=2, seed_reuse_intentional=True,
            split_scope='TRAIN source/zero technical compatibility only, no quality or fitting.' if zero_compat else 'Held-out fresh audit worlds, observed past-damage roots; no fitting.' if underfire else 'TRAIN-only repeats of future policy randomness in fixed worlds.' if continuation else 'Internal development uses separate whole TRAIN-registry seeds and never enters fitting.', promotion=False))
        shell = shutil.which('pwsh')
        assert shell
        quote = lambda p: "'" + str(p).replace("'", "''") + "'"
        command = '& ' + quote(repo/'scripts/run_registered_combat_pool.ps1')
        command += ' -Plans @(' + ','.join(map(quote, plans)) + ') -MaxInstances 16 -Port ' + str(a.port)
        command += ' -OutputRoot ' + quote(out/'pool')
        assert shutil.disk_usage('F:/').free >= 32 * 1024**3
        stage('native-movement-pairs', cases=len(variants), contexts=len(contexts))
        run('native-pool', [shell, '-NoProfile', '-Command', command])
        pool = read(out/'pool/report.json')
        assert pool['state'] == 'complete' and pool['source_unchanged']
        assert pool['usable_captures'] == len(pool['jobs']) == len(variants)
        assert all(not j['error'] for j in pool['jobs'])
        exporter = out/'q2ppo-data.exe'
        run('build-exporter', ['go', 'build', '-o', exporter, './cmd/q2ppo-data'])
        stage('native-export', cases=len(variants))

        def export(job):
            i = job['plan_index']
            v, context = variants[i], contexts[variants[i]['context_index']]['context']
            capture, directory = Path(job['root']), out/f'case-{i:02d}'
            directory.mkdir()
            r = read(capture/'report.json')
            m = read(capture/'manifest.json')
            result = r['results'][0]
            assert r['capture_complete'] and r['provenance_valid'] and result['capture_valid'] and result['dispatch_valid']
            assert result['seed_confirmed'] and result['seed'] == context['seed'] and result['generated_fixture'] == context['instance']
            assert m['model_weights_sha256'].lower() == v['model_sha256']
            assert m['observation_timing_contract'] == 'frozen_world_v1'
            run(f'export-{i:03d}', [exporter, '--batch', capture, '--model', v['model'], '--cuda-only', '--out', directory/'native'])
            verify_memory_export(Path(result['root'])/'bot.jsonl', directory/'native/replay-0')
            compact_closed_exports(directory/'native')
            return dict(model=v['model'], model_sha256=v['model_sha256'], data=str(directory/'native'), out=str(directory/'cuda'),
                native_report_sha256=sha(directory/'native/report.json'))

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as workers:
            tasks = list(workers.map(export, sorted(pool['jobs'], key=lambda j:j['plan_index'])))
        request = out/'cuda-request.json'
        save(request, dict(version='combat_cuda_finalize_batch_v1', compact_finalized_streams=True, tasks=tasks))
        stage('cuda-movement-pairs', cases=len(variants))
        run('cuda-batch', [sys.executable, repo/'scripts/finalize_combat_cuda_batch.py', '--request', request, '--receipt-root', out/'cuda-batch'])
        receipt = read(out/'cuda-batch/report.json')
        assert receipt['state'] == 'complete' and receipt['device'] == 'cuda' and len(receipt['tasks']) == len(variants)
        for path, h in pins.items():
            assert sha(path) == h, path
        save(out/'complete.json', dict(state='complete', version=capture_version,
            native_cases=len(variants), cuda_cases=len(variants), contexts=len(contexts),
            spec_sha256=sha(out/'spec.json'), pool_sha256=sha(out/'pool/report.json'),
            cuda_receipt_sha256=sha(out/'cuda-batch/report.json'), diagnostic_only=True,
            ordinary_training_eligible=False, training_performed=False, promotion=False,
            scope='Capture/CUDA completion only; admission and dataset gate are separate.'))
        stage('native-proof', contexts=len(contexts))
        for ci, item in enumerate(contexts):
            script='audit_combat_action_inertia_zero.py' if inertia_zero else 'audit_combat_memory_recovery_zero.py' if zero_compat else 'audit_combat_continuation_probe.py' if continuation else 'audit_combat_movement_probe.py'
            audit_args=[sys.executable, repo/'scripts'/script,'--probe',out,'--context-index',ci,'--out',out/'proofs'/item['unit_id']]
            if continuation and protocol.get('goal_tail_protocol'):audit_args+=['--goal-tail-protocol',repo/protocol['goal_tail_protocol']]
            run(f'proof-{ci:02d}',audit_args)
        continuation_reporters={'combat_memory_contact_movement_refresh_v1':'report_combat_memory_contact_refresh.py',
            'combat_memory_contact_movement_pilot_v1':'report_combat_memory_contact_movement.py',
            'combat_movement_mc_budget8_v1':'report_combat_continuation_budget8.py',
            'combat_underfire_horizon_pilot_v1':'report_combat_underfire_continuations.py',
            'combat_underfire_constrained_confirmation_v1':'report_combat_underfire_constrained_confirmation.py',
            'combat_underfire_full_task_replication_v1':'report_combat_underfire_full_task_replication.py',
            'combat_movement_window_comparison_v1':'report_combat_movement_window_comparison.py'}
        builder='report_combat_action_inertia_zero.py' if inertia_zero else 'report_combat_memory_recovery_zero.py' if zero_compat else 'report_combat_underfire_pairs.py' if underfire else continuation_reporters.get(protocol.get('study'),'report_combat_continuation_stability.py') if continuation else 'build_combat_movement_pair_dataset.py'
        run('dataset', [sys.executable, repo/'scripts'/builder, '--collection', out, '--out', out/'dataset'])
        dataset = read(out/'dataset/report.json')
        stage('complete', native_cases=len(variants), cuda_cases=len(variants), contexts=len(contexts),
            dataset_admitted=dataset['gate_passed'], training_performed=False)
        print(out/'dataset/report.json')
    except BaseException as error:
        stage('failed', error=str(error), exception=type(error).__name__)
        raise


if __name__ == '__main__':
    main()

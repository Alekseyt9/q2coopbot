"""Sealed development selection -> equal-budget own-policy CUDA curriculum A/B."""
import argparse
import pathlib
import shutil
import sys

from process_combat_architecture_pool import read, save, sha, run
from run_combat_sampling_evaluation import collect_compressed
from run_combat_spatial_ppo import wait_process


def compile_mix(repo, planner, registry, weights, folder, parts, offset):
    folder.mkdir()
    combined = None
    for index, (families, count) in enumerate(parts):
        path = folder / f'part-{index}.json'
        run([planner, '--registry', registry, '--episodes', ','.join(families), '--split', 'train',
             '--mode', 'learned', '--model', weights, '--count', count, '--seed-offset', offset,
             '--root', repo, '--out', path, '--artifacts', folder / 'capture'], folder / f'part-{index}.log')
        plan = read(path)
        if combined is None:
            combined = plan
        else:
            assert {k: v for k, v in plan.items() if k != 'tasks'} == {k: v for k, v in combined.items() if k != 'tasks'}
            combined['tasks'].extend(plan['tasks'])
    assert sum(len(t['seeds']) for t in combined['tasks']) == 80
    path = folder / 'plan.json'
    save(path, combined)
    run([planner, '--verify-plan', path, '--root', repo], folder / 'verify-plan.log')
    return path


def main():
    parser = argparse.ArgumentParser()
    for name in ('evaluation', 'before', 'after', 'curriculum', 'capture', 'processing'):
        parser.add_argument('--' + name, type=pathlib.Path, required=True)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    evaluation, before, after, curriculum, capture, processing = [p.resolve() for p in
        (args.evaluation, args.before, args.after, args.curriculum, args.capture, args.processing)]
    assert not capture.exists() and not processing.exists()
    capture.mkdir()
    pool_script = repo / 'scripts/run_registered_combat_episode_pool.ps1'
    pool_sha = sha(pool_script)
    curriculum_sha = sha(curriculum / 'preparation.json')
    save(capture / 'queue.json', dict(evaluation=str(evaluation), observed_process_pid=args.wait_pid,
        curriculum_sha256=curriculum_sha, pool_script_sha256=pool_sha, driver_sha256=sha(__file__),
        training_device='cuda', slots=16, timescale=2, episodes_per_arm=80, planned_episodes=160))
    save(capture / 'execution.json', dict(stage='queued', episodes=160, promotion=None))
    try:
        if args.wait_pid:
            wait_process(args.wait_pid, capture / 'execution.json', 'waiting_for_sealed_parent_selection')
        status = read(evaluation / 'progress.json')
        assert status['stage'] == 'complete' and status['diagnostics_complete']
        assert status['quality_report_sha256'] == sha(evaluation / 'quality-report.json')
        assert status['diagnostics_acceptance_sha256'] == sha(evaluation / 'diagnostics-acceptance.json')
        assert status['ownership_acceptance_sha256'] == sha(evaluation / 'ownership-acceptance.json')
        assert status['physical_storage_sha256'] == sha(evaluation / 'storage-final.json')
        assert sha(pool_script) == pool_sha, 'Pool source changed while queued'
        assert sha(curriculum / 'preparation.json') == curriculum_sha, 'Curriculum changed while queued'
        quality, protocol, ownership = [read(evaluation / name) for name in
            ('quality-report.json', 'protocol.json', 'control-ownership.json')]
        assert quality['state'] == ownership['state'] == 'complete'
        assert quality['protocol_sha256'] == ownership['protocol_sha256'] == sha(evaluation / 'protocol.json')
        assert ownership['quality_sha256'] == sha(evaluation / 'quality-report.json')
        scores = []
        by_variant = {v['variant']: v for v in quality['variants']}
        for model in ('m1', 'parent3'):
            for phase in ('before', 'after'):
                labels = [f'{model}-stochastic-{arm}-{phase}' for arm in ('a', 'b')]
                for label in labels:
                    group = ownership['groups'][label]
                    assert not group['counts'].get('rules_with_clear_target', 0)
                    assert not group['fallback_reasons'].get('pilot_equip_not_ready', 0)
                    assert by_variant[label]['episodes'] == 80
                scores.append(dict(model=model, phase=phase,
                    wins=sum(by_variant[v]['wins'] for v in labels),
                    deaths=sum(by_variant[v]['deaths'] for v in labels),
                    received_damage=sum(by_variant[v]['mean_received_damage'] for v in labels)))
        scores.sort(key=lambda s: (-s['wins'], s['deaths'], s['received_damage'], s['model'], s['phase']))
        selected = scores[0]
        parent = before if selected['phase'] == 'before' else after
        parent_report = read(parent / 'report.json')
        assert parent_report['state'] == 'complete'
        prior = next(t for t in parent_report['training'] if t['model'] == selected['model'])
        update = parent / selected['model'] / 'update'
        seal, audit, report = [read(update / name) for name in ('complete.json', 'checkpoint-cuda-audit.json', 'report.json')]
        for filename, field in [('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'), ('report.json', 'report_sha256')]:
            assert sha(update / filename) == seal[field]
        assert audit['device'] == report['device'] == 'cuda' and audit['actor_value_std_exact'] and audit['optimizer_state_exact']
        assert audit['weights_sha256'] == sha(update / 'weights.json') and audit['checkpoint_sha256'] == sha(update / 'checkpoint.pt')
        selection_entry = next(e for e in protocol['evaluations'] if e['model'] == selected['model'] + '-stochastic-a' and e['label'] == selected['phase'])
        assert selection_entry['source_weights_sha256'] == sha(update / 'weights.json')
        draft = read(curriculum / 'preparation.json')
        assert draft['state'] == 'static_preflight_complete_native_pending'
        for p in draft['plans']:
            assert sha(p['path']) == p['sha256']
        registry = pathlib.Path(draft['registry'])
        assert sha(registry) == draft['registry_sha256']
        focused = next(p['families'] for p in draft['plans'] if p['name'] == 'wall-sites')
        core = next(p['families'] for p in draft['plans'] if p['name'] == 'mixed-retention')
        families = protocol['families']
        assert len(families) == 20 and len(focused) == 8 and len(core) == 4
        assert shutil.disk_usage(capture).free > 4 * 1024**3, 'Reserve4GiB before160 captures'
        shutil.copyfile(parent / 'config.json', capture / 'config.json')
        assert sha(capture / 'config.json') == report['config_sha256']
        for file, field in [('anchor.json', 'anchor_sha256'), ('bank.json', 'bank_sha256')]:
            assert sha(parent / file) == report[field]
        save(capture / 'selection.json', dict(selected=selected, candidates=scores,
            quality_sha256=sha(evaluation / 'quality-report.json'), ownership_sha256=sha(evaluation / 'control-ownership.json'),
            checkpoint_sha256=sha(update / 'checkpoint.pt'), curriculum_sha256=sha(curriculum / 'preparation.json'),
            scope='Development selection across sums of both stochastic arms; tie deaths then received damage. No independent-test selection or promotion.'))
        bindings, plans, entries = [], [], []
        planner = repo / 'workspace/build/q2episode-sampling-v1.exe'
        for name, parts in [('uniform', [(families, 4)]), ('wall', [(focused, 8), (core, 4)])]:
            weights = capture / (name + '-weights.json')
            shutil.copyfile(update / 'weights.json', weights)
            assert not read(weights)['deterministic']
            path = compile_mix(repo, planner, registry, weights, capture / name, parts, draft['training_seed_offset'])
            plan = read(path)
            bindings.append(dict(id=name, architecture=prior['architecture'], model=str(weights), plan=str(path),
                capture_root=plan['output_root'], resume_checkpoint=str(update / 'checkpoint.pt'),
                resume_checkpoint_sha256=sha(update / 'checkpoint.pt'), resume_report=str(update / 'report.json'),
                resume_report_sha256=sha(update / 'report.json'), parent_updates_completed=report['updates_completed']))
            plans.append(str(path))
            entries.append(dict(model=name, label='behavior', root=plan['output_root'], plan=str(path),
                plan_sha256=sha(path), deterministic_weights_sha256=sha(weights), source_weights_sha256=sha(weights)))
        save(capture / 'models.json', bindings)
        save(capture / 'plans.json', plans)
        save(capture / 'protocol.json', dict(version='combat_wall_curriculum_ab_v1', evaluations=entries,
            families=families, episodes_per_model=80, total_episodes=160,
            scope='Training outcomes only, different curricula are not paired quality evidence. Same parent actor/value/std/Adam/config;80 own-policy captures and one CUDA update each. Common development evaluation required afterwards.'))
        save(capture / 'execution.json', dict(stage='collecting_own_policy', episodes=160, promotion=None))
        result = collect_compressed(repo, [pathlib.Path(p) for p in plans], capture)
        manifest = read(pathlib.Path(result['jobs'][0]['root']) / 'manifest.json')
        run([sys.executable, repo / 'scripts/verify_combat_evaluation_members.py', '--root', capture,
            '--source-fingerprint', manifest['source_fingerprint'], '--native-fingerprint', manifest['native_source_fingerprint']], capture / 'verify.log')
        run([sys.executable, repo / 'scripts/report_combat_architecture_evaluation.py', '--root', capture,
            '--member-proof', capture / 'recovery/verified-members.json'], capture / 'training-outcomes.log')
        run([sys.executable, repo / 'scripts/report_combat_control_ownership.py', '--root', capture], capture / 'ownership.log')
        own = read(capture / 'control-ownership.json')
        assert all(not g['counts'].get('rules_with_clear_target', 0) and not g['fallback_reasons'].get('pilot_equip_not_ready', 0) for g in own['groups'].values())
        save(capture / 'execution.json', dict(stage='cuda_processing', episodes=160, promotion=None))
        run([sys.executable, repo / 'scripts/process_combat_architecture_pool.py', '--capture-root', capture,
            '--out', processing, '--config', capture / 'config.json', '--exporter', parent / 'q2ppo-data.exe',
            '--anchor', parent / 'anchor.json', '--bank', parent / 'bank.json', '--export-workers', 4,
            '--cuda-only-export', '--cuda-batch-finalize'], capture / 'process.log')
        processed = read(processing / 'report.json')
        assert processed['state'] == 'complete' and len(processed['training']) == 2
        for trained in processed['training']:
            assert trained['device'] == 'cuda' and trained['allocated_episodes'] == 80
            assert trained['updates_completed'] == report['updates_completed'] + 1
            assert trained['resume_checkpoint_sha256'] == sha(update / 'checkpoint.pt')
        run([sys.executable, repo / 'scripts/audit_combat_spatial_ppo_checkpoint_cuda.py', '--roots',
            processing / 'uniform/update', processing / 'wall/update'], capture / 'checkpoint-audit.log')
        save(capture / 'execution.json', dict(stage='complete', episodes=160,
            processing_report_sha256=sha(processing / 'report.json'), parent_optimizer_preserved=True, promotion=None))
    except Exception as error:
        save(capture / 'execution.json', dict(stage='failed', error=str(error), promotion=None))
        raise


if __name__ == '__main__':
    main()

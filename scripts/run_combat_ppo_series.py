"""Sequential fresh native corpora and sealed CUDA PPO checkpoint continuation."""
import argparse
import os
import pathlib
import shutil
import sys

from process_combat_architecture_pool import read, save, sha, run
from run_combat_sampling_evaluation import collect_compressed
from run_combat_stochastic_heldout import verify_sources
from run_combat_target_refresh import compile_plan


def sealed_update(update):
    seal, audit, report = [read(update / name) for name in ('complete.json', 'checkpoint-cuda-audit.json', 'report.json')]
    for name, field in [('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'), ('report.json', 'report_sha256')]:
        assert sha(update / name) == seal[field]
    assert audit['device'] == report['device'] == 'cuda'
    assert audit['actor_value_std_exact'] and audit['optimizer_state_exact']
    assert audit['weights_sha256'] == sha(update / 'weights.json')
    assert audit['checkpoint_sha256'] == sha(update / 'checkpoint.pt')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=pathlib.Path, required=True)
    parser.add_argument('--parent-update', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--train-offset', type=int, default=200)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--training-plan', type=pathlib.Path,
                        help='Use a sealed parent and explicit training registry instead of the historical uniform fork')
    args = parser.parse_args()
    assert 1 <= args.rounds <= 4 and args.train_offset >= 0
    repo = pathlib.Path(__file__).resolve().parents[1]
    reference, initial, out = args.reference.resolve(), args.parent_update.resolve(), args.out.resolve()
    status = read(reference / 'progress.json')
    assert status['stage'] == 'complete'
    if args.training_plan:
        assert sha(reference/'result.json') == status['result_sha256']
        assert sha(reference/'quality-report.json') == read(reference/'result.json')['quality_report_sha256']
    else:
        assert status['diagnostics_complete']
        for filename, field in [('quality-report.json', 'quality_report_sha256'),
            ('diagnostics-acceptance.json', 'diagnostics_acceptance_sha256'),
            ('ownership-acceptance.json', 'ownership_acceptance_sha256'), ('storage-final.json', 'physical_storage_sha256')]:
            assert sha(reference / filename) == status[field]
        assert read(reference / 'ownership-acceptance.json')['clean']
    source_binding = verify_sources(repo, reference)
    parent_report = sealed_update(initial)
    assert parent_report['updates_completed'] >= 1 if args.training_plan else parent_report['updates_completed'] == 2
    source = initial.parents[1]
    for filename, field in [('config.json', 'config_sha256'), ('anchor.json', 'anchor_sha256'), ('bank.json', 'bank_sha256')]:
        assert sha(source / filename) == parent_report[field]
    if args.training_plan:
        original_plan = read(args.training_plan)
        architecture = {'combat_causal_attention_v1':'attention','combat_gru_v1':'gru'}[parent_report['architecture']]
        binding = dict(architecture=dict(architecture=architecture))
        assert all(t['split']=='train' for t in original_plan['tasks'])
        assert sha(original_plan['registry_path']) == original_plan['registry_sha256']
        assert sha(initial/'weights.json') == read(reference/'protocol.json')['evaluations'][0]['source_weights_sha256']
    else:
        old_bindings = read(repo / 'workspace/artifacts/wall-curriculum-ab-capture-v1-20261010/models.json')
        binding = next(b for b in old_bindings if b['id'] == 'uniform')
        assert binding['resume_checkpoint_sha256'] == sha(initial / 'checkpoint.pt')
        original_plan = read(binding['plan'])
    families = [t['episode']['id'] for t in original_plan['tasks']]
    assert len(families) == 20
    protocol = dict(version='combat_sequential_cuda_ppo_series_v1', reference=str(reference),
        reference_protocol_sha256=sha(reference / 'protocol.json'), parent_update=str(initial),
        parent_weights_sha256=sha(initial / 'weights.json'), parent_checkpoint_sha256=sha(initial / 'checkpoint.pt'),
        config_sha256=sha(source / 'config.json'), anchor_sha256=sha(source / 'anchor.json'), bank_sha256=sha(source / 'bank.json'),
        families=families, rounds=args.rounds, episodes_per_round=80, slots=16, timescale=2,
        train_offsets=[args.train_offset + 4 * i for i in range(args.rounds)], planned_validation_offset=32,
        scope='Fresh physical own-policy captures; disjoint train blocks within this series, no global unseen-seed claim. '
              'Same parent actor/value/std/Adam/config/reward, sequential CUDA updates. '
              'No quality conclusion from training outcomes; separate common development evaluation required. No independent test or promotion.')
    if args.training_plan:
        protocol['training_plan_sha256'] = sha(args.training_plan)
    if args.resume:
        assert read(out / 'protocol.json') == protocol
    else:
        assert not out.exists()
        assert shutil.disk_usage(out.parent).free > 4 * 1024**3, 'Reserve4GiB before sequential training'
        out.mkdir()
        save(out / 'protocol.json', protocol)
    current = initial
    compiler = out/'q2episode.exe' if args.training_plan else repo/'workspace/build/q2episode-sampling-v1.exe'
    exporter = out/'q2ppo-data.exe' if args.training_plan else source/'q2ppo-data.exe'
    if args.training_plan:
        os.environ['GOCACHE'] = str(repo/'workspace/build/go-cache')
        os.environ['GOTOOLCHAIN'] = 'auto'
        for package,binary in [('q2episode',compiler),('q2ppo-data',exporter)]:
            if not binary.exists():
                run(['go','build','-buildvcs=false','-o',binary,'./cmd/'+package],out/('build-'+package+'.log'))
    try:
        for index, offset in enumerate(protocol['train_offsets'], 1):
            assert verify_sources(repo, reference) == source_binding
            previous = sealed_update(current)
            folder = out / ('round-' + str(index))
            capture, processing = folder / 'capture', folder / 'processing'
            update = processing / 'series/update'
            if (capture / 'execution.json').exists() and read(capture / 'execution.json')['stage'] == 'complete':
                execution = read(capture / 'execution.json')
                assert execution['processing_report_sha256'] == sha(processing / 'report.json')
                trained = sealed_update(update)
                assert trained['resume_sha256'] == sha(current / 'checkpoint.pt')
                assert trained['updates_completed'] == previous['updates_completed'] + 1
                current = update
                continue
            if not capture.exists():
                assert shutil.disk_usage(out).free > 2 * 1024**3
                folder.mkdir(exist_ok=True)
                capture.mkdir()
                weights = capture / 'behavior-weights.json'
                shutil.copyfile(current / 'weights.json', weights)
                assert sha(weights) == sha(current / 'weights.json') and not read(weights)['deterministic']
                path = compile_plan(compiler, original_plan['registry_path'],
                    repo, weights, capture / 'series', 'train', families, offset)
                plan = read(path)
                assert sum(len(t['seeds']) for t in plan['tasks']) == 80
                run([compiler, '--verify-plan', path, '--root', repo], capture / 'verify-plan.log')
                save(capture / 'models.json', [dict(id='series', architecture=binding['architecture'], model=str(weights),
                    plan=str(path), capture_root=plan['output_root'], resume_checkpoint=str(current / 'checkpoint.pt'),
                    resume_checkpoint_sha256=sha(current / 'checkpoint.pt'), resume_report=str(current / 'report.json'),
                    resume_report_sha256=sha(current / 'report.json'), parent_updates_completed=previous['updates_completed'])])
                save(capture / 'protocol.json', dict(version='combat_series_own_policy_round_v1', families=families,
                    episodes_per_model=80, total_episodes=80, evaluations=[dict(model='series', label='behavior', root=plan['output_root'],
                        plan=str(path), plan_sha256=sha(path), deterministic_weights_sha256=sha(weights), source_weights_sha256=sha(weights))],
                    scope='Training outcomes only; own current actor on all20 train families. Not evaluation or independent test.'))
                shutil.copyfile(source / 'config.json', capture / 'config.json')
                save(capture / 'execution.json', dict(stage='collecting_own_policy', round=index, episodes=80, promotion=None))
                save(out / 'progress.json', dict(stage='collecting', round=index, rounds=args.rounds, completed_rounds=index - 1))
                collect_compressed(repo, [path], capture)
            terminal = read(capture / 'pool/report.json')
            assert terminal['state'] == 'complete' and terminal['source_unchanged'] and not any(j['error'] for j in terminal['jobs']), 'Recover existing native pool before resume; never redispatch completed members'
            assert len(terminal['jobs']) == 80
            manifest = read(pathlib.Path(terminal['jobs'][0]['root']) / 'manifest.json')
            run([sys.executable, repo / 'scripts/verify_combat_evaluation_members.py', '--root', capture,
                '--source-fingerprint', manifest['source_fingerprint'], '--native-fingerprint', manifest['native_source_fingerprint']], capture / 'verify.log')
            run([sys.executable, repo / 'scripts/report_combat_architecture_evaluation.py', '--root', capture,
                '--member-proof', capture / 'recovery/verified-members.json'], capture / 'training-outcomes.log')
            run([sys.executable, repo / 'scripts/report_combat_control_ownership.py', '--root', capture], capture / 'ownership.log')
            ownership = read(capture / 'control-ownership.json')
            assert all(not g['counts'].get('rules_with_clear_target', 0) and not g['fallback_reasons'].get('pilot_equip_not_ready', 0) for g in ownership['groups'].values())
            save(capture / 'execution.json', dict(stage='cuda_processing', round=index, episodes=80, promotion=None))
            command = [sys.executable, repo / 'scripts/process_combat_architecture_pool.py', '--capture-root', capture,
                '--out', processing, '--config', capture / 'config.json', '--exporter', exporter,
                '--anchor', source / 'anchor.json', '--bank', source / 'bank.json', '--export-workers', 4,
                '--cuda-only-export', '--cuda-batch-finalize']
            if processing.exists():
                command += ['--resume']
            run(command, capture / 'process.log')
            processed = read(processing / 'report.json')
            assert processed['state'] == 'complete' and len(processed['training']) == 1
            training = processed['training'][0]
            assert training['device'] == 'cuda' and training['allocated_episodes'] == 80
            assert training['resume_checkpoint_sha256'] == sha(current / 'checkpoint.pt')
            assert training['updates_completed'] == previous['updates_completed'] + 1
            run([sys.executable, repo / 'scripts/audit_combat_spatial_ppo_checkpoint_cuda.py', '--roots', update], capture / 'checkpoint-audit.log')
            sealed_update(update)
            save(capture / 'execution.json', dict(stage='complete', round=index, episodes=80,
                processing_report_sha256=sha(processing / 'report.json'), parent_optimizer_preserved=True, promotion=None))
            save(out / 'progress.json', dict(stage='round_complete', completed_rounds=index, rounds=args.rounds,
                latest_update=str(update), latest_weights_sha256=sha(update / 'weights.json')))
            current = update
        save(out / 'progress.json', dict(stage='complete', completed_rounds=args.rounds, episodes=80 * args.rounds,
            final_update=str(current), final_weights_sha256=sha(current / 'weights.json'), final_checkpoint_sha256=sha(current / 'checkpoint.pt'),
            updates_completed=sealed_update(current)['updates_completed'], parent_optimizer_preserved=True, promotion=None))
    except Exception as error:
        save(out / 'progress.json', dict(stage='failed', error=str(error), promotion=None))
        raise


if __name__ == '__main__':
    main()

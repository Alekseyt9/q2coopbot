"""Common native evaluation after both equal-budget CUDA curriculum updates."""
import argparse
import pathlib
import shutil
import sys

from process_combat_architecture_pool import read, save, sha, run
from run_combat_sampling_evaluation import collect_compressed
from run_combat_spatial_ppo import wait_process
from run_combat_target_refresh import compile_plan


def main():
    parser = argparse.ArgumentParser()
    for name in ('capture', 'processing', 'reference', 'out'):
        parser.add_argument('--' + name, type=pathlib.Path, required=True)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    capture, processing, reference, out = [p.resolve() for p in
        (args.capture, args.processing, args.reference, args.out)]
    assert not out.exists()
    out.mkdir()
    source_sha = sha(repo / 'scripts/run_registered_combat_episode_pool.ps1')
    reference_sha = sha(reference / 'protocol.json')
    save(out / 'queue.json', dict(training_process_pid=args.wait_pid, capture=str(capture),
        processing=str(processing), reference_protocol_sha256=reference_sha,
        pool_script_sha256=source_sha, planned_episodes=560, slots=16, timescale=2,
        scope='Three actors, two stochastic RNG arms each,80 common development conditions plus rules. No training or test in this evaluation.'))
    try:
        if args.wait_pid:
            wait_process(args.wait_pid, out / 'progress.json', 'waiting_for_cuda_curriculum_ab')
        execution = read(capture / 'execution.json')
        assert execution['stage'] == 'complete' and execution['parent_optimizer_preserved']
        assert execution['processing_report_sha256'] == sha(processing / 'report.json')
        assert read(processing / 'report.json')['state'] == 'complete'
        assert sha(repo / 'scripts/run_registered_combat_episode_pool.ps1') == source_sha
        assert sha(reference / 'protocol.json') == reference_sha
        assert shutil.disk_usage(out).free > 6 * 1024**3, 'Reserve6GiB before560-case comparison'
        reference_protocol = read(reference / 'protocol.json')
        template = read(reference_protocol['evaluations'][0]['plan'])
        families = reference_protocol['families']
        assert len(families) == 20
        bindings = {b['id']: b for b in read(capture / 'models.json')}
        assert bindings.keys() == {'uniform', 'wall'}
        assert bindings['uniform']['resume_checkpoint_sha256'] == bindings['wall']['resume_checkpoint_sha256']
        assert sha(bindings['uniform']['model']) == sha(bindings['wall']['model'])
        variants = [('parent', pathlib.Path(bindings['uniform']['model']))]
        for name in ('uniform', 'wall'):
            update = processing / name / 'update'
            audit, seal = read(update / 'checkpoint-cuda-audit.json'), read(update / 'complete.json')
            assert audit['device'] == 'cuda' and audit['actor_value_std_exact'] and audit['optimizer_state_exact']
            for file, field in [('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'), ('report.json', 'report_sha256')]:
                assert sha(update / file) == seal[field]
            assert audit['weights_sha256'] == sha(update / 'weights.json')
            assert audit['checkpoint_sha256'] == sha(update / 'checkpoint.pt')
            variants.append((name, update / 'weights.json'))
        variants.append(('rules', None))
        plans, entries = [], []
        for name, source in variants:
            modes = [('stochastic-a', 20261011), ('stochastic-b', 20261012)] if source else [('baseline', 0)]
            for label, offset in modes:
                weights = None
                if source:
                    assert not read(source)['deterministic']
                    weights = out / (name + '-' + label + '-weights.json')
                    shutil.copyfile(source, weights)
                plan = compile_plan(repo / 'workspace/build/q2episode-sampling-v1.exe', template['registry_path'],
                    repo, weights, out / (name + '-' + label), 'validation', families, 24)
                compiled = read(plan)
                compiled['policy_sampling_seed_offset'] = offset
                save(plan, compiled)
                assert {t['episode']['id']: (t['seeds'], t['instances']) for t in compiled['tasks']} == {
                    t['episode']['id']: (t['seeds'], t['instances']) for t in template['tasks']}
                run([repo / 'workspace/build/q2episode-sampling-v1.exe', '--verify-plan', plan, '--root', repo], plan.parent / 'sampling-plan-verify.log')
                plans.append(plan)
                entries.append(dict(model=name, label=label, root=compiled['output_root'], plan=str(plan),
                    plan_sha256=sha(plan), deterministic_weights_sha256=sha(weights) if weights else None,
                    source_weights_sha256=sha(source) if source else None, policy_sampling_seed_offset=offset,
                    deterministic=False if source else True))
        save(out / 'protocol.json', dict(version='combat_wall_curriculum_common_development_v1', evaluations=entries,
            families=families, episodes_per_model=80, total_episodes=560, slots=16, timescale=2,
            comparison_reference='rules-baseline', processing_sha256=sha(processing / 'report.json'),
            scope='Parent/uniform/wall two whole-action stochastic RNG arms on identical registered validation24..27. '
                  'Training budgets equal80 captures/one CUDA update; train curricula differ. Reused development, no independent test or promotion.'))
        save(out / 'progress.json', dict(stage='evaluating', episodes=560, promotion=None))
        result = collect_compressed(repo, plans, out)
        manifest = read(pathlib.Path(result['jobs'][0]['root']) / 'manifest.json')
        run([sys.executable, repo / 'scripts/verify_combat_evaluation_members.py', '--root', out,
            '--source-fingerprint', manifest['source_fingerprint'], '--native-fingerprint', manifest['native_source_fingerprint']], out / 'verify.log')
        for script, log in [('audit_combat_sampling_configs.py', 'sampling-config-audit.log'),
                            ('report_combat_architecture_evaluation.py', 'quality.log')]:
            command = [sys.executable, repo / 'scripts' / script, '--root', out]
            if script.startswith('report_'):
                command += ['--member-proof', out / 'recovery/verified-members.json']
            run(command, out / log)
        assert not read(out / 'sampling-config-audit.json')['repeated_stochastic_execution_configs']
        run([sys.executable, repo / 'scripts/finalize_combat_machinegun_evaluation.py', '--root', out], out / 'diagnostics.log')
        run([sys.executable, repo / 'scripts/report_combat_control_ownership.py', '--root', out], out / 'control-ownership.log')
        ownership = read(out / 'control-ownership.json')
        counts = {name: dict(visible_target_rules=g['counts'].get('rules_with_clear_target', 0),
            equip_fallback=g['fallback_reasons'].get('pilot_equip_not_ready', 0))
            for name, g in ownership['groups'].items() if name != 'rules-baseline'}
        save(out / 'ownership-acceptance.json', dict(state='complete', counts=counts,
            clean=all(not v['visible_target_rules'] and not v['equip_fallback'] for v in counts.values()),
            control_ownership_sha256=sha(out / 'control-ownership.json'), promotion=None))
        run([sys.executable, repo / 'scripts/report_combat_outcome_behavior.py', '--root', out], out / 'outcome-behavior.log')
        run([sys.executable, repo / 'scripts/audit_combat_physical_storage.py', '--root', out, '--out', out / 'storage-final.json'], out / 'storage-final.log')
        progress = read(out / 'progress.json')
        progress.update(ownership_acceptance_sha256=sha(out / 'ownership-acceptance.json'),
            outcome_behavior_sha256=sha(out / 'outcome-behavior.json'), physical_storage_sha256=sha(out / 'storage-final.json'))
        save(out / 'progress.json', progress)
    except Exception as error:
        save(out / 'progress.json', dict(stage='failed', error=str(error), promotion=None))
        raise


if __name__ == '__main__':
    main()

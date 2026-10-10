"""Native pilot of categorical aim confidence, preserving continuous action noise."""
import argparse
import pathlib
import shutil
import sys

from process_combat_architecture_pool import read, save, sha, run
from run_combat_sampling_evaluation import collect_compressed
from run_combat_spatial_ppo import wait_process
from run_combat_stochastic_heldout import verify_sources
from run_combat_target_refresh import compile_plan


def main():
    parser = argparse.ArgumentParser()
    for name in ('preparation', 'reference', 'out'):
        parser.add_argument('--' + name, type=pathlib.Path, required=True)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    preparation, reference, out = [p.resolve() for p in (args.preparation, args.reference, args.out)]
    assert not out.exists()
    audit_sha, reference_sha = sha(preparation / 'cuda-audit.json'), sha(reference / 'protocol.json')
    out.mkdir()
    save(out / 'queue.json', dict(preparation=str(preparation), cuda_audit_sha256=audit_sha,
        reference=str(reference), reference_protocol_sha256=reference_sha, observed_process_pid=args.wait_pid,
        episodes=112, slots=16, timescale=2, scope='Four development families, three categorical confidence profiles, two RNG arms plus rules. No training or independent test.'))
    try:
        if args.wait_pid:
            wait_process(args.wait_pid, out / 'progress.json', 'waiting_for_sealed_noise_evaluation')
        assert sha(preparation / 'cuda-audit.json') == audit_sha and sha(reference / 'protocol.json') == reference_sha
        progress = read(reference / 'progress.json')
        assert progress['stage'] == 'complete' and progress['diagnostics_complete']
        for filename, field in [('quality-report.json', 'quality_report_sha256'),
            ('diagnostics-acceptance.json', 'diagnostics_acceptance_sha256'),
            ('ownership-acceptance.json', 'ownership_acceptance_sha256'), ('storage-final.json', 'physical_storage_sha256')]:
            assert sha(reference / filename) == progress[field]
        assert read(reference / 'ownership-acceptance.json')['clean']
        source_binding = verify_sources(repo, reference)
        assert shutil.disk_usage(out).free > 2 * 1024**3, 'Reserve2GiB for112 pilot battles'
        audit = read(preparation / 'cuda-audit.json')
        assert audit['state'] == 'passed' and audit['device'] == 'cuda'
        assert audit['real_sequence_rows'] == 32
        assert sha(audit['parent_model']) == audit['parent_model_sha256']
        reference_protocol = read(reference / 'protocol.json')
        baseline_entry = next(e for e in reference_protocol['evaluations'] if e['model'] == 'noise1' and e['label'] == 'stochastic-a')
        assert sha(baseline_entry['plan']) == baseline_entry['plan_sha256']
        template = read(baseline_entry['plan'])
        assert read(template['model_path']) == read(audit['parent_model'])
        models = {m['name']: m for m in audit['models']}
        assert set(models) == {'baseline', 'mode4', 'targetmode4'}
        families = ['parasite-gunner-blaster-generated', 'parasite-gunner-machinegun-recoil',
                    'campaign-base2-site-02-blaster', 'campaign-base2-site-02-machinegun']
        expected = {t['episode']['id']: (t['seeds'], t['instances']) for t in template['tasks'] if t['episode']['id'] in families}
        assert len(expected) == 4
        plans, entries = [], []
        for name in ('baseline', 'mode4', 'targetmode4', 'rules'):
            weights = None
            if name != 'rules':
                record = models[name]
                assert record['unchanged_outputs_exact'] and record['critic_exact'] and record['raw_scaling_max_error'] < 2e-5
                assert sha(record['model']) == record['model_sha256']
                weights = out / (name + '-weights.json')
                shutil.copyfile(record['model'], weights)
                assert sha(weights) == record['model_sha256'] and not read(weights)['deterministic']
            modes = [('stochastic-a', 20261011), ('stochastic-b', 20261012)] if weights else [('baseline', 0)]
            for label, offset in modes:
                path = compile_plan(repo / 'workspace/build/q2episode-sampling-v1.exe', template['registry_path'],
                    repo, weights, out / (name + '-' + label), 'validation', families, 24)
                plan = read(path)
                plan['policy_sampling_seed_offset'] = offset
                save(path, plan)
                assert {t['episode']['id']: (t['seeds'], t['instances']) for t in plan['tasks']} == expected
                run([repo / 'workspace/build/q2episode-sampling-v1.exe', '--verify-plan', path, '--root', repo], path.parent / 'verify.log')
                plans.append(path)
                entries.append(dict(model=name, label=label, root=plan['output_root'], plan=str(path),
                    plan_sha256=sha(path), deterministic_weights_sha256=sha(weights) if weights else None,
                    source_weights_sha256=sha(weights) if weights else None, deterministic=False if weights else True,
                    policy_sampling_seed_offset=offset))
        save(out / 'protocol.json', dict(version='combat_aim_categorical_temperature_pilot_v1', evaluations=entries,
            families=families, episodes_per_model=16, total_episodes=112, slots=16, timescale=2,
            comparison_reference='rules-baseline', cuda_audit_sha256=audit_sha, reference_protocol_sha256=reference_sha,
            scope='Baseline versus4x mode logits versus4x target+mode logits, all additive actor residuals scaled. '
                  'Continuous std and other heads unchanged. Two common RNG arms on16 reused development conditions, not independent32 cases. '
                  'No training, checkpoint resume, independent test or automatic promotion. Pilot cannot establish full-registry superiority.'))
        assert verify_sources(repo, reference) == source_binding
        save(out / 'progress.json', dict(stage='evaluating', episodes=112, promotion=None))
        result = collect_compressed(repo, plans, out)
        manifest = read(pathlib.Path(result['jobs'][0]['root']) / 'manifest.json')
        run([sys.executable, repo / 'scripts/verify_combat_evaluation_members.py', '--root', out,
            '--source-fingerprint', manifest['source_fingerprint'], '--native-fingerprint', manifest['native_source_fingerprint']], out / 'verify.log')
        run([sys.executable, repo / 'scripts/audit_combat_sampling_configs.py', '--root', out], out / 'sampling-config.log')
        assert not read(out / 'sampling-config-audit.json')['repeated_stochastic_execution_configs']
        run([sys.executable, repo / 'scripts/report_combat_architecture_evaluation.py', '--root', out,
            '--member-proof', out / 'recovery/verified-members.json'], out / 'quality.log')
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
        run([sys.executable, repo / 'scripts/audit_combat_physical_storage.py', '--root', out, '--out', out / 'storage-final.json'], out / 'storage.log')
        status = read(out / 'progress.json')
        status.update(ownership_acceptance_sha256=sha(out / 'ownership-acceptance.json'),
            outcome_behavior_sha256=sha(out / 'outcome-behavior.json'), physical_storage_sha256=sha(out / 'storage-final.json'))
        save(out / 'progress.json', status)
    except Exception as error:
        save(out / 'progress.json', dict(stage='failed', error=str(error), promotion=None))
        raise


if __name__ == '__main__':
    main()

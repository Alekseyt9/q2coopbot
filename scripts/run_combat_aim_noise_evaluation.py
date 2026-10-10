"""Isolate continuous aim sampling variance in a frozen actor, native only."""
import argparse
import copy
import math
import pathlib
import shutil
import sys

from process_combat_architecture_pool import read, save, sha, run
from run_combat_sampling_evaluation import collect_compressed
from run_combat_spatial_ppo import wait_process
from run_combat_stochastic_heldout import verify_sources
from run_combat_target_refresh import compile_plan
from select_combat_stochastic_test_candidate import select


def scaled_model(model, scale):
    assert scale in (1.0, 0.5, 0.25) and not model['deterministic']
    result = copy.deepcopy(model)
    for index in (2, 3):
        result['log_std'][index] += math.log(scale)
        assert -8 <= result['log_std'][index] <= 1
    unchanged = copy.deepcopy(result)
    unchanged['log_std'] = model['log_std']
    assert unchanged == model
    assert result['log_std'][:2] == model['log_std'][:2]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    reference, out = args.reference.resolve(), args.out.resolve()
    assert not out.exists()
    protocol_hash = sha(reference / 'protocol.json')
    protocol = read(reference / 'protocol.json')
    parent = next(e for e in protocol['evaluations'] if e['model'] == 'parent' and e['label'] == 'stochastic-a')
    assert sha(parent['plan']) == parent['plan_sha256']
    template = read(parent['plan'])
    source = pathlib.Path(template['model_path'])
    source_hash = sha(source)
    assert source_hash == parent['source_weights_sha256'] == template['model_sha256']
    out.mkdir()
    save(out / 'queue.json', dict(reference=str(reference), reference_protocol_sha256=protocol_hash,
        parent_weights_sha256=source_hash, observed_process_pid=args.wait_pid,
        factors=[1.0, 0.5, 0.25], offsets=[20261011, 20261012], slots=16, timescale=2,
        episodes=560, scope='Frozen parent actor. Only yaw/pitch latent standard deviations change; all categorical heads and movement noise unchanged. Development only, no training or test.'))
    try:
        if args.wait_pid:
            wait_process(args.wait_pid, out / 'progress.json', 'waiting_for_sealed_curriculum_evaluation')
        assert sha(reference / 'protocol.json') == protocol_hash and sha(source) == source_hash
        assert read(reference / 'progress.json')['diagnostics_complete']
        source_binding = verify_sources(repo, reference)
        decision_root = out / 'selection-revalidation'
        decision_root.mkdir()
        select(reference, {name: [name + '-stochastic-a', name + '-stochastic-b'] for name in ('parent', 'uniform', 'wall')}, decision_root)
        if read(decision_root / 'decision.json')['eligible']:
            save(out / 'progress.json', dict(stage='skipped', reason='Independent test takes priority', promotion=None))
            return
        assert shutil.disk_usage(out).free > 6 * 1024**3, 'Reserve6GiB before560 development battles'
        plans, entries = [], []
        model = read(source)
        families = protocol['families']
        assert len(families) == 20
        for name, factor in [('noise1', 1.0), ('noisehalf', .5), ('noisequarter', .25), ('rules', None)]:
            weights = out / (name + '-weights.json') if factor else None
            if weights:
                save(weights, scaled_model(model, factor))
            modes = [('stochastic-a', 20261011), ('stochastic-b', 20261012)] if weights else [('baseline', 0)]
            for label, offset in modes:
                plan_path = compile_plan(repo / 'workspace/build/q2episode-sampling-v1.exe', template['registry_path'],
                    repo, weights, out / (name + '-' + label), 'validation', families, 24)
                plan = read(plan_path)
                plan['policy_sampling_seed_offset'] = offset
                save(plan_path, plan)
                assert {t['episode']['id']: (t['seeds'], t['instances']) for t in plan['tasks']} == {
                    t['episode']['id']: (t['seeds'], t['instances']) for t in template['tasks']}
                run([repo / 'workspace/build/q2episode-sampling-v1.exe', '--verify-plan', plan_path, '--root', repo], plan_path.parent / 'verify.log')
                plans.append(plan_path)
                entries.append(dict(model=name, label=label, root=plan['output_root'], plan=str(plan_path),
                    plan_sha256=sha(plan_path), deterministic_weights_sha256=sha(weights) if weights else None,
                    source_weights_sha256=sha(weights) if weights else None, deterministic=False if weights else True,
                    policy_sampling_seed_offset=offset, aim_std_factor=factor, frozen_parent_sha256=source_hash))
        save(out / 'protocol.json', dict(version='combat_aim_sampling_variance_development_v1', evaluations=entries,
            families=families, episodes_per_model=80, total_episodes=560, slots=16, timescale=2,
            comparison_reference='rules-baseline', reference_protocol_sha256=protocol_hash,
            scope='Same parent actor and all model fields except log_std yaw/pitch shifted by log(factor). '
                  'Three factors x two common RNG arms plus rules. Reused validation24..27, not independent test; no training or promotion. '
                  'Categorical aim mode/target randomness remains. Moving observations may diverge; frozen parameters do not imply identical outputs on different states.'))
        assert verify_sources(repo, reference) == source_binding
        save(out / 'progress.json', dict(stage='evaluating', episodes=560, promotion=None))
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

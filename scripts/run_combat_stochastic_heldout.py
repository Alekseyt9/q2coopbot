"""One reserved native test of a development-selected stochastic actor."""
import argparse
import hashlib
import json
import pathlib
import shutil
import sys

from process_combat_architecture_pool import read, save, sha, run
from run_combat_sampling_evaluation import collect_compressed
from select_combat_stochastic_test_candidate import select


def eligible_selection(prepared):
    selection = pathlib.Path(prepared['stochastic_selection'])
    assert sha(selection / 'selection-protocol.json') == prepared['stochastic_selection_protocol_sha256']
    decision = read(selection / 'decision.json')
    assert decision['state'] == 'selection_complete' and decision['eligible'], 'Development eligibility not met; test stays reserved'
    assert read(selection / 'progress.json')['state'] == 'selection_complete'
    return selection, decision


def conditions(plan):
    return {t['episode']['id']: dict(seeds=t['seeds'], episode_sha256=hashlib.sha256(
        json.dumps(t['episode'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()) for t in plan['tasks']}


def verify_sources(repo, development):
    proof = read(development / 'recovery/verified-members.json')
    assert proof['state'] == 'complete' and proof['protocol_sha256'] == sha(development / 'protocol.json')
    folder = pathlib.Path(next(iter(proof['members'])))
    manifest = read(folder / 'manifest.json')
    for field, base in [('sources', repo), ('native_sources', repo.parent / 'yquake2')]:
        for record in manifest[field]:
            assert sha(base / record['path']) == record['sha256'], 'Source changed after development; revalidate before test'
    return manifest['sources'], manifest['native_sources']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preparation', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    preparation, out = args.preparation.resolve(), args.out.resolve()
    prepared = read(preparation / 'protocol.json')
    assert prepared['state'] == 'static_prepared_not_dispatched'
    selection, decision = eligible_selection(prepared)
    original = pathlib.Path(prepared['original_preparation'])
    assert sha(original / 'selection-protocol.json') == prepared['original_selection_sha256']
    assert sha(original / 'seed-inventory.json') == prepared['original_inventory_sha256']
    reservation = original / 'dispatch-reservation.json'
    assert not reservation.exists(), 'Independent test already reserved for dispatch'
    assert not out.exists()
    assert shutil.disk_usage(out.parent).free > 8 * 1024**3, 'Reserve8GiB before800-case test'
    out.mkdir()
    try:
        selection_protocol = read(selection / 'selection-protocol.json')
        development = pathlib.Path(selection_protocol['evaluation'])
        assert sha(development / 'protocol.json') == selection_protocol['protocol_sha256']
        source_binding = verify_sources(repo, development)
        revalidated = out / 'development-revalidation'
        revalidated.mkdir()
        select(development, selection_protocol['groups'], revalidated)
        assert read(revalidated / 'decision.json') == decision, 'Sealed selection changed'
        refresh = out / 'inventory-refresh'
        run([sys.executable, repo / 'scripts/audit_combat_heldout_seeds.py', '--evaluation', development,
             '--out', refresh], out / 'inventory-refresh.log')
        inventory = read(refresh / 'seed-inventory.json')
        assert inventory['state'] == 'audited'
        assert inventory['scan_report_sha256'] == sha(refresh / 'scan-report.json')
        scan = read(refresh / 'scan-report.json')
        assert not scan['unreadable']
        expected_inventory = read(original / 'seed-inventory.json')
        assert inventory['conditions'] == expected_inventory['conditions'], 'Reserved conditions used or changed; do not rotate test seeds'
        expected = {c['episode']: dict(seeds=c['seeds'], episode_sha256=c['episode_sha256']) for c in inventory['conditions']}
        entries_by_name = {e['model'] + '-' + e['label']: e for e in read(development / 'protocol.json')['evaluations']}
        planner = repo / 'workspace/build/q2episode-sampling-v1.exe'
        plans, entries, common_instances = [], [], None
        specifications = []
        for arm in decision['selected']['arms']:
            entry = entries_by_name[arm]
            assert sha(entry['plan']) == entry['plan_sha256']
            source_plan = read(entry['plan'])
            model = pathlib.Path(source_plan['model_path'])
            assert not read(model)['deterministic']
            assert sha(model) == source_plan['model_sha256'] == decision['selected']['actor_sha256']
            specifications.append((arm, model, source_plan, source_plan['policy_sampling_seed_offset']))
        for control in prepared['controls']:
            assert sha(control['plan']) == control['plan_sha256']
            source_plan = read(control['plan'])
            model = pathlib.Path(control['model']) if control['model'] else None
            if model:
                assert sha(model) == control['model_sha256'] and read(model)['deterministic']
            specifications.append((control['name'], model, source_plan, 0))
        assert len(specifications) == 5
        for name, source, template, offset in specifications:
            folder = out / name
            folder.mkdir()
            model = out / (name + '-weights.json') if source else None
            if source:
                shutil.copyfile(source, model)
                assert sha(model) == sha(source)
            path = folder / 'plan.json'
            command = [planner, '--registry', template['registry_path'], '--episodes', ','.join(prepared['families']),
                       '--split', 'test', '--mode', 'learned' if model else 'rules', '--count', 8,
                       '--seed-offset', prepared['seed_offset'], '--root', repo, '--out', path,
                       '--artifacts', folder / 'capture']
            if model:
                command += ['--model', model]
            run(command, folder / 'compile.log')
            plan = read(path)
            plan['policy_sampling_seed_offset'] = offset
            save(path, plan)
            run([planner, '--verify-plan', path, '--root', repo], folder / 'verify.log')
            assert conditions(plan) == expected
            instances = {t['episode']['id']: t['instances'] for t in plan['tasks']}
            if common_instances is None:
                common_instances = instances
            assert instances == common_instances and sum(len(v) for v in instances.values()) == 160
            plans.append(path)
            selected_arm = name in decision['selected']['arms']
            label = ('stochastic-a' if name == decision['selected']['arms'][0] else 'stochastic-b') if selected_arm else 'test'
            entries.append(dict(model='selected' if selected_arm else name, label=label, root=plan['output_root'], plan=str(path),
                plan_sha256=sha(path), source_weights_sha256=sha(source) if source else None,
                deterministic_weights_sha256=sha(model) if model else None,
                deterministic=read(model)['deterministic'] if model else True, policy_sampling_seed_offset=offset))
        save(out / 'protocol.json', dict(version='combat_stochastic_independent_test_v1', evaluations=entries,
            families=prepared['families'], episodes_per_model=160, total_episodes=800, slots=16, timescale=2,
            comparison_reference='rules-test', selection_decision_sha256=sha(selection / 'decision.json'),
            preparation_sha256=sha(preparation / 'protocol.json'), seed_inventory_sha256=sha(refresh / 'seed-inventory.json'),
            scope='One development-selected actor, two RNG arms sharing160 test conditions, original parent3-before/FireBC/rules controls. '
                  'No training, candidate reselection or automatic promotion; novelty bounded to refreshed workspace metadata.'))
        # Exclusive creation prevents two output roots from dispatching this reserved test.
        assert verify_sources(repo, development) == source_binding
        with reservation.open('x', encoding='utf-8') as stream:
            json.dump(dict(state='reserved', output=str(out), candidate=decision['selected']['name'],
                protocol_sha256=sha(out / 'protocol.json'), decision_sha256=sha(selection / 'decision.json')), stream, indent=2)
        save(out / 'progress.json', dict(stage='testing', episodes=800, promotion=None))
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
            for name, g in ownership['groups'].items() if name != 'rules-test'}
        save(out / 'ownership-acceptance.json', dict(state='complete', counts=counts,
            clean=all(not v['visible_target_rules'] and not v['equip_fallback'] for v in counts.values()),
            control_ownership_sha256=sha(out / 'control-ownership.json'), promotion=None))
        run([sys.executable, repo / 'scripts/report_combat_outcome_behavior.py', '--root', out], out / 'outcome-behavior.log')
        run([sys.executable, repo / 'scripts/audit_combat_physical_storage.py', '--root', out,
             '--out', out / 'storage-final.json'], out / 'storage.log')
        status = read(out / 'progress.json')
        status.update(ownership_acceptance_sha256=sha(out / 'ownership-acceptance.json'),
            outcome_behavior_sha256=sha(out / 'outcome-behavior.json'), physical_storage_sha256=sha(out / 'storage-final.json'))
        save(out / 'progress.json', status)
        save(reservation, dict(state='captured', output=str(out), protocol_sha256=sha(out / 'protocol.json'),
            quality_report_sha256=sha(out / 'quality-report.json'), promotion=None))
    except Exception as error:
        save(out / 'progress.json', dict(stage='failed', error=str(error), promotion=None))
        raise


if __name__ == '__main__':
    main()

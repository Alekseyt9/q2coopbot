"""Freeze held-out controls/conditions without selecting or running an actor."""
import argparse
import hashlib
import json
import pathlib

from process_combat_architecture_pool import read, save, sha, run


def main():
    parser = argparse.ArgumentParser()
    for name in ('preparation', 'selection', 'out'):
        parser.add_argument('--' + name, type=pathlib.Path, required=True)
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    preparation, selection, out = args.preparation.resolve(), args.selection.resolve(), args.out.resolve()
    original = read(preparation / 'selection-protocol.json')
    inventory = read(preparation / 'seed-inventory.json')
    assert original['state'] == 'prepared_not_dispatched' and inventory['state'] == 'audited'
    assert original['seed_inventory_sha256'] == sha(preparation / 'seed-inventory.json')
    assert inventory['scan_report_sha256'] == sha(preparation / 'scan-report.json')
    assert not read(preparation / 'scan-report.json')['unreadable']
    assert not (preparation / 'dispatch-reservation.json').exists(), 'Test already reserved for execution'
    selection_protocol = read(selection / 'selection-protocol.json')
    assert selection_protocol['test_dispatch'] is False and len(selection_protocol['groups']) == 3
    development = pathlib.Path(selection_protocol['evaluation'])
    assert sha(development / 'protocol.json') == selection_protocol['protocol_sha256']
    controls_root = pathlib.Path(original['development_evaluation'])
    assert sha(controls_root / 'protocol.json') == original['development_protocol_sha256']
    controls = {e['model'] + '-' + e['label']: e for e in read(controls_root / 'protocol.json')['evaluations']}
    families = original['families']
    assert len(families) == 20 and original['count_per_family'] == 8
    assert inventory['episodes_per_arm'] == 160
    assert not out.exists()
    out.mkdir()
    planner = repo / 'workspace/build/q2episode-sampling-v1.exe'
    frozen = []
    for name, key in [('parent3-before', 'parent3-before'), ('firebc', 'firebc-baseline'), ('rules', 'rules-baseline')]:
        entry = controls[key]
        assert sha(entry['plan']) == entry['plan_sha256']
        source_plan = read(entry['plan'])
        model = pathlib.Path(source_plan['model_path']) if source_plan.get('model_path') else None
        if model:
            assert sha(model) == source_plan['model_sha256'] == entry['deterministic_weights_sha256']
            assert read(model)['deterministic']
        folder = out / name
        folder.mkdir()
        path = folder / 'plan.json'
        command = [planner, '--registry', source_plan['registry_path'], '--episodes', ','.join(families),
            '--split', 'test', '--mode', 'learned' if model else 'rules', '--count', 8,
            '--seed-offset', original['seed_offset'], '--root', repo, '--out', path,
            '--artifacts', folder / 'not-dispatched']
        if model:
            command += ['--model', model]
        run(command, folder / 'compile.log')
        run([planner, '--verify-plan', path, '--root', repo], folder / 'verify-plan.log')
        plan = read(path)
        assert {t['episode']['id']: t['seeds'] for t in plan['tasks']} == {c['episode']: c['seeds'] for c in inventory['conditions']}
        assert {t['episode']['id']: hashlib.sha256(json.dumps(t['episode'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            for t in plan['tasks']} == {c['episode']: c['episode_sha256'] for c in inventory['conditions']}
        assert sum(len(t['instances']) for t in plan['tasks']) == 160
        frozen.append(dict(name=name, reference_variant=key, model=str(model) if model else None,
            model_sha256=sha(model) if model else None, plan=str(path), plan_sha256=sha(path)))
    save(out / 'protocol.json', dict(version='combat_stochastic_heldout_preparation_v1',
        state='static_prepared_not_dispatched', original_preparation=str(preparation),
        original_selection_sha256=sha(preparation / 'selection-protocol.json'),
        original_inventory_sha256=sha(preparation / 'seed-inventory.json'),
        stochastic_selection=str(selection), stochastic_selection_protocol_sha256=sha(selection / 'selection-protocol.json'),
        controls=frozen, selected_arms=['stochastic-a', 'stochastic-b'],
        families=families, split='test', seed_offset=original['seed_offset'], count_per_family=8,
        episodes_per_arm=160, planned_variants=5, planned_total_episodes=800,
        slots=16, timescale=2, inventory_refresh_required=True, single_dispatch_reservation_required=True,
        scope='Static controls and reserved conditions only, no native test or neural inference. '
        'Two RNG arms of one actor selected solely from sealed development plus original parent3-before/FireBC/rules controls. '
        'Fresh metadata audit, eligible selection, selected plans and atomic single-dispatch reservation required before execution. '
        'Old inventory novelty is bounded to its recorded metadata snapshot, not global history.'))
    print('Prepared three160-condition test control plans; no actor selection or native dispatch', flush=True)


if __name__ == '__main__':
    main()

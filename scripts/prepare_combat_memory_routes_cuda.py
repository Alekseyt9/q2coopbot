"""Migrate frozen v12 actor/critic to route observations; verify real histories on CUDA.

This creates an initial model, never resumes an optimizer or fits combat outcomes.
"""
import argparse
import copy
import json
import subprocess
from pathlib import Path

from ppo_combat import torch, nn, layers, training_devices
from ppo_recurrent import prepare_context, sequence
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model
from combat_memory_route_features import VERSION, WIDTH, PARENT_WIDTH, TAIL_WIDTH, encode_tail
from combat_memory_route_movement import VERSION as MOVEMENT_VERSION, INPUT_WIDTH, validate, inputs
from evaluate_registered_combat_models import read, save, sha


def migrate(source):
    assert source['feature_version'] == 'combat_features_v12_projectile_types_neutral'
    assert source.get('attention') and not source.get('memory')
    assert not any(source.get(k) for k in ('memory_route_movement', 'memory_recovery', 'movement_inertia', 'entity_attention'))
    assert not any(k.startswith('diagnostic_') for k in source)
    result = copy.deepcopy(source)
    for role in ('actor', 'value'):
        for row in result[role][0]['weight']:
            assert len(row) == PARENT_WIDTH
            row.extend([0.]*TAIL_WIDTH)
    result['feature_version'] = VERSION
    head = nn.Sequential(nn.Linear(INPUT_WIDTH, 32, device='cuda'), nn.ReLU(),
                         nn.Linear(32, 16, device='cuda'), nn.ReLU(), nn.Linear(16, 2, device='cuda'))
    with torch.no_grad():
        head[-1].weight.zero_()
        head[-1].bias.zero_()
    result['memory_route_movement'] = dict(version=MOVEMENT_VERSION, layers=layers(head))
    validate(result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--exporter', type=Path, required=True)
    ap.add_argument('--game-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    training_devices()
    torch.set_default_device('cuda')
    torch.set_num_threads(2)
    torch.manual_seed(20261025)
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=False)
    selected = read(args.selection)
    source_path = Path(selected['source_model'])
    assert sha(source_path) == selected['source_weights_sha256']
    source = read(source_path)
    migrated = migrate(source)
    save(args.out/'weights.json', migrated)
    old_actor = branch(base_model(source, allow_shared=True), 'actor').requires_grad_(False)
    new_actor = branch(base_model(migrated, allow_shared=True), 'actor').requires_grad_(False)
    old_value = branch(source, 'value').requires_grad_(False)
    new_value = branch(migrated, 'value').requires_grad_(False)
    pins = {str(source_path.resolve()): sha(source_path), str(args.selection.resolve()): sha(args.selection),
            str(args.exporter.resolve()): sha(args.exporter)}
    checks = []
    for ci, case in enumerate(selected['cases']):
        root = Path(case['source_case'])
        steps = list((root/'native').rglob('steps.jsonl'))
        assert len(steps) == 1
        path = steps[0]
        pins[str(path.resolve())] = sha(path)
        observations = [json.loads(s)['observation'] for s in path.open(encoding='utf-8-sig')]
        inp, enriched = args.out/f'observations-{ci}.jsonl', args.out/f'enriched-{ci}.jsonl'
        with inp.open('x', encoding='utf-8') as f:
            for o in observations:
                f.write(json.dumps(o, allow_nan=False)+'\n')
        subprocess.run([str(args.exporter.resolve()), '--input', str(inp.resolve()), '--output', str(enriched.resolve()),
                        '--game-dir', str(args.game_dir.resolve())], check=True)
        data = [json.loads(s) for s in enriched.open(encoding='utf-8')]
        enriched_obs = [r['observation'] for r in data]
        encoded = encode_tail(enriched_obs)
        expected = torch.tensor([r['features'][PARENT_WIDTH:] for r in data], device='cuda')
        torch.testing.assert_close(encoded, expected, rtol=1e-6, atol=1e-6)
        by_frame = {r['observation']['identity']['frame']: r for r in data}
        assert len(by_frame) == len(data), 'Replay includes duplicate frames; need full identity selection'
        paths = [root/'cuda'/'rollout.jsonl', root/'cuda'/'sequence.jsonl']
        for p in paths:
            pins[str(p.resolve())] = sha(p)
        rows, history = [[json.loads(s) for s in p.open(encoding='utf-8-sig')] for p in paths]
        new_rows, new_history = copy.deepcopy(rows), copy.deepcopy(history)
        for r in [*new_rows, *new_history]:
            native = by_frame[r['frame']]
            x = native['features']
            assert len(x) == WIDTH
            torch.testing.assert_close(torch.tensor(x[:PARENT_WIDTH]), torch.tensor(r['features']), rtol=1e-6, atol=1e-6)
            r['features'] = x
        old_context = prepare_context(history, rows, 'cuda')
        new_context = prepare_context(new_history, new_rows, 'cuda')
        errors = {}
        with torch.inference_mode():
            for role, old, new in [('actor', old_actor, new_actor), ('critic', old_value, new_value)]:
                a, old_memory = sequence(old, old_context, 32, True, 'actor' if role == 'actor' else 'value')
                b, new_memory = sequence(new, new_context, 32, True, 'actor' if role == 'actor' else 'value')
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
                assert max(old_memory, new_memory) < 2e-5
                errors[role] = dict(output_max_error=float((a-b).abs().max()), old_memory_error=old_memory, new_memory_error=new_memory)
            x = torch.tensor([r['features'] for r in new_rows], device='cuda')
            raw, _ = sequence(new_actor, new_context, 32, False, 'actor')
            _, active = inputs(x, raw)
        assert int(active.sum()) > 0, 'No actual route-active rows in selected replay'
        checks.append(dict(map=case['map'], rows=len(rows), route_active_rows=int(active.sum()),
                           tail_encoding_max_error=float((encoded-expected).abs().max()), errors=errors))
        print(case['map'], 'CUDA zero migration verified', len(rows), 'rows;', int(active.sum()), 'route-active', flush=True)
    for path, digest in pins.items():
        assert sha(Path(path)) == digest, 'Input changed during verification'
    report = dict(version='combat_memory_route_cuda_migration_v1', state='passed', device=torch.cuda.get_device_name(),
                  feature_version=VERSION, width=WIDTH, checks=checks, inputs=pins,
                  weights=str((args.out/'weights.json').resolve()), weights_sha256=sha(args.out/'weights.json'),
                  optimizer_migrated=False, training_performed=False, promotion=False,
                  scope='Real historical TRAIN observations and temporal context; static geometry enrichment only. No live-game equivalence or combat improvement established.')
    save(args.out/'report.json', report)


if __name__ == '__main__':
    main()

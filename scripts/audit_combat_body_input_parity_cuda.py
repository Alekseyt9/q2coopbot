"""Audit actual initial commands/worlds and CUDA body-channel encoding after migration."""
import argparse
import collections
import json
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha
from ppo_combat import torch, training_devices
from combat_body_features import VERSION, ABLATED_VERSION, LEGACY_WIDTH, WIDTH, encode_tail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    training_devices(); torch.set_default_device('cuda'); torch.set_num_threads(2)
    root, pins = args.evaluation.resolve(), {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = pins.get(str(path)) or sha(path)
        assert expected is None or expected == digest, str(path)
        pins[str(path)] = digest
        return path

    pin(__file__); pin(Path(__file__).with_name('combat_body_features.py'))
    closed = read(pin(root/'complete.json'))
    assert closed['state'] == 'complete' and closed['native_cases'] == closed['cuda_cases'] == 12
    pool = read(pin(root/'pool/report.json', closed['pool_sha256']))
    assert pool['state'] == 'complete' and pool['source_unchanged'] and pool['usable_captures'] == 12
    fixture = read(pin(root/'fixture-proof.json'))
    assert fixture['spec_sha256'] == closed['spec_sha256']
    labels = {index: Path(binding['path']).stem.removesuffix('-plan') for index, binding in enumerate(fixture['plans'])}
    expected_versions = {'source-v10': 'combat_features_v10', 'body-initial': VERSION, 'body-ablated-initial': ABLATED_VERSION}
    assert set(labels.values()) == set(expected_versions)
    models = {model['label']: model for model in fixture['models']}
    traces, cases, rows_total, tail_maximum = {}, [], 0, 0.
    for job in pool['jobs']:
        label = labels[job['plan_index']]
        directory = root/('processing-'+label)/f"case-{job['job']:04d}"
        meta = read(pin(directory/'cuda/report.json'))
        assert meta['feature_version'] == expected_versions[label] and meta['model_sha256'] == models[label]['model_sha256']
        verification = read(pin(directory/'cuda/cuda-verification.json', meta['cuda_verification_sha256']))
        assert verification['state'] == 'passed' and verification['device'] == 'cuda'
        with pin(directory/'cuda/rollout.jsonl', meta['rollout_sha256']).open(encoding='utf-8-sig') as stream:
            rows = list(map(json.loads, stream))
        accepted = {(row['index'], row['frame']) for row in rows}
        assert len(accepted) == len(rows) == meta['rows']
        steps = directory/'native/replay-0/steps.jsonl'
        pin(steps, meta['source_sha256'][str(steps)])
        with steps.open(encoding='utf-8-sig') as stream:
            all_steps = list(map(json.loads, stream))
        selected = {step['observation']['identity']['frame']: step for step in all_steps
                    if (step['index'], step['observation']['identity']['frame']) in accepted}
        assert len(selected) == len(rows)
        capture = read(pin(Path(job['root'])/'report.json'))
        assert capture['capture_complete'] and capture['provenance_valid']
        result = capture['results'][0]
        stop = result['goal_stop']['kill_frame'] if result['goal_stop'] else result['first_life']['end_frame']
        gameplay = {step['observation']['identity']['frame']: step for step in all_steps
                    if step['observation']['identity']['life'] == 1 and step['observation']['health'] > 0 and
                    min(selected) <= step['observation']['identity']['frame'] < stop}
        features = {row['frame']: row['features'] for row in rows}
        assert len(features) == len(rows)
        if label != 'source-v10':
            frames = sorted(selected)
            encoded = encode_tail([selected[frame]['observation'] for frame in frames], expected_versions[label])
            recorded = torch.tensor([features[frame][LEGACY_WIDTH:] for frame in frames], device='cuda')
            assert recorded.shape == encoded.shape == (len(rows), WIDTH-LEGACY_WIDTH)
            error = float((recorded-encoded).abs().max())
            tail_maximum = max(tail_maximum, error)
            assert error < 5e-6, 'Go-recorded/CUDA body feature encoding differs'
        rows_total += len(rows)
        traces[label, job['seed']] = (selected, gameplay, features)
        cases.append(dict(label=label, seed=job['seed'], accepted_rows=len(rows), gameplay_frames=len(gameplay)))

    world_fields = ('health', 'armor', 'position', 'velocity', 'on_ground', 'ducked', 'weapon', 'ammo',
                    'gun_frame', 'view_angles', 'kick_angles_degrees', 'enemies', 'visible_projectiles', 'visible_props')
    comparisons = []
    for seed in sorted({job['seed'] for job in pool['jobs']}):
        before, before_gameplay, before_features = traces['source-v10', seed]
        for label in ('body-initial', 'body-ablated-initial'):
            after, after_gameplay, after_features = traces[label, seed]
            frame_sets = set(before) == set(after) and set(before_gameplay) == set(after_gameplay)
            command_mismatches, world_mismatches, ownership_mismatches = [], [], []
            for frame in sorted(set(before_gameplay) & set(after_gameplay)):
                a, b = before_gameplay[frame], after_gameplay[frame]
                if a['sent_command'] != b['sent_command']:
                    command_mismatches.append(frame)
                fields = [key for key in world_fields if a['observation'].get(key) != b['observation'].get(key)]
                if fields:
                    world_mismatches.append(dict(frame=frame, fields=fields))
                if a['owner'] != b['owner']:
                    ownership_mismatches.append(frame)
            common = sorted(set(before_features) & set(after_features))
            a = torch.tensor([before_features[frame] for frame in common], device='cuda')
            b = torch.tensor([after_features[frame][:LEGACY_WIDTH] for frame in common], device='cuda')
            prefix_error = float((a-b).abs().max())
            comparisons.append(dict(label=label, seed=seed, frame_sets_equal=frame_sets,
                command_mismatch_frames=command_mismatches, world_mismatches=world_mismatches,
                ownership_mismatch_frames=ownership_mismatches, prefix_max_error=prefix_error,
                passed=frame_sets and not command_mismatches and not world_mismatches and not ownership_mismatches and prefix_error == 0))
    assert rows_total == 1338 and len(comparisons) == 8
    assert all(sha(path) == digest for path, digest in pins.items())
    args.out.mkdir(parents=True)
    checks = dict(all_native_cuda_cases_closed=True, all_eight_native_pairs_identical=all(row['passed'] for row in comparisons),
                  body_channels_match_go_recording=True, preserved_v10_input_prefix=all(row['prefix_max_error'] == 0 for row in comparisons))
    save(args.out/'report.json', dict(state='complete', version='combat_body_input_initial_parity_v1', device='cuda',
        gate_passed=all(checks.values()), checks=checks, cases=cases, comparisons=comparisons,
        native_cases=12, cuda_cases=12, rows=rows_total, body_encoding_max_error=tail_maximum, inputs=pins,
        training_performed=False, promotion=False,
        scope='Four repeated technical fixtures after the documented threat-memory admission fix. '
              'Compare actual sent commands, ownership and observed world state through first-life goal/death; '
              'shutdown tail excluded. CUDA compares input prefix and recorded body encoding; NN replay is separately '
              'sealed for all12 cases. No fitting, fresh quality confirmation or general-superiority claim.'))
    save(args.out/'complete.json', dict(state='complete', gate_passed=all(checks.values()), report_sha256=sha(args.out/'report.json')))
    print(dict(checks=checks, rows=rows_total, body_encoding_max_error=tail_maximum))


if __name__ == '__main__':
    main()

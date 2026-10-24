"""Seal inference-only blaster exploration profiles; all neural checks on CUDA."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from evaluate_registered_combat_models import read, save, sha
from ppo_combat import torch, training_devices
from ppo_recurrent import prepare_context, sequence
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model
from combat_weapon_exploration import VERSION, BLASTER_VERSION, validate, apply
from combat_weapon_head import feature_mask
from fork_conditional_weapon_exploration_cuda import equal_tree


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'data', 'protocol', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    a = ap.parse_args()
    assert not a.out.exists()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    def lines(path):
        with path.open(encoding='utf-8-sig') as stream:
            return [json.loads(line) for line in stream]

    pin(__file__)
    for name in ('combat_weapon_exploration.py', 'combat_body_features.py', 'combat_spatial_aim.py',
            'combat_weapon_head.py', 'ppo_combat.py', 'ppo_recurrent.py', 'migrate_combat_navigation_cuda.py'):
        pin(Path(__file__).with_name(name))
    protocol = read(pin(a.protocol))
    assert protocol['version'] == 'combat_blaster_exploration_probe_v1'
    seal = read(pin(a.source/'complete.json'))
    model = read(pin(a.source/'weights.json', seal['weights_sha256']))
    pin(a.source/'checkpoint.pt', seal['checkpoint_sha256'])
    assert seal['weights_sha256'] == protocol['source_weights_sha256']
    assert model['weapon_exploration'] == dict(version=VERSION, temperature=2)
    meta = read(pin(a.data/'report.json'))
    assert meta['model_sha256'] == seal['weights_sha256'] and meta['numerical_verification'] == 'cuda_verified_v1'
    rows = lines(pin(a.data/'rollout.jsonl', meta['rollout_sha256']))
    histories = lines(pin(a.data/'sequence.jsonl', meta['sequence_sha256']))
    training_devices()
    torch.set_default_device('cuda')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    actor = branch(base_model(model, allow_shared=True), 'actor').requires_grad_(False)
    value = branch(base_model(model, allow_shared=True), 'value').requires_grad_(False)
    context = prepare_context(histories, rows, 'cuda')
    x = torch.tensor([r['features'] for r in rows], device='cuda', dtype=torch.float32)
    with torch.no_grad():
        original, state_error = sequence(actor, context, 32, True, 'actor')
        assert state_error < 2e-5 and original.is_cuda
        mask = feature_mask(x[:, :845])
        blaster_gate = (x[:, 1122] == 1) & mask[:, 2:].any(-1)
        original_probabilities = original[:, 8:20].masked_fill(~mask, -torch.inf).softmax(-1)
    assert blaster_gate.any()
    a.out.mkdir(parents=True)
    source_target = a.out/'models/source'
    source_target.mkdir(parents=True)
    shutil.copyfile(a.source/'weights.json', source_target/'weights.json')
    variants = [dict(label='source', model=str((source_target/'weights.json').resolve()),
        model_sha256=sha(source_target/'weights.json'))]
    probability_rows = []
    for label, temperature in protocol['profiles'].items():
        changed = copy.deepcopy(model)
        changed['weapon_exploration'] = dict(version=BLASTER_VERSION, temperature=2, blaster_temperature=temperature)
        validate(changed)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            decoded = branch(base_model(changed, allow_shared=True), 'actor').requires_grad_(False)
            decoded_value = branch(base_model(changed, allow_shared=True), 'value').requires_grad_(False)
        assert equal_tree(actor.state_dict(), decoded.state_dict())
        assert equal_tree(value.state_dict(), decoded_value.state_dict())
        old_serial, new_serial = copy.deepcopy(model), copy.deepcopy(changed)
        old_serial.pop('weapon_exploration'); new_serial.pop('weapon_exploration')
        assert old_serial == new_serial
        with torch.no_grad():
            current, error = sequence(decoded, context, 32, True, 'actor')
            assert error < 2e-5
            assert torch.equal(original[:, :8], current[:, :8]) and torch.equal(original[:, 20:], current[:, 20:])
            assert torch.equal(original[~blaster_gate], current[~blaster_gate])
            assert torch.equal(current[blaster_gate, 8:20], original[blaster_gate, 8:20]/temperature)
            if temperature == 1:
                assert torch.equal(original, current)
            probabilities = current[:, 8:20].masked_fill(~mask, -torch.inf).softmax(-1)
            assert torch.equal(original_probabilities.argmax(-1), probabilities.argmax(-1))
        folder = a.out/'models'/label
        folder.mkdir()
        save(folder/'weights.json', changed)
        variants.append(dict(label=label, model=str((folder/'weights.json').resolve()), model_sha256=sha(folder/'weights.json')))
        for i, row in enumerate(rows):
            if row['frame'] in (100, 110, 120, 130, 132):
                probability_rows.append(dict(label=label, frame=row['frame'], keep=float(probabilities[i, 0]), shotgun=float(probabilities[i, 2])))
    # Independent CUDA edge fixtures cover the old grenade gate, stale/absent
    # inventory masks, unusable alternatives, other weapons and padding.
    features = torch.zeros((9, x.shape[1]), device='cuda')
    cases = [(1, [0, 1, 2], 4), (1, [0, 1], 1), (1, [0], 1), (1, [0, 1, 11], 4),
        (11, [0, 1, 11], 2), (11, [0, 7, 11], 1), (2, [0, 1, 2], 1), (0, [0, 1, 2], 1), (None, [], 1)]
    for i, (equipped, available, _) in enumerate(cases):
        features[i, [833+j for j in available]] = 1
        if equipped is not None:
            features[i, 1121+equipped] = 1
    raw = torch.randn((len(cases), original.shape[1]), device='cuda').requires_grad_()
    old = apply(model['weapon_exploration'], features, raw)
    unit = dict(version=BLASTER_VERSION, temperature=2, blaster_temperature=1)
    candidate = dict(unit, blaster_temperature=4)
    assert torch.equal(old, apply(unit, features, raw))
    expected = raw.detach().clone()
    gradient = torch.ones_like(raw)
    for i, (_, _, divisor) in enumerate(cases):
        expected[i, 8:20] /= divisor
        gradient[i, 8:20] /= divisor
    output = apply(candidate, features, raw)
    assert torch.equal(output, expected)
    output.sum().backward()
    assert torch.equal(raw.grad, gradient)
    # v1 rejects v2-only fields, and v2 requires its explicit temperature.
    for header in (dict(version=VERSION, temperature=2, blaster_temperature=1),
            dict(version=BLASTER_VERSION, temperature=2)):
        invalid = dict(model, weapon_exploration=header)
        try:
            validate(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid versioned header admitted')
    assert all(sha(path) == digest for path, digest in pins.items())
    save(a.out/'report.json', dict(state='complete', version='combat_blaster_exploration_preparation_v1', device='cuda',
        protocol_sha256=sha(a.protocol), variants=variants, source_weights_sha256=seal['weights_sha256'],
        history_rows=len(rows), blaster_gate_rows=int(blaster_gate.sum()), state_max_error=state_error,
        actor_value_parameters_exact=True, other_serialized_fields_exact=True, zero_control_exact=True,
        other_heads_exact=True, legacy_grenade_gate_exact=True, masks_unchanged=True, argmax_preserved=True,
        cuda_gate_gradient_verified=True, synthetic_cases=len(cases), probabilities=probability_rows, inputs=pins,
        checkpoint_written=False, training_performed=False, promotion=False,
        scope='Inference-only metadata profiles; no resumable checkpoints produced and source checkpoint untouched. '
              'Fixed histories verify tensors/outputs on CUDA, not quality. Fresh native replay required.'))
    save(a.out/'complete.json', dict(state='complete', report_sha256=sha(a.out/'report.json')))
    print(dict(device='cuda', variants=len(variants), rows=len(rows), gated_rows=int(blaster_gate.sum()), probabilities=probability_rows))


if __name__ == '__main__':
    main()

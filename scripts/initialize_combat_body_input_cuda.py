"""Widen v10 inputs on CUDA, preserving source weights and explicit optimizer anchors."""
import argparse
import copy
import json
from pathlib import Path

from evaluate_registered_combat_models import read, sha
from ppo_combat import torch, nn, training_devices, advantages
from ppo_recurrent import prepare_context, sequence, durable_json, durable_write
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model, events, probabilities
from combat_geometry_ppo import GeometryJoint
from combat_checkpoint_scope import require_checkpoint_scope
from combat_execution_contract import SCOPE as SOURCE_SCOPE, require_same, LEGACY
from combat_body_features import VERSION, ABLATED_VERSION, LEGACY_WIDTH, WIDTH, SCOPE, encode_tail
from combat_body_scope import BodyScope
from combat_weapon_state_residual import optimizer as actor_optimizer
from fork_conditional_weapon_exploration_cuda import equal_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'corpus', 'protocol', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--mode', choices=('observed', 'ablated'), required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    training_devices(); torch.set_default_device('cuda'); torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = pins.get(str(path)) or sha(path)
        assert expected is None or expected == digest, str(path)
        pins[str(path)] = digest
        return path

    for path in Path(__file__).parent.glob('*.py'):
        pin(path)
    protocol = read(pin(args.protocol))
    assert protocol['version'] == SCOPE and protocol['legacy_width'] == LEGACY_WIDTH and protocol['input_width'] == WIDTH
    assert protocol['active_critic_optimizer'] == 'new_matched_adam_source_optimizer_retained_as_anchor'
    source_seal = read(pin(args.source/'complete.json'))
    for name, key in (('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'), ('report.json', 'report_sha256')):
        pin(args.source/name, source_seal[key])
    assert source_seal['weights_sha256'] == protocol['initial_source_weights_sha256']
    assert source_seal['checkpoint_sha256'] == protocol['initial_source_checkpoint_sha256']
    model = read(args.source/'weights.json')
    assert model['feature_version'] == 'combat_features_v10'
    require_same(LEGACY, model.get('combat_execution_contract'))
    cp = torch.load(args.source/'checkpoint.pt', map_location='cuda', weights_only=True)
    require_checkpoint_scope(cp, SOURCE_SCOPE)
    original = copy.deepcopy(cp)
    old_core = base_model(model, allow_shared=True)
    old_actor, old_value = branch(old_core, 'actor'), branch(old_core, 'value')
    assert equal_tree(old_actor.state_dict(), cp['actor']) and equal_tree(old_value.state_dict(), cp['value'])
    assert torch.equal(torch.tensor(model['log_std'], device='cuda'), cp['log_std'])
    version = VERSION if args.mode == 'observed' else ABLATED_VERSION
    updated = copy.deepcopy(model)
    updated['feature_version'] = version
    for name in ('actor', 'value'):
        for row in updated[name][0]['weight']:
            assert len(row) == LEGACY_WIDTH
            row.extend([0.]*(WIDTH-LEGACY_WIDTH))
    core = base_model(updated, allow_shared=True)
    actor, value = branch(core, 'actor'), branch(core, 'value')
    for before, after in ((old_actor, actor), (old_value, value)):
        old_state, state = before.state_dict(), after.state_dict()
        assert old_state.keys() == state.keys()
        for name, tensor in state.items():
            if name.endswith('encoder.0.weight'):
                assert tensor.shape[1] == WIDTH and old_state[name].shape[1] == LEGACY_WIDTH
                assert torch.equal(tensor[:, :LEGACY_WIDTH], old_state[name]) and not bool(tensor[:, LEGACY_WIDTH:].any())
            else:
                assert torch.equal(tensor, old_state[name]), name

    receipt = read(pin(args.corpus/'complete.json'))
    assert receipt['state'] == 'complete' and receipt['native_cases'] == receipt['cuda_cases'] == 4 and not receipt['training_performed']
    processed = receipt['processed'][0]
    assert processed['label'] == 'train'
    processing = read(pin(processed['report'], processed['sha256']))
    assert processing['model_sha256'] == source_seal['weights_sha256'] and processing['device'] == 'cuda'
    data = Path(processed['report']).parent/'merged'
    meta = read(pin(data/'report.json'))
    assert meta['feature_version'] == 'combat_features_v10' and meta['rows'] == 501
    with pin(data/'rollout.jsonl', meta['rollout_sha256']).open(encoding='utf-8-sig') as stream:
        rows = list(map(json.loads, stream))
    with pin(data/'sequence.jsonl', meta['sequence_sha256']).open(encoding='utf-8-sig') as stream:
        context = list(map(json.loads, stream))
    observations = {}
    for case in processing['cases']:
        directory = Path(processed['report']).parent/f"case-{case['job']:04d}"
        case_meta = read(pin(directory/'cuda/report.json'))
        steps = directory/'native/replay-0/steps.jsonl'
        pin(steps, case_meta['source_sha256'][str(steps)])
        with steps.open(encoding='utf-8-sig') as stream:
            for step in map(json.loads, stream):
                observations[case['seed'], step['index']] = step['observation']
    prepared = prepare_context(context, rows, 'cuda')
    source_x, indices, inverse, segments = prepared
    tails = []
    for segment in segments:
        encoded = encode_tail([observations[c['seed'], c['index']] for c in segment], version)
        tails.append(nn.functional.pad(encoded, (0, 0, 0, source_x.shape[1]-len(segment))))
    expanded_x = torch.cat((source_x, torch.stack(tails)), -1)
    expanded = (expanded_x, indices, inverse, segments)
    x = expanded_x[indices[:, 0], indices[:, 1]][inverse]
    source_features = source_x[indices[:, 0], indices[:, 1]][inverse]
    assert x.shape == (len(rows), WIDTH) and torch.equal(x[:, :LEGACY_WIDTH], source_features)
    category = lambda name: torch.tensor([row['sample'].get(name, 0) for row in rows], device='cuda', dtype=torch.long)
    target, mode, vertical, weapon = (category(name) for name in ('target', 'aim_mode', 'vertical', 'weapon'))
    final = torch.tensor([row['sample']['latent'] for row in rows], device='cuda')
    attack = torch.tensor([row['sample']['attack'] for row in rows], device='cuda', dtype=torch.float32)
    source_z, choice, timing, _ = events(rows, x, target, mode, vertical, weapon)
    joint = GeometryJoint(updated); joint.requires_grad_(False)
    std = nn.Parameter(torch.tensor(updated['log_std'], device='cuda'))

    def likelihood(module, raw, features, gaussian_std):
        return probabilities(joint, raw, gaussian_std, source_z, final, attack, vertical, weapon, features, target, mode,
            choice, timing, module.sampled_motion_aim_branch, module.motor_branch, module.conditional_fire_branch)

    with torch.no_grad():
        old_raw, old_memory = sequence(old_actor, prepared, 32, True, 'actor')
        raw, memory = sequence(actor, expanded, 32, True, 'actor')
        old_values, old_value_memory = sequence(old_value, prepared, 32, True, 'value')
        values, value_memory = sequence(value, expanded, 32, True, 'value')
        old_lp = likelihood(old_actor, old_raw, source_features, std)[0]
        lp = likelihood(actor, raw, x, std)[0]
        raw_error = float((raw-old_raw).abs().max())
        value_error = float((values-old_values).abs().max())
        lp_error = float((lp-old_lp).abs().max())
        assert max(raw_error, value_error, old_memory, memory, old_value_memory, value_memory) < 5e-5
        assert lp_error < .0005
    legacy = actor_optimizer(actor, std, 3e-5, .0003, .0003, .0003)
    legacy.load_state_dict(copy.deepcopy(cp['actor_optimizer']))
    active = BodyScope(actor, std, legacy, inherited_optimizer=cp['execution_contract_optimizer'])
    active_state = copy.deepcopy(active.optimizer.state_dict())
    active.materialize(); active.verify_frozen()
    assert equal_tree(legacy.state_dict(), cp['actor_optimizer'])
    critic = torch.optim.Adam(value.parameters(), lr=cp['config']['value_lr'])
    fork = copy.deepcopy(cp)
    fork.update(checkpoint_training_scope=SCOPE, actor=actor.state_dict(), value=value.state_dict(),
        config=dict(cp['config'], body_feature_version=version), body_input_optimizer=active_state,
        body_value_optimizer=critic.state_dict(), body_input_history=[],
        body_input_scope=dict(version=SCOPE, protocol_sha256=sha(args.protocol), updates_completed=0,
            feature_version=version, initial_source_weights_sha256=source_seal['weights_sha256'],
            active_critic_optimizer=protocol['active_critic_optimizer']))
    args.out.mkdir(parents=True)
    durable_json(args.out/'weights.json', updated)
    fork['weights_sha256'] = sha(args.out/'weights.json')
    durable_write(args.out/'checkpoint.pt', lambda stream: torch.save(fork, stream))
    loaded = torch.load(args.out/'checkpoint.pt', map_location='cuda', weights_only=True)
    require_checkpoint_scope(loaded, SCOPE)
    assert equal_tree(fork, loaded) and equal_tree(cp, original)
    for key in cp:
        if key not in ('actor', 'value', 'config', 'checkpoint_training_scope', 'weights_sha256'):
            assert equal_tree(cp[key], loaded[key]), key
    assert not loaded['body_value_optimizer']['state']
    adv, _ = advantages(rows, .99, .95)
    advantage = torch.tensor(adv, device='cuda')
    advantage = (advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
    continued, gradient_norms, body_changes = [], [], []
    for state in (fork, loaded):
        module = branch(core, 'actor')
        probe_std = nn.Parameter(torch.tensor(updated['log_std'], device='cuda'))
        anchor = actor_optimizer(module, probe_std, 3e-5, .0003, .0003, .0003)
        anchor.load_state_dict(copy.deepcopy(state['actor_optimizer']))
        probe = BodyScope(module, probe_std, anchor, saved_optimizer=state['body_input_optimizer'])
        before_body = probe.input.weight.detach().clone()
        probe_raw, _ = sequence(module, expanded, 32, False, 'actor')
        probe_lp, entropy, _, _ = likelihood(module, probe_raw, x, probe_std)
        loss = -((probe_lp-old_lp).exp()*advantage).mean()-.001*entropy.mean()
        probe.optimizer.zero_grad(set_to_none=True); loss.backward()
        gradient_norms.append(float(probe.input.weight.grad.norm()))
        nn.utils.clip_grad_norm_(probe.parameters, .5); probe.optimizer.step()
        body_changes.append(not torch.equal(before_body, probe.input.weight))
        probe.materialize(); probe.verify_frozen()
        continued.append((copy.deepcopy(module.state_dict()), copy.deepcopy(probe.optimizer.state_dict())))
    assert equal_tree(continued[0], continued[1])
    if args.mode == 'observed':
        assert min(gradient_norms) > 0 and all(body_changes)
    else:
        assert not any(gradient_norms) and not any(body_changes)
    for expected in (None, SOURCE_SCOPE):
        try:
            require_checkpoint_scope(loaded, expected)
        except ValueError:
            pass
        else:
            raise AssertionError('Old trainer admitted body scope')
    assert all(sha(path) == digest for path, digest in pins.items())
    report = dict(state='complete', version='combat_body_input_initializer_v1', device='cuda', feature_version=version,
        source_weights_sha256=source_seal['weights_sha256'], protocol_sha256=sha(args.protocol),
        legacy_input_weights_exact=True, zero_body_columns_verified=True, inherited_other_tensors_exact=True,
        source_optimizers_retained_exact=True, inherited_decision_adam_exact=True,
        active_critic_optimizer=protocol['active_critic_optimizer'], source_rng_exact=True,
        raw_max_error=raw_error, value_max_error=value_error, likelihood_max_error=lp_error,
        checkpoint_roundtrip_exact=True, disposable_resume_adam_step_exact=True,
        disposable_body_gradient_norm=gradient_norms[0], disposable_body_columns_changed=body_changes[0],
        source_unchanged=True, fitting_performed=False, scope_updates=0, inputs=pins,
        scope='Widened inputs preserve the v10 prefix and source tensors. Both forks retain old optimizer states as '
              'anchors and inherit the five decision groups; the new body group starts cold. A dedicated matched '
              'critic Adam starts empty in both forks, old critic Adam retained. Real TRAIN history is used for '
              'CUDA parity and one disposable PPO/Adam gradient/resume proof only; exported body columns remain zero.')
    durable_json(args.out/'report.json', report)
    durable_json(args.out/'complete.json', dict(state='complete', weights_sha256=sha(args.out/'weights.json'),
        checkpoint_sha256=sha(args.out/'checkpoint.pt'), report_sha256=sha(args.out/'report.json')))
    print({key: value for key, value in report.items() if key not in ('inputs', 'scope')})


if __name__ == '__main__':
    main()

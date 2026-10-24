"""Fork a symmetric seven-group body-input scope without changing policy weights."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from evaluate_registered_combat_models import read, sha
from ppo_combat import torch, nn, training_devices, advantages
from ppo_recurrent import prepare_context, sequence, durable_json, durable_write
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model, events, probabilities
from combat_geometry_ppo import GeometryJoint
from combat_checkpoint_scope import require_checkpoint_scope
from combat_execution_contract import LEGACY, require_same
from combat_body_features import VERSION, ABLATED_VERSION, LEGACY_WIDTH, WIDTH, SCOPE as INPUT_SCOPE, encode_tail
from combat_body_training_scope import BodyTrainingScope, SCOPE, RATES
from combat_weapon_state_residual import optimizer as actor_optimizer
from fork_conditional_weapon_exploration_cuda import equal_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'corpus', 'protocol', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--mode', choices=('observed', 'ablated'), required=True)
    args = parser.parse_args(); assert not args.out.exists()
    training_devices(); torch.set_default_device('cuda'); torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve(); digest = pins.get(str(path)) or sha(path)
        assert expected is None or expected == digest, str(path)
        pins[str(path)] = digest; return path

    for path in Path(__file__).parent.glob('*.py'):
        pin(path)
    protocol = read(pin(args.protocol))
    assert protocol['version'] == SCOPE and protocol['group_lrs'] == list(RATES)
    assert protocol['shared_input_bias'] is True and protocol['input_width'] == WIDTH
    expected = protocol['initial_sources'][args.mode]
    source_seal = read(pin(args.source/'complete.json'))
    for name, key in (('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'), ('report.json', 'report_sha256')):
        pin(args.source/name, source_seal[key])
        assert source_seal[key] == expected[key]
    source_report = read(args.source/'report.json')
    assert source_report['source_weights_sha256'] == protocol['root_source_weights_sha256']
    parity = read(pin(protocol['parity_audit']))
    assert parity['gate_passed'] and parity['native_cases'] == parity['cuda_cases'] == 12
    assert all(parity['checks'].values())
    for path,digest in parity['inputs'].items():
        pin(path,digest)
    fixture_path = next(Path(path) for path in parity['inputs'] if Path(path).name == 'fixture-proof.json')
    fixture = read(fixture_path)
    parity_label = 'body-initial' if args.mode == 'observed' else 'body-ablated-initial'
    assert next(item for item in fixture['models'] if item['label'] == parity_label)['model_sha256'] == source_seal['weights_sha256']
    model = read(args.source/'weights.json')
    version = VERSION if args.mode == 'observed' else ABLATED_VERSION
    assert model['feature_version'] == version and source_report['feature_version'] == version
    require_same(LEGACY, model.get('combat_execution_contract'))
    cp = torch.load(args.source/'checkpoint.pt', map_location='cuda', weights_only=True)
    original = copy.deepcopy(cp); require_checkpoint_scope(cp, INPUT_SCOPE)
    assert cp['body_input_scope']['updates_completed'] == 0 and not cp['body_value_optimizer']['state']
    assert cp['config']['body_feature_version'] == version and cp['weights_sha256'] == source_seal['weights_sha256']
    core = base_model(model, allow_shared=True)
    actor, value = branch(core, 'actor'), branch(core, 'value')
    assert equal_tree(actor.state_dict(), cp['actor']) and equal_tree(value.state_dict(), cp['value'])
    std = nn.Parameter(torch.tensor(model['log_std'], device='cuda')); assert torch.equal(std, cp['log_std'])
    anchor = actor_optimizer(actor, std, 3e-5, .0003, .0003, .0003)
    anchor.load_state_dict(copy.deepcopy(cp['actor_optimizer']))
    scope = BodyTrainingScope(actor, std, anchor, inherited_optimizer=cp['body_input_optimizer'])
    active = copy.deepcopy(scope.optimizer.state_dict())
    assert active['param_groups'][:6] == cp['body_input_optimizer']['param_groups']
    assert equal_tree(active['state'], cp['body_input_optimizer']['state'])
    scope.materialize(); scope.verify_frozen()
    assert equal_tree(actor.state_dict(), cp['actor']) and equal_tree(anchor.state_dict(), cp['actor_optimizer'])
    fork = copy.deepcopy(cp)
    metadata = scope.metadata(0)
    metadata.update(protocol_sha256=sha(args.protocol), feature_version=version,
        initial_source_weights_sha256=source_seal['weights_sha256'], root_source_weights_sha256=protocol['root_source_weights_sha256'],
        combat_execution_contract=LEGACY)
    fork.update(checkpoint_training_scope=SCOPE, body_training_scope=metadata,
        body_training_optimizer=active, body_training_history=[])
    args.out.mkdir(parents=True)
    shutil.copyfile(args.source/'weights.json', args.out/'weights.json')
    assert sha(args.out/'weights.json') == source_seal['weights_sha256']
    durable_write(args.out/'checkpoint.pt', lambda stream: torch.save(fork, stream))
    loaded = torch.load(args.out/'checkpoint.pt', map_location='cuda', weights_only=True)
    require_checkpoint_scope(loaded, SCOPE)
    assert equal_tree(fork, loaded) and equal_tree(cp, original)
    for key in cp:
        if key != 'checkpoint_training_scope':
            assert equal_tree(cp[key], loaded[key]), key

    receipt = read(pin(args.corpus/'complete.json')); assert receipt['state'] == 'complete' and not receipt['training_performed']
    binding = receipt['processed'][0]; assert binding['label'] == 'train'
    processing = read(pin(binding['report'], binding['sha256']))
    assert processing['device'] == 'cuda' and processing['model_sha256'] == protocol['root_source_weights_sha256']
    data = Path(binding['report']).parent/'merged'; meta = read(pin(data/'report.json'))
    with pin(data/'rollout.jsonl', meta['rollout_sha256']).open(encoding='utf-8-sig') as stream:
        rows = list(map(json.loads, stream))
    with pin(data/'sequence.jsonl', meta['sequence_sha256']).open(encoding='utf-8-sig') as stream:
        context = list(map(json.loads, stream))
    observations = {}
    for case in processing['cases']:
        directory = Path(binding['report']).parent/f"case-{case['job']:04d}"
        case_meta = read(pin(directory/'cuda/report.json')); path = directory/'native/replay-0/steps.jsonl'
        pin(path, case_meta['source_sha256'][str(path)])
        with path.open(encoding='utf-8-sig') as stream:
            for step in map(json.loads, stream):
                observations[case['seed'], step['index']] = step['observation']
    source_x, indices, inverse, segments = prepare_context(context, rows, 'cuda')
    tails = [nn.functional.pad(encode_tail([observations[row['seed'], row['index']] for row in segment], version),
        (0, 0, 0, source_x.shape[1]-len(segment))) for segment in segments]
    x_all = torch.cat((source_x, torch.stack(tails)), -1)
    prepared = (x_all, indices, inverse, segments); x = x_all[indices[:, 0], indices[:, 1]][inverse]
    category = lambda name: torch.tensor([row['sample'].get(name, 0) for row in rows], device='cuda', dtype=torch.long)
    target, mode, vertical, weapon = (category(name) for name in ('target', 'aim_mode', 'vertical', 'weapon'))
    final = torch.tensor([row['sample']['latent'] for row in rows], device='cuda')
    attack = torch.tensor([row['sample']['attack'] for row in rows], device='cuda', dtype=torch.float32)
    source_z, choice, timing, _ = events(rows, x, target, mode, vertical, weapon)
    native_lp = torch.tensor([row['sample']['log_probability'] for row in rows], device='cuda')
    adv, _ = advantages(rows, .99, .95); advantage = torch.tensor(adv, device='cuda')
    advantage = (advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
    joint = GeometryJoint(model); joint.requires_grad_(False)
    results, gradients = [], []
    for state in (fork, loaded):
        module = branch(core, 'actor'); probe_std = nn.Parameter(torch.tensor(model['log_std'], device='cuda'))
        legacy = actor_optimizer(module, probe_std, 3e-5, .0003, .0003, .0003)
        legacy.load_state_dict(copy.deepcopy(state['actor_optimizer']))
        probe = BodyTrainingScope(module, probe_std, legacy, saved_optimizer=state['body_training_optimizer'])
        before_body, before_bias = probe.input.weight.detach().clone(), probe.input.prefix_bias.detach().clone()
        raw, memory_error = sequence(module, prepared, 32, True, 'actor'); assert memory_error < 5e-5
        lp, entropy, _, _ = probabilities(joint, raw, probe_std, source_z, final, attack, vertical, weapon, x, target, mode,
            choice, timing, module.sampled_motion_aim_branch, module.motor_branch, module.conditional_fire_branch)
        assert float((lp-native_lp).abs().max()) < .003
        loss = -((lp-native_lp).exp()*advantage).mean()-.001*entropy.mean()
        probe.optimizer.zero_grad(set_to_none=True); loss.backward()
        body_norm, bias_norm = float(probe.input.weight.grad.norm()), float(probe.input.prefix_bias.grad.norm())
        assert bias_norm > 0
        nn.utils.clip_grad_norm_(probe.parameters, .5); probe.optimizer.step()
        assert not torch.equal(before_bias, probe.input.prefix_bias)
        if args.mode == 'observed':
            assert body_norm > 0 and not torch.equal(before_body, probe.input.weight)
        else:
            assert body_norm == 0 and torch.equal(before_body, probe.input.weight)
        probe.materialize(); probe.verify_frozen()
        gradients.append(dict(body=body_norm, shared_bias=bias_norm))
        results.append((copy.deepcopy(module.state_dict()), copy.deepcopy(probe.optimizer.state_dict())))
    assert equal_tree(results[0], results[1])
    for old in (None, INPUT_SCOPE, 'combat_execution_contract_ppo_v1'):
        try:
            require_checkpoint_scope(loaded, old)
        except ValueError:
            pass
        else:
            raise AssertionError('Old trainer admitted new seven-group scope')
    assert all(sha(path) == digest for path, digest in pins.items())
    report = dict(state='complete', version='combat_body_training_initializer_v2', device='cuda', feature_version=version,
        combat_execution_contract=LEGACY, source_weights_sha256=source_seal['weights_sha256'], protocol_sha256=sha(args.protocol),
        source_training_tensors_exact=True, source_optimizers_exact=True, source_rng_exact=True,
        inherited_six_groups_exact=True, shared_input_bias=True, disposable_gradients=gradients[0],
        checkpoint_roundtrip_exact=True, disposable_resume_adam_step_exact=True, source_unchanged=True,
        training_performed=False, fitting_performed=False, scope_updates=0, inputs=pins,
        scope='Identical JSON weights and CUDA tensors. Six active Adam groups inherited; identical fresh shared-input-bias '
              'group in both arms. Dedicated critic optimizer retained from v1 initialization. TRAIN history only for two '
              'disposable gradient/Adam continuation proofs, not a consumed on-policy update or a quality result.')
    durable_json(args.out/'report.json', report)
    durable_json(args.out/'complete.json', dict(state='complete', weights_sha256=sha(args.out/'weights.json'),
        checkpoint_sha256=sha(args.out/'checkpoint.pt'), report_sha256=sha(args.out/'report.json')))
    print({key: value for key, value in report.items() if key not in ('inputs', 'scope')})


if __name__ == '__main__':
    main()

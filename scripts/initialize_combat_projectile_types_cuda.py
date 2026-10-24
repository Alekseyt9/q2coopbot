"""Zero-column CUDA migration with inherited actor and widened critic Adam state."""
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
from combat_body_training_scope import SCOPE as PARENT_SCOPE
from combat_projectile_type_features import VERSION, NEUTRAL_VERSION, PARENT_WIDTH, WIDTH, encode_tail
from combat_projectile_training_scope import SCOPE, ProjectileTrainingScope, widen_value_optimizer
from combat_weapon_state_residual import optimizer as actor_optimizer
from fork_conditional_weapon_exploration_cuda import equal_tree


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'corpus', 'protocol', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--mode', choices=('typed', 'neutral'), required=True)
    a = ap.parse_args()
    assert not a.out.exists()
    training_devices()
    torch.set_default_device('cuda'); torch.set_num_threads(2)
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
    protocol = read(pin(a.protocol))
    assert protocol['version'] == SCOPE and protocol['parent_width'] == PARENT_WIDTH and protocol['width'] == WIDTH
    seal = read(pin(a.source / 'complete.json', protocol['source_seal_sha256']))
    for name, field in (('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'), ('report.json', 'report_sha256')):
        pin(a.source / name, seal[field])
    assert seal['weights_sha256'] == protocol['source_weights_sha256']
    model = read(a.source / 'weights.json')
    assert model['feature_version'] == 'combat_features_v11_body_ablated'
    require_same(LEGACY, model.get('combat_execution_contract'))
    cp = torch.load(a.source / 'checkpoint.pt', map_location='cuda', weights_only=True)
    original = copy.deepcopy(cp)
    require_checkpoint_scope(cp, PARENT_SCOPE)
    old_core = base_model(model, allow_shared=True)
    old_actor, old_value = branch(old_core, 'actor'), branch(old_core, 'value')
    assert equal_tree(old_actor.state_dict(), cp['actor']) and equal_tree(old_value.state_dict(), cp['value'])
    version = VERSION if a.mode == 'typed' else NEUTRAL_VERSION
    updated = copy.deepcopy(model)
    updated['feature_version'] = version
    for name in ('actor', 'value'):
        for row in updated[name][0]['weight']:
            assert len(row) == PARENT_WIDTH
            row.extend([0.] * (WIDTH - PARENT_WIDTH))
    core = base_model(updated, allow_shared=True)
    actor, value = branch(core, 'actor'), branch(core, 'value')
    for module, parent in ((actor, old_actor), (value, old_value)):
        previous = parent.state_dict()
        for name, tensor in module.state_dict().items():
            if name.endswith('encoder.0.weight'):
                assert torch.equal(tensor[:, :PARENT_WIDTH], previous[name]) and not bool(tensor[:, PARENT_WIDTH:].any())
            else:
                assert torch.equal(tensor, previous[name]), name
    std = nn.Parameter(torch.tensor(updated['log_std'], device='cuda'))
    assert torch.equal(std, cp['log_std'])
    anchor = actor_optimizer(actor, std, 3e-5, .0003, .0003, .0003)
    anchor.load_state_dict(copy.deepcopy(cp['actor_optimizer']))
    scope = ProjectileTrainingScope(actor, std, anchor, inherited_optimizer=cp['body_training_optimizer'])
    active = copy.deepcopy(scope.optimizer.state_dict())
    assert active['param_groups'][:7] == cp['body_training_optimizer']['param_groups']
    assert equal_tree(active['state'], cp['body_training_optimizer']['state'])
    scope.materialize(); scope.verify_frozen()
    critic_state = widen_value_optimizer(cp['body_value_optimizer'], value)
    critic = torch.optim.Adam(value.parameters(), lr=.001)
    critic.load_state_dict(copy.deepcopy(critic_state))
    assert equal_tree(critic.state_dict(), critic_state)
    metadata = scope.metadata(0)
    metadata.update(feature_version=version, protocol_sha256=sha(a.protocol),
        source_weights_sha256=seal['weights_sha256'], combat_execution_contract=LEGACY)
    fork = copy.deepcopy(cp)
    fork.update(checkpoint_training_scope=SCOPE, actor=actor.state_dict(), value=value.state_dict(),
        config=dict(cp['config'], projectile_type_feature_version=version),
        projectile_training_scope=metadata, projectile_training_optimizer=active,
        projectile_value_optimizer=critic_state, projectile_training_history=[])
    a.out.mkdir(parents=True)
    durable_json(a.out / 'weights.json', updated)
    fork['weights_sha256'] = sha(a.out / 'weights.json')
    durable_write(a.out / 'checkpoint.pt', lambda stream: torch.save(fork, stream))
    loaded = torch.load(a.out / 'checkpoint.pt', map_location='cuda', weights_only=True)
    require_checkpoint_scope(loaded, SCOPE)
    assert equal_tree(fork, loaded) and equal_tree(cp, original)
    for key in cp:
        if key not in ('checkpoint_training_scope', 'actor', 'value', 'config', 'weights_sha256'):
            assert equal_tree(cp[key], loaded[key]), key
    assert all(equal_tree(v, loaded['config'][k]) for k, v in cp['config'].items())

    # These earlier own TRAIN histories were already consumed by the parent.
    # They are used only for equivalence/disposable resume, not a PPO update.
    collection = read(pin(a.corpus / 'complete.json', protocol['corpus_seal_sha256']))
    assert collection['state'] == 'complete' and collection['native_cases'] == collection['cuda_cases'] == 24
    collected = read(pin(a.corpus / 'protocol.json', collection['protocol_sha256']))
    assert collected['train_only']
    binding = collection['processed'][0]
    processing = read(pin(binding['report'], binding['sha256']))
    assert processing['model_sha256'] == protocol['corpus_behavior_weights_sha256']
    data = Path(binding['report']).parent
    meta = read(pin(data / 'merged/report.json'))
    assert meta['feature_version'] == model['feature_version']
    assert meta['rollout_sha256'] in cp['consumed_rollouts']
    def lines(path):
        with path.open(encoding='utf-8-sig') as stream:
            return [json.loads(line) for line in stream]
    rows = lines(pin(data / 'merged/rollout.jsonl', meta['rollout_sha256']))
    context = lines(pin(data / 'merged/sequence.jsonl', meta['sequence_sha256']))
    observations = {}
    for case in processing['cases']:
        folder = data / f"case-{case['job']:04d}"
        case_meta = read(pin(folder / 'cuda/report.json'))
        steps = next((folder / 'native').glob('replay-*/steps.jsonl')).resolve()
        for step in lines(pin(steps, case_meta['source_sha256'][str(steps)])):
            observations[case['seed'], step['index']] = step['observation']
    prepared = prepare_context(context, rows, 'cuda')
    source_x, indices, inverse, segments = prepared
    assert source_x.shape[-1] == PARENT_WIDTH
    tails = [nn.functional.pad(encode_tail([observations[r['seed'], r['index']] for r in segment], version),
        (0, 0, 0, source_x.shape[1] - len(segment))) for segment in segments]
    expanded_x = torch.cat((source_x, torch.stack(tails)), -1)
    expanded = (expanded_x, indices, inverse, segments)
    source_features = source_x[indices[:, 0], indices[:, 1]][inverse]
    features = expanded_x[indices[:, 0], indices[:, 1]][inverse]
    assert torch.equal(features[:, :PARENT_WIDTH], source_features)
    assert bool(features[:, PARENT_WIDTH:].any())
    category = lambda name: torch.tensor([r['sample'].get(name, 0) for r in rows], device='cuda', dtype=torch.long)
    target, mode, vertical, weapon = (category(name) for name in ('target', 'aim_mode', 'vertical', 'weapon'))
    final = torch.tensor([r['sample']['latent'] for r in rows], device='cuda')
    attack = torch.tensor([r['sample']['attack'] for r in rows], device='cuda', dtype=torch.float32)
    z, choice, timing, _ = events(rows, features, target, mode, vertical, weapon)
    joint = GeometryJoint(updated); joint.requires_grad_(False)
    def likelihood(module, raw):
        return probabilities(joint, raw, std, z, final, attack, vertical, weapon, features, target, mode,
            choice, timing, module.sampled_motion_aim_branch, module.motor_branch, module.conditional_fire_branch)
    with torch.no_grad():
        old_raw, _ = sequence(old_actor, prepared, 32, False, 'actor')
        raw, _ = sequence(actor, expanded, 32, False, 'actor')
        old_values, _ = sequence(old_value, prepared, 32, False, 'value')
        values, _ = sequence(value, expanded, 32, False, 'value')
        parent_lp = likelihood(old_actor, old_raw)[0]
        lp = likelihood(actor, raw)[0]
        errors = dict(raw=float((raw-old_raw).abs().max()), value=float((values-old_values).abs().max()),
            log_probability=float((lp-parent_lp).abs().max()))
        assert errors['raw'] < 5e-5 and errors['value'] < 1e-4 and errors['log_probability'] < .003, errors
    adv, returns = advantages(rows, .99, .95)
    advantage = torch.tensor(adv, device='cuda')
    advantage = (advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
    returns = torch.tensor(returns, device='cuda')
    actor_results, critic_results, gradients = [], [], []
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        for state in (fork, loaded):
            module = branch(core, 'actor')
            anchor = actor_optimizer(module, std, 3e-5, .0003, .0003, .0003)
            anchor.load_state_dict(copy.deepcopy(state['actor_optimizer']))
            probe = ProjectileTrainingScope(module, std, anchor, saved_optimizer=state['projectile_training_optimizer'])
            before = probe.input.projectile.detach().clone()
            output, _ = sequence(module, expanded, 32, False, 'actor')
            current_lp, entropy, _, _ = likelihood(module, output)
            loss = -((current_lp-parent_lp).exp()*advantage).mean()-.001*entropy.mean()
            probe.optimizer.zero_grad(set_to_none=True); loss.backward()
            gradient = dict(projectile=float(probe.input.projectile.grad.norm()), shared_bias=float(probe.input.bias.grad.norm()))
            assert gradient['projectile'] > 0 and gradient['shared_bias'] > 0
            nn.utils.clip_grad_norm_(probe.parameters, .5); probe.optimizer.step()
            assert not torch.equal(before, probe.input.projectile)
            probe.materialize(); probe.verify_frozen()
            actor_results.append((copy.deepcopy(module.state_dict()), copy.deepcopy(probe.optimizer.state_dict())))
            gradients.append(gradient)
            module = branch(core, 'value')
            optimizer = torch.optim.Adam(module.parameters(), lr=.001)
            optimizer.load_state_dict(copy.deepcopy(state['projectile_value_optimizer']))
            before = {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}
            prediction, _ = sequence(module, expanded, 32, False, 'value')
            loss = (prediction[:, 0]-returns).square().mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(module.parameters(), .5); optimizer.step()
            assert any(not torch.equal(before[k], v) for k, v in module.state_dict().items())
            critic_results.append((copy.deepcopy(module.state_dict()), copy.deepcopy(optimizer.state_dict())))
    assert equal_tree(actor_results[0], actor_results[1]) and equal_tree(critic_results[0], critic_results[1])
    for expected in (None, PARENT_SCOPE, 'combat_body_input_ppo_v1', 'combat_execution_contract_ppo_v1'):
        try:
            require_checkpoint_scope(loaded, expected)
        except ValueError:
            pass
        else:
            raise AssertionError('Old trainer admitted projectile scope')
    assert equal_tree(cp, original) and all(sha(p) == h for p, h in pins.items())
    frozen = a.out / 'initializer-sources'; frozen.mkdir()
    helpers = {}
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path, frozen / path.name); helpers[path.name] = sha(frozen / path.name)
    report = dict(state='complete', version=SCOPE, device='cuda', feature_version=version,
        source_weights_sha256=seal['weights_sha256'], weights_sha256=sha(a.out / 'weights.json'),
        protocol_sha256=sha(a.protocol), combat_execution_contract=LEGACY, scope_updates=0,
        source_training_tensors_exact=True, inherited_seven_actor_groups_exact=True,
        critic_old_moment_prefix_exact=True, critic_step_counters_preserved=True, source_rng_exact=True,
        checkpoint_roundtrip_exact=True, disposable_actor_resume_adam_step_exact=True,
        disposable_critic_resume_adam_step_exact=True, old_scopes_rejected=True,
        errors=errors, gradients=gradients[0], parity_train_rows=len(rows),
        training_performed=False, fitting_performed=False, source_unchanged=True, inputs=pins, initializer_sources=helpers,
        scope='Zero input-column migration. Seven inherited actor groups plus one fresh type group; active critic '
              'input moments padded with zeros, other moments/clocks preserved. Earlier consumed TRAIN histories only '
              'for fixed-parent likelihood equivalence and disposable Adam resume, never a new on-policy update or quality result.')
    durable_json(a.out / 'report.json', report)
    durable_json(a.out / 'complete.json', dict(state='complete', weights_sha256=report['weights_sha256'],
        checkpoint_sha256=sha(a.out / 'checkpoint.pt'), report_sha256=sha(a.out / 'report.json')))
    print({k: v for k, v in report.items() if k not in ('inputs', 'initializer_sources', 'scope')})


if __name__ == '__main__':
    main()

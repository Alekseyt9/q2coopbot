"""Disposable own-policy PPO probe with all Gaussian movement/aim parameters frozen."""
import argparse
import copy
import json
from pathlib import Path
import sys

from evaluate_registered_combat_models import read, save, sha
from ppo_combat import torch, nn, training_devices, advantages, guarded_actor_step, validate_objective
from ppo_recurrent import prepare_context, sequence
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model, events, probabilities
from combat_geometry_ppo import GeometryJoint
from combat_precision_head import selected_means, targets
from combat_weapon_head import distribution as weapons, feature_mask
from combat_conditional_fire import logits as fire_logits
from combat_weapon_state_residual import optimizer as actor_optimizer
from combat_checkpoint_scope import DECISION_SCOPE, require_checkpoint_scope
from combat_decision_scope import DecisionScope
from fork_conditional_weapon_exploration_cuda import equal_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    training_devices()
    torch.set_default_device('cuda')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve(); digest = sha(path)
        assert expected is None or expected == digest, str(path)
        pins[str(path)] = digest
        return path

    def sealed(root):
        receipt = read(pin(root / 'complete.json'))
        assert receipt['state'] == 'complete'
        return receipt, read(pin(root / 'report.json', receipt['report_sha256']))

    pin(__file__)
    protocol = read(pin(args.protocol))
    assert protocol['version'] == 'combat_categorical_decision_probe_v1'
    assert protocol['actor_steps'] == 30 and protocol['group_lrs'] == [3e-5, 3e-5, .0003, .0003]
    assert protocol['full_sample_k3_max'] == .005 and protocol['base_sample_k3_max'] == .0025
    assert protocol['clip'] == .2 and protocol['entropy'] == .001 and protocol['gradient_norm'] == .5
    assert protocol['mechanical_gate'] == dict(selected_conditional_kl_ratio_to_control_at_least=3,
        gaussian_means_exact=True, std_exact=True, source_checkpoint_exact=True, source_optimizers_exact=True, guard_passed=True)
    source, corpus, control = (Path(protocol[name]) for name in ('source', 'corpus', 'control_update'))
    source_seal, source_report = sealed(source)
    control_seal, control_report = sealed(control)
    _, reference = sealed(Path(protocol['factor_audit']))
    assert sha(Path(protocol['factor_audit']) / 'report.json') == protocol['factor_audit_report_sha256']
    assert reference['device'] == 'cuda' and reference['inputs'][str((control / 'weights.json').resolve())] == control_seal['weights_sha256']
    control_factors = next(c['conditional_factor_kl'] for c in reference['cases'] if c['branch'] == 'balanced')
    assert source_seal['weights_sha256'] == protocol['source_weights_sha256'] == control_report['behavior_sha256']
    assert control_seal['weights_sha256'] == protocol['control_weights_sha256']
    model = read(pin(source / 'weights.json', source_seal['weights_sha256']))
    cp = torch.load(pin(source / 'checkpoint.pt', source_seal['checkpoint_sha256']), map_location='cuda', weights_only=True)
    snapshot = copy.deepcopy(cp)
    require_checkpoint_scope(cp, DECISION_SCOPE)
    assert cp['decision_scope']['updates_completed'] == protocol['source_scope_updates'] == 4
    assert cp['config']['clip'] == .2 and cp['config']['entropy'] == .001
    closed = read(pin(corpus / 'complete.json'))
    collection = read(pin(corpus / 'protocol.json', closed['protocol_sha256']))
    assert collection['train_only'] and closed['native_cases'] == closed['cuda_cases'] == protocol['cases'] == 24
    assert len(collection['assignments']) == 1 and collection['assignments'][0]['split'] == 'train'
    data = corpus / 'processing-train/merged'
    meta = read(pin(data / 'report.json'))
    assert meta['model_sha256'] == source_seal['weights_sha256'] and meta['numerical_verification'] == 'cuda_verified_v1'
    assert meta['rollout_sha256'] == control_report['rollout_sha256'] and meta['rollout_sha256'] not in cp['consumed_rollouts']
    validate_objective(cp['config'], meta)
    for name in ('native-credit', 'advantage-credit'):
        _, audit = sealed(corpus.parent / name)
        assert audit.get('rows', audit.get('totals', {}).get('rows')) == meta['rows']
    with pin(data / 'rollout.jsonl', meta['rollout_sha256']).open(encoding='utf-8-sig') as stream:
        rows = list(map(json.loads, stream))
    with pin(data / 'sequence.jsonl', meta['sequence_sha256']).open(encoding='utf-8-sig') as stream:
        context = list(map(json.loads, stream))
    assert len(rows) == protocol['rows'] == 2889
    assert {r['seed'] for r in rows} == set(collection['assignments'][0]['seeds'])
    prepared = prepare_context(context, rows, 'cuda')
    x = torch.tensor([r['features'] for r in rows], device='cuda', dtype=torch.float32)
    category = lambda key: torch.tensor([r['sample'].get(key, 0) for r in rows], device='cuda', dtype=torch.long)
    target, mode, vertical, weapon = (category(k) for k in ('target', 'aim_mode', 'vertical', 'weapon'))
    final = torch.tensor([r['sample']['latent'] for r in rows], device='cuda', dtype=torch.float32)
    attack = torch.tensor([float(r['sample']['attack']) for r in rows], device='cuda')
    source_z, choice, timing, active = events(rows, x, target, mode, vertical, weapon)
    actor, joint = branch(base_model(model, allow_shared=True), 'actor'), GeometryJoint(model)
    std = nn.Parameter(torch.tensor(model['log_std'], device='cuda'))
    assert equal_tree(actor.state_dict(), cp['actor'])
    legacy = actor_optimizer(actor, std, 3e-5, .0003, .0003, .0003)
    legacy.load_state_dict(copy.deepcopy(cp['actor_optimizer']))

    def likelihood(raw):
        lp, entropy, _, current_active = probabilities(joint, raw, std, source_z, final, attack, vertical, weapon,
            x, target, mode, choice, timing, actor.sampled_motion_aim_branch, actor.motor_branch, actor.conditional_fire_branch)
        assert torch.equal(current_active, active)
        return lp, entropy

    with torch.no_grad():
        old_raw, memory_error = sequence(actor, prepared, 32, True, 'actor')
        old_lp, _ = likelihood(old_raw)
        native_lp = torch.tensor([r['sample']['log_probability'] for r in rows], device='cuda')
        replay_error = float((old_lp - native_lp).abs().max())
        assert replay_error < .003 and memory_error < 2e-5
        old_means = selected_means(old_raw, target, mode, x, source_z[:, :2], actor.sampled_motion_aim_branch, vertical, actor.motor_branch)
        old_fire = torch.distributions.Bernoulli(logits=fire_logits(actor.conditional_fire_branch, x, old_raw, target, mode, final, vertical, weapon))
    scope = DecisionScope(actor, std, legacy, cp['decision_scope_optimizer'])
    untouched_optimizer = copy.deepcopy(scope.optimizer.state_dict())
    frozen_motor = copy.deepcopy(actor.motor_branch.state_dict())
    actor.motor_branch.requires_grad_(False)
    joint.requires_grad_(False)
    groups = []
    for group in scope.optimizer.param_groups:
        if group['name'] == 'motor_branch':
            continue
        groups.append(dict({k: copy.deepcopy(v) for k, v in group.items() if k != 'params'}, params=list(group['params'])))
    assert [g['name'] for g in groups] == protocol['groups']
    optimizer = torch.optim.Adam(groups)
    parameters = [p for group in groups for p in group['params']]
    for p in parameters:
        if scope.optimizer.state.get(p):
            optimizer.state[p] = copy.deepcopy(scope.optimizer.state[p])
    assert [g['lr'] for g in optimizer.param_groups] == protocol['group_lrs']
    adv, _ = advantages(rows, cp['config']['gamma'], cp['config']['lambda'])
    advantage = torch.tensor(adv, device='cuda')
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)

    def metrics():
        raw, _ = sequence(actor, prepared, 32, False, 'actor')
        lp, entropy = likelihood(raw)
        delta, base_delta = lp - native_lp, lp - old_lp
        ratio = delta.exp()
        loss = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).mean() - .001 * entropy.mean()
        return loss, (ratio - 1 - delta).mean(), (base_delta.exp() - 1 - base_delta).mean(), raw

    def objective():
        loss, full, base, _ = metrics()
        return loss, torch.maximum(full, 2 * base)

    with torch.no_grad():
        initial = metrics()
        assert float((initial[3] - old_raw).abs().max()) < 5e-5
        initial_loss = float(initial[0])
    trials = []
    for _ in range(30):
        loss, _ = objective()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert bool(torch.isfinite(nn.utils.clip_grad_norm_(parameters, .5)))
        trial = guarded_actor_step(parameters, optimizer, objective, loss.detach(), 3e-5, .005,
            group_lrs=protocol['group_lrs'])
        trials.append(trial)
        if not trial['accepted']:
            break
    with torch.no_grad():
        loss, full, base, raw = metrics()
        means = selected_means(raw, target, mode, x, source_z[:, :2], actor.sampled_motion_aim_branch, vertical, actor.motor_branch)
        categorical = dict(target=(targets(old_raw[:, :45], x), targets(raw[:, :45], x)),
            pose=(torch.distributions.Categorical(logits=old_raw[:, 5:8]), torch.distributions.Categorical(logits=raw[:, 5:8])),
            weapon=(weapons(old_raw[:, 8:20], feature_mask(x[:, :845])), weapons(raw[:, 8:20], feature_mask(x[:, :845]))),
            fire=(old_fire, torch.distributions.Bernoulli(logits=fire_logits(actor.conditional_fire_branch, x, raw, target, mode, final, vertical, weapon))))
        factors = {k: float(torch.distributions.kl_divergence(*v).mean()) for k, v in categorical.items()}
        assert torch.equal(means, old_means), 'All conditional Gaussian means must remain exact'
    scope.materialize()
    scope.verify_frozen()
    assert equal_tree(actor.motor_branch.state_dict(), frozen_motor)
    assert equal_tree(scope.optimizer.state_dict(), untouched_optimizer) and equal_tree(legacy.state_dict(), scope.legacy_before)
    assert equal_tree(cp, snapshot)
    selected_kl = sum(factors.values())
    control_kl = sum(control_factors[k] for k in factors)
    checks = dict(selected_kl_three_times_control=selected_kl >= 3 * control_kl,
        gaussian_means_exact=True, std_exact=torch.equal(std, scope.std_before), source_checkpoint_exact=True,
        source_optimizers_exact=True, guard_passed=float(full) <= .005 + 1e-6 and float(base) <= .0025 + 1e-6,
        accepted_actor_step=any(t['accepted'] for t in trials))
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and Path(path).resolve().parent == Path(__file__).resolve().parent and Path(path).suffix == '.py':
            pin(path)
    assert all(sha(path) == digest for path, digest in pins.items())
    args.out.mkdir(parents=True)
    save(args.out / 'report.json', dict(state='complete', version=protocol['version'], device='cuda',
        checks=checks, mechanical_gate_passed=all(checks.values()), rows=len(rows),
        actor_steps=sum(t['accepted'] for t in trials), trials=trials,
        initial_loss=initial_loss, final_loss=float(loss), full_sample_k3=float(full), base_sample_k3=float(base),
        conditional_factor_kl=factors, control_conditional_factor_kl=control_factors,
        selected_kl_ratio_to_control=selected_kl / control_kl, gaussian_mean_max_delta=float((means - old_means).abs().max()),
        source_lp_native_max_error=replay_error, trainable_parameters=sum(p.numel() for p in parameters), inputs=pins,
        source_weights_sha256=source_seal['weights_sha256'], control_weights_sha256=control_seal['weights_sha256'],
        training_performed=True, disposable_fit=True, new_games=False, weights_written=False, checkpoint_written=False,
        promotion=False, scope=protocol['scope'] + ' Conditional KL values are not an additive decomposition of sampled joint K3.'))
    save(args.out / 'complete.json', dict(state='complete', report_sha256=sha(args.out / 'report.json')))
    print(dict(mechanical_gate_passed=all(checks.values()), steps=sum(t['accepted'] for t in trials),
        factor_kl=factors, selected_kl_ratio_to_control=selected_kl / control_kl, gaussian_max_delta=0.0))


if __name__ == '__main__':
    main()

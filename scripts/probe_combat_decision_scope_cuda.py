"""Disposable TRAIN fit of decision rows/residuals while learned aiming stays frozen."""
import argparse
import copy
import json
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha
from ppo_combat import torch, nn, training_devices, advantages, guarded_actor_step
from ppo_recurrent import prepare_context, sequence
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model, events, probabilities
from combat_geometry_ppo import GeometryJoint
from combat_precision_head import selected_means, targets, mode_distribution
from combat_weapon_head import distribution as weapons, feature_mask
from combat_conditional_fire import logits as fire_logits
from combat_weapon_state_residual import optimizer as actor_optimizer
from fork_conditional_weapon_exploration_cuda import equal_tree


class DecisionRows(nn.Module):
    def __init__(self, original):
        super().__init__()
        assert original.out_features == 81
        self.original = original
        self.weight = nn.Parameter(original.weight[4:29].detach().clone())
        self.bias = nn.Parameter(original.bias[4:29].detach().clone())

    def forward(self, inputs):
        result = self.original(inputs).clone()
        result[..., 4:29] = nn.functional.linear(inputs, self.weight, self.bias)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--update', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
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
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest == expected
        pins[str(path)] = digest
        return path

    pin(__file__)
    seal = read(pin(args.update / 'complete.json'))
    full = read(pin(args.update / 'report.json', seal['report_sha256']))
    assert full['version'] == 'combat_blocked_ground_ppo_v1' and full['branch'] == 'cost' and full['round'] == 2
    source = Path(full['source'])
    parent_seal = read(pin(source / 'complete.json'))
    model = read(pin(source / 'weights.json', parent_seal['weights_sha256']))
    pin(source / 'checkpoint.pt', parent_seal['checkpoint_sha256'])
    cp = torch.load(source / 'checkpoint.pt', map_location='cuda', weights_only=True)
    snapshot = copy.deepcopy(cp)
    corpus = args.update.parent / 'corpus'
    closed = read(pin(corpus / 'complete.json'))
    protocol = read(pin(corpus / 'protocol.json', closed['protocol_sha256']))
    assert protocol['train_only'] and protocol['assignments'][0]['split'] == 'train'
    data = corpus / 'processing-train/merged'
    meta = read(pin(data / 'report.json'))
    assert meta['model_sha256'] == parent_seal['weights_sha256'] == full['behavior_sha256']
    assert meta['rollout_sha256'] == full['rollout_sha256'] and meta['rollout_sha256'] not in cp['consumed_rollouts']
    with pin(data / 'rollout.jsonl', meta['rollout_sha256']).open(encoding='utf-8-sig') as stream:
        rows = list(map(json.loads, stream))
    with pin(data / 'sequence.jsonl', meta['sequence_sha256']).open(encoding='utf-8-sig') as stream:
        context = list(map(json.loads, stream))
    prepared = prepare_context(context, rows, 'cuda')
    x = torch.tensor([row['features'] for row in rows], device='cuda', dtype=torch.float32)
    category = lambda key: torch.tensor([row['sample'].get(key, 0) for row in rows], device='cuda', dtype=torch.long)
    target, mode, vertical, weapon = (category(key) for key in ('target', 'aim_mode', 'vertical', 'weapon'))
    final = torch.tensor([row['sample']['latent'] for row in rows], device='cuda', dtype=torch.float32)
    attack = torch.tensor([float(row['sample']['attack']) for row in rows], device='cuda')
    source_z, choice, timing, active = events(rows, x, target, mode, vertical, weapon)
    actor, joint = branch(base_model(model, allow_shared=True), 'actor'), GeometryJoint(model)
    std = nn.Parameter(torch.tensor(model['log_std'], device='cuda'))
    rates = [3e-5, .0003, .0003, .0003]
    original_opt = actor_optimizer(actor, std, *rates)
    original_opt.load_state_dict(copy.deepcopy(cp['actor_optimizer']))
    assert equal_tree(actor.state_dict(), cp['actor'])

    def evaluate(raw):
        lp, entropy, _, current_active = probabilities(joint, raw, std, source_z, final, attack, vertical, weapon,
            x, target, mode, choice, timing, actor.sampled_motion_aim_branch, actor.motor_branch, actor.conditional_fire_branch)
        assert torch.equal(current_active, active)
        return lp, entropy

    with torch.no_grad():
        old_raw, memory_error = sequence(actor, prepared, 32, True, 'actor')
        old_lp, _ = evaluate(old_raw)
        old_means = selected_means(old_raw, target, mode, x, source_z[:, :2], actor.sampled_motion_aim_branch, vertical, actor.motor_branch)
        assert memory_error < 2e-5
        native = torch.tensor([row['sample']['log_probability'] for row in rows], device='cuda')
        assert float((old_lp-native).abs().max()) < .003
        old_fire = torch.distributions.Bernoulli(logits=fire_logits(actor.conditional_fire_branch, x, old_raw, target, mode, final, vertical, weapon))
    actor.requires_grad_(False)
    joint.requires_grad_(False)
    std.requires_grad_(False)
    frozen = {parameter: parameter.detach().clone() for parameter in actor.parameters()}
    groups, state_copies = [], {}
    # Separate row Parameters/Adam clocks avoid modifying frozen coordinates or
    # their legacy Adam state. This adapter is not exported as a resumable model.
    for name in ('head', 'residual'):
        original = getattr(actor.base, name)
        wrapped = DecisionRows(original)
        setattr(actor.base, name, wrapped)
        parameters = [wrapped.weight, wrapped.bias]
        groups.append(dict(params=parameters, lr=rates[0], name='decision_rows_' + name))
        for before, after in zip((original.weight, original.bias), parameters):
            state_copies[after] = {key:(value[4:29].clone() if isinstance(value, torch.Tensor) and value.shape == before.shape else copy.deepcopy(value))
                                  for key,value in original_opt.state.get(before, {}).items()}
    selected = []
    for name, rate in (('weapon_state_branch', rates[1]), ('motor_branch', rates[2]), ('conditional_fire_branch', rates[3])):
        module = getattr(actor, name)
        assert module is not None
        module.requires_grad_(True)
        parameters = list(module.parameters())
        selected.extend(parameters)
        groups.append(dict(params=parameters, lr=rate, name=name))
        for parameter in parameters:
            state_copies[parameter] = copy.deepcopy(original_opt.state.get(parameter, {}))
    for parameter in selected:
        frozen.pop(parameter)
    optimizer = torch.optim.Adam(groups)
    for parameter, state in state_copies.items():
        if state:
            optimizer.state[parameter] = state
    parameters = [parameter for group in optimizer.param_groups for parameter in group['params']]
    adv, _ = advantages(rows, cp['config']['gamma'], cp['config']['lambda'])
    advantage = torch.tensor(adv, device='cuda')
    advantage = (advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)

    def metrics():
        raw, _ = sequence(actor, prepared, 32, False, 'actor')
        lp, entropy = evaluate(raw)
        delta = lp-old_lp
        ratio = delta.exp()
        loss = -torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage).mean() - .001*entropy.mean()
        return loss, (ratio-1-delta).mean(), raw

    def objective():
        loss, kl, _ = metrics()
        # Same inherited base-factor guard as full: base KL <= .0025;
        # geometry is frozen and remains outside the trainable parameter set.
        return loss, 2*kl

    with torch.no_grad():
        initial_loss, initial_kl, initial_raw = metrics()
        assert float((initial_raw-old_raw).abs().max()) < 5e-5 and float(initial_kl) < 1e-6
        initial_loss = float(initial_loss)
    trials = []
    for _ in range(30):
        loss, _ = objective()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(parameters, .5)
        assert bool(torch.isfinite(norm))
        trial = guarded_actor_step(parameters, optimizer, objective, loss.detach(), rates[0], .005,
                                   group_lrs=[group['lr'] for group in groups])
        trials.append(trial)
        if not trial['accepted']:
            break
    with torch.no_grad():
        final_loss, final_kl, raw = metrics()
        means = selected_means(raw, target, mode, x, source_z[:, :2], actor.sampled_motion_aim_branch, vertical, actor.motor_branch)
        normal_kl = torch.distributions.kl_divergence(torch.distributions.Normal(old_means, std.exp()), torch.distributions.Normal(means, std.exp())).mean(0)
        categorical = dict(target=(targets(old_raw[:, :45],x), targets(raw[:, :45],x)),
            pose=(torch.distributions.Categorical(logits=old_raw[:,5:8]), torch.distributions.Categorical(logits=raw[:,5:8])),
            weapon=(weapons(old_raw[:,8:20],feature_mask(x[:,:845])),weapons(raw[:,8:20],feature_mask(x[:,:845]))),
            fire=(old_fire,torch.distributions.Bernoulli(logits=fire_logits(actor.conditional_fire_branch,x,raw,target,mode,final,vertical,weapon))))
        factors = {name:float(torch.distributions.kl_divergence(*pair).mean()) for name,pair in categorical.items()}
        factors.update(movement=float(normal_kl[:2].sum()), conditional_aim=float(normal_kl[2:].sum()))
        aim_delta = float((means[:,2:]-old_means[:,2:]).abs().max())
        assert all(torch.equal(parameter,value) for parameter,value in frozen.items())
        assert equal_tree(cp,snapshot)
        assert factors['conditional_aim'] < 1e-6 and aim_delta < 1e-5
    assert all(sha(path) == digest for path,digest in pins.items())
    args.out.mkdir(parents=True)
    save(args.out / 'report.json', dict(state='complete', version='combat_decision_scope_probe_v1', device='cuda',
        rows=len(rows), actor_steps=sum(trial['accepted'] for trial in trials), trials=trials,
        initial_loss=initial_loss, final_loss=float(final_loss), full_sample_k3=float(final_kl),
        conditional_factor_kl=factors, max_conditional_aim_mean_delta=aim_delta,
        frozen_original_weights_exact=True, source_checkpoint_exact=True,
        trainable_parameters=sum(parameter.numel() for parameter in parameters),
        source_weights_sha256=parent_seal['weights_sha256'], full_control_weights_sha256=seal['weights_sha256'],
        inputs=pins, training_performed=True, disposable_fit=True, new_games=False, weights_written=False,
        scope='One disposable actor-only fit on its own source-policy TRAIN data, same LR/cap/base KL guard. '
              'Conditional aiming frozen; changing target selection may change actual gaze. No critic fit, serialized optimizer, '
              'new inference artifact, quality result, rollout consumption or approved resume migration.'))
    save(args.out / 'complete.json',dict(state='complete',report_sha256=sha(args.out/'report.json')))
    print(dict(steps=sum(trial['accepted'] for trial in trials), factors=factors, loss=float(final_loss), kl=float(final_kl)))


if __name__ == '__main__':
    main()

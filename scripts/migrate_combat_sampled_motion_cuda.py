"""Zero-effect conditional-motion migration and autoregressive likelihood audit."""
import argparse
import copy
import json
import pathlib

from ppo_recurrent import torch, nn, prepare_context, sequence, durable_json, durable_write
from migrate_combat_navigation_cuda import branch
from combat_sampled_motion_aim import initialize, correction, VERSION
from combat_precision_head import selected_means, probabilities
from combat_weapon_state_residual import optimizer
from fork_conditional_weapon_exploration_cuda import equal_tree
from process_combat_architecture_pool import read, sha
from combat_sequence_retention import policy_kl


def main():
    ap = argparse.ArgumentParser()
    for name in ('source', 'data', 'out'):
        ap.add_argument('--'+name, type=pathlib.Path, required=True)
    args = ap.parse_args()
    assert torch.cuda.is_available() and not args.out.exists()
    torch.set_default_device('cuda'); torch.set_num_threads(2); torch.manual_seed(20261022)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    seal = read(args.source/'complete.json'); pins = {}
    for name, key in [('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'), ('report.json', 'report_sha256')]:
        path = args.source/name; pins[path] = sha(path); assert pins[path] == seal[key]
    assert read(args.source/'report.json')['device'] == 'cuda'
    model = read(args.source/'weights.json'); updated = initialize(model)
    cp = torch.load(args.source/'checkpoint.pt', map_location='cuda', weights_only=True)
    assert cp['weights_sha256'] == seal['weights_sha256']
    before, after = branch(model, 'actor'), branch(updated, 'actor')
    assert equal_tree(before.state_dict(), cp['actor'])
    assert equal_tree(branch(model, 'value').state_dict(), cp['value'])
    assert equal_tree(branch(updated, 'value').state_dict(), cp['value'])
    for key, tensor in cp['actor'].items(): assert torch.equal(tensor, after.state_dict()[key])
    meta = read(args.data/'report.json'); pins[args.data/'report.json'] = sha(args.data/'report.json')
    assert meta['model_sha256'] == seal['weights_sha256']
    streams = {}
    for name in ('rollout', 'sequence'):
        path = args.data/(name+'.jsonl'); pins[path] = sha(path); assert pins[path] == meta[name+'_sha256']
        streams[name] = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()]
    rows = streams['rollout']; prepared = prepare_context(streams['sequence'], rows, 'cuda')
    x = torch.tensor([r['features'] for r in rows], dtype=torch.float32, device='cuda')
    def samples(name, dtype):
        return torch.tensor([r['sample'].get(name, 0) for r in rows], dtype=dtype, device='cuda')
    z = samples('latent', torch.float32); attack = samples('attack', torch.float32)
    vertical, weapon, target, mode = [samples(k, torch.long) for k in ('vertical', 'weapon', 'target', 'aim_mode')]
    std = nn.Parameter(cp['log_std'].clone()); new_std = nn.Parameter(cp['log_std'].clone())
    assert torch.equal(std, torch.tensor(model['log_std'], dtype=torch.float32, device='cuda'))
    with torch.no_grad():
        old, error = sequence(before, prepared, check_states=True)
        new, new_error = sequence(after, prepared, check_states=True)
        torch.testing.assert_close(new, old, rtol=0, atol=1e-5)
        old_means = selected_means(old, target, mode)
        new_means = selected_means(new, target, mode, x, z[:, :2], after.sampled_motion_aim_branch)
        torch.testing.assert_close(new_means, old_means, rtol=0, atol=1e-5)
        old_lp, _ = probabilities(old, std, z, attack, vertical, weapon, x, target, mode)
        new_lp, _ = probabilities(new, new_std, z, attack, vertical, weapon, x, target, mode, after.sampled_motion_aim_branch)
        torch.testing.assert_close(new_lp, old_lp, rtol=0, atol=1e-5)
        native_error = float((new_lp-samples('log_probability', torch.float32)).abs().max())
        assert max(error, new_error) < 2e-5 and native_error < .003

    # A nonzero path through the sampled-forward input must alter conditional aim
    # and its likelihood, while target=none remains exactly neutral.
    probe = copy.deepcopy(after.sampled_motion_aim_branch)
    with torch.no_grad():
        for p in probe.parameters(): p.zero_()
        probe[0].weight[0, 119] = 1
        probe[2].weight[0, 0] = 1
        probe[4].weight[0, 0] = .25
    movement = torch.full((len(rows), 2), .4, dtype=torch.float32, device='cuda', requires_grad=True)
    residual = correction(probe, x, new.detach(), movement, target, mode)
    expected = torch.where(target > 0, movement[:, 0].tanh()*.25, torch.zeros_like(movement[:, 0]))
    torch.testing.assert_close(residual[:, 0], expected, rtol=1e-5, atol=1e-6)
    assert torch.equal(residual[:, 1], torch.zeros_like(residual[:, 1])) and bool((target > 0).any())
    gradient, = torch.autograd.grad(residual.sum(), movement)
    assert bool((gradient[target > 0, 0] > 0).all()) and bool(torch.isfinite(gradient).all())
    none = torch.zeros_like(target)
    assert torch.equal(correction(probe, x, new, movement.detach(), none, mode), torch.zeros_like(residual))
    with torch.no_grad():
        probe_z = z.clone(); probe_z[:, :2] = movement.detach()
        delta = correction(probe, x, new, probe_z[:, :2], target, mode)
        base_means = selected_means(new, target, mode)
        independent, _ = probabilities(new, new_std, probe_z, attack, vertical, weapon, x, target, mode)
        conditional, _ = probabilities(new, new_std, probe_z, attack, vertical, weapon, x, target, mode, probe)
        before_normal = (probe_z[:, 2:4]-base_means[:, 2:4])/new_std[2:].exp()
        after_normal = (probe_z[:, 2:4]-base_means[:, 2:4]-delta)/new_std[2:].exp()
        expected_change = -.5*(after_normal.square()-before_normal.square()).sum(-1)
        torch.testing.assert_close(conditional-independent, expected_change, rtol=1e-4, atol=.003)
        likelihood_change = float((conditional-independent).abs().max())
        assert likelihood_change > .1

    # Conditional retention must see the branch, including at equal raw logits.
    same=policy_kl(new,new_std,new.detach(),new_std.detach(),x,
        after.sampled_motion_aim_branch,after.sampled_motion_aim_branch,z[:,:2].detach())
    assert float(same.abs().max()) < 1e-8
    changed=policy_kl(new,new_std,new.detach(),new_std.detach(),x,
        probe,after.sampled_motion_aim_branch,movement.detach())
    assert bool(torch.isfinite(changed).all()) and bool((changed>=-1e-8).all()) and float(changed.max()) > .01
    changed.mean().backward()
    assert any(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.norm())>0 for p in probe.parameters())

    old_opt = optimizer(before, std, cp['config']['actor_lr'], cp['actor_training']['weapon_state_residual']['learning_rate'])
    new_opt = optimizer(after, new_std, cp['config']['actor_lr'], cp['actor_training']['weapon_state_residual']['learning_rate'])
    old_opt.load_state_dict(copy.deepcopy(cp['actor_optimizer']))
    old_named = dict(before.named_parameters()); old_named['@log_std'] = std
    new_named = dict(after.named_parameters()); new_named['@log_std'] = new_std
    for name, old_parameter in old_named.items():
        new_opt.state[new_named[name]] = copy.deepcopy(old_opt.state.get(old_parameter, {}))
        assert equal_tree(new_opt.state[new_named[name]], old_opt.state.get(old_parameter, {}))
    for new_group, old_group in zip(new_opt.param_groups, old_opt.param_groups):
        for key, value in old_group.items():
            if key != 'params': new_group[key] = copy.deepcopy(value)
    assert all(not new_opt.state.get(p) for n, p in new_named.items() if n.startswith('sampled_motion_aim_branch.'))
    args.out.mkdir(parents=True)
    durable_json(args.out/'weights.json', updated)
    receipt = dict(state='complete', device='cuda', version=VERSION, gpu=torch.cuda.get_device_name(0),
        source_weights_sha256=seal['weights_sha256'], source_checkpoint_sha256=seal['checkpoint_sha256'],
        zero_initialized_output=True, history_rows=len(rows), sequence_memory_max_error=max(error, new_error),
        native_log_probability_max_error=native_error, nonzero_probe_likelihood_change=likelihood_change,
        nonzero_sampled_movement_gradient=True, none_target_neutral=True, old_optimizer_states_preserved=True,
        new_parameters=sum(p.numel() for p in after.sampled_motion_aim_branch.parameters()),
        training_performed=False, promotion=False, conditional_retention_zero_identity_and_positive_perturbation=True,
        inputs=[dict(path=str(p), sha256=h) for p, h in pins.items()], script_sha256=sha(__file__),
        scope='Zero conditional residual preserves old policy on pinned histories. Nonzero CUDA path tests sampled command sensitivity, joint likelihood and conditional retention, not native combat quality. Retention uses one fixed behavior movement MC sample/history and exact target/mode summation.')
    fork = copy.deepcopy(cp)
    fork.update(actor=after.state_dict(), weights_sha256=sha(args.out/'weights.json'), actor_optimizer=new_opt.state_dict(), sequence_retention=None)
    fork['model_migrations'] = cp.get('model_migrations', [])+[copy.deepcopy(receipt)]
    durable_write(args.out/'checkpoint.pt', lambda f: torch.save(fork, f))
    assert equal_tree(torch.load(args.out/'checkpoint.pt', map_location='cuda', weights_only=True), fork)
    assert equal_tree(branch(updated, 'actor').state_dict(), fork['actor'])
    for path, digest in pins.items(): assert sha(path) == digest
    durable_json(args.out/'report.json', receipt)
    durable_json(args.out/'complete.json', dict(version='combat_update_complete_v1',
        weights_sha256=sha(args.out/'weights.json'), checkpoint_sha256=sha(args.out/'checkpoint.pt'), report_sha256=sha(args.out/'report.json')))
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('inputs', 'scope')}))


if __name__ == '__main__':
    main()

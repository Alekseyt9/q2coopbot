"""Small supervised geometric-label pilot; fixed budget, CUDA only, no promotion."""
import argparse
import copy
import gzip
import json
import pathlib

from ppo_recurrent import torch, nn, prepare_context, sequence, durable_json, durable_write
from migrate_combat_navigation_cuda import branch
from combat_spatial_aim import inputs, export
from combat_precision_head import selected_means
from combat_weapon_state_residual import optimizer
from fork_conditional_weapon_exploration_cuda import equal_tree
from process_combat_architecture_pool import read, sha


def main():
    ap = argparse.ArgumentParser()
    for name in ('source', 'labels', 'out'):
        ap.add_argument('--'+name, type=pathlib.Path, required=True)
    ap.add_argument('--steps', type=int, default=80)
    ap.add_argument('--learning-rate', type=float, default=.0003)
    args = ap.parse_args()
    assert torch.cuda.is_available(), 'CUDA required; CPU fallback forbidden'
    assert 1 <= args.steps <= 200 and 0 < args.learning_rate <= .001 and not args.out.exists()
    torch.set_default_device('cuda'); torch.set_num_threads(2); torch.manual_seed(20261021)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    pins = {}

    def pin(path, expected=None):
        path = pathlib.Path(path); digest = sha(path)
        assert expected is None or digest == expected
        if str(path) in pins: assert pins[str(path)] == digest
        pins[str(path)] = digest
        return path

    seal = read(pin(args.source/'complete.json'))
    for name, key in [('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'),
                      ('report.json', 'report_sha256')]:
        pin(args.source/name, seal[key])
    assert read(args.source/'report.json')['device'] == 'cuda'
    model = read(args.source/'weights.json')
    assert not model.get('sampled_motion_aim'), 'This older output-layer pilot does not fit conditional motion policies'
    cp = torch.load(args.source/'checkpoint.pt', map_location='cuda', weights_only=True)
    assert cp['weights_sha256'] == seal['weights_sha256']
    actor = branch(model, 'actor'); actor.eval()
    assert equal_tree(actor.state_dict(), cp['actor'])
    assert equal_tree(branch(model, 'value').state_dict(), cp['value'])
    assert all(p.is_cuda for p in actor.parameters())
    meta = read(pin(args.labels/'report.json'))
    assert meta['state'] == 'complete' and not meta['training_performed']
    for binding in meta['inputs']: pin(binding['path'], binding['sha256'])
    corpus = pathlib.Path(meta['corpus'])
    assert read(corpus/'protocol.json')['source'] == str(args.source.resolve())
    for parameter in actor.parameters(): parameter.requires_grad_(False)
    last = actor.branch[-1]
    assert isinstance(last, nn.Linear) and last.weight.shape == (6, 32)
    datasets = {}
    seeds = set()
    with torch.no_grad():
        for split in ('train', 'development'):
            binding = meta['splits'][split]
            path = pin(binding['output'], binding['sha256'])
            with gzip.open(path, 'rt', encoding='utf-8') as stream: labels = list(map(json.loads, stream))
            info = read(corpus/('processing-'+split)/'merged/report.json')
            paths = {name: pin(corpus/('processing-'+split)/'merged'/(name+'.jsonl'), info[name+'_sha256'])
                     for name in ('rollout', 'sequence')}
            rows = [json.loads(line) for line in paths['rollout'].read_text(encoding='utf-8-sig').splitlines()]
            context = [json.loads(line) for line in paths['sequence'].read_text(encoding='utf-8-sig').splitlines()]
            assert info['model_sha256'] == seal['weights_sha256']
            current_seeds = {r['seed'] for r in rows}
            assert not seeds.intersection(current_seeds); seeds.update(current_seeds)
            prepared = prepare_context(context, rows, 'cuda')
            raw, error = sequence(actor, prepared, check_states=True)
            assert error < 2e-5
            base, _ = sequence(actor.base, prepared)
            features = torch.tensor([r['features'] for r in rows], dtype=torch.float32, device='cuda')
            hidden = actor.branch[:-1](inputs(features, base))
            positions = {(r['seed'], r['index']): i for i, r in enumerate(rows)}
            selected = []; excluded = 0
            for item in labels:
                i = positions[(item['seed'], item['index'])]; row = rows[i]
                assert item['split'] == split and item['sample'] == row['sample'] and item['features'] == row['features']
                target = item['sample']['target']; mode = item['sample'].get('aim_mode', 0)
                assert 1 <= target <= 8 and mode in (0, 1)
                # Frozen mode head: do not teach an unreachable fine command.
                desired = item['offline_label']['desired_command_delta_degrees'][::-1]
                scale = 15. if mode else 180.
                if max(map(abs, desired)) >= scale*.99 or item['offline_label']['muzzle_blocked'] or item['offline_label']['water']:
                    excluded += 1; continue
                selected.append((i, target, mode, desired, scale))
            assert selected
            ix = torch.tensor([s[0] for s in selected], dtype=torch.long, device='cuda')
            targets = torch.tensor([s[1] for s in selected], dtype=torch.long, device='cuda')
            modes = torch.tensor([s[2] for s in selected], dtype=torch.long, device='cuda')
            h = hidden[ix, targets-1].detach()
            output_channels = torch.where(modes[:, None] == 1,
                torch.tensor([0, 1], device='cuda'), torch.tensor([4, 5], device='cuda'))
            baseline = selected_means(raw[ix], targets, modes)[:, 2:4].detach()
            old_last = last(h).gather(1, output_channels).detach()
            datasets[split] = dict(h=h, channels=output_channels, baseline=baseline, old=old_last,
                desired=torch.tensor([s[3] for s in selected], dtype=torch.float32, device='cuda'),
                scale=torch.tensor([s[4] for s in selected], dtype=torch.float32, device='cuda')[:, None],
                prepared=prepared, raw=raw, excluded=excluded, rows=len(rows), samples=len(selected))

    def prediction(data):
        change = last(data['h']).gather(1, data['channels'])-data['old']
        return (data['baseline']+change).tanh()*data['scale'], change

    def metrics(data):
        predicted, change = prediction(data)
        error = (predicted-data['desired']).abs()
        return dict(samples=data['samples'], excluded=data['excluded'], mean_absolute_axis_error_degrees=float(error.mean()),
            mean_joint_error_degrees=float((error.square().sum(-1)).sqrt().mean()),
            residual_latent_rms=float(change.square().mean().sqrt()))

    last.weight.requires_grad_(True); last.bias.requires_grad_(True)
    opt = torch.optim.Adam(last.parameters(), lr=args.learning_rate)
    trajectory = []
    with torch.no_grad(): trajectory.append(dict(step=0, **{s: metrics(d) for s, d in datasets.items()}))
    initial = copy.deepcopy(last.state_dict())
    for step in range(1, args.steps+1):
        opt.zero_grad(set_to_none=True)
        predicted, change = prediction(datasets['train'])
        loss = nn.functional.smooth_l1_loss(predicted, datasets['train']['desired'], beta=2.) + .1*change.square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        # Mode rows2/3 and hidden features remain exactly frozen.
        for p in last.parameters(): p.grad[2:4].zero_()
        torch.nn.utils.clip_grad_norm_(last.parameters(), 5.)
        opt.step()
        if step in (10, 30, args.steps):
            with torch.no_grad(): trajectory.append(dict(step=step, **{s: metrics(d) for s, d in datasets.items()}))
    assert torch.equal(last.weight[2:4], initial['weight'][2:4]) and torch.equal(last.bias[2:4], initial['bias'][2:4])
    updated = copy.deepcopy(model); export(updated, actor)
    # Runtime contract and every model field outside the spatial output layer stay identical.
    old_view, new_view = copy.deepcopy(model), copy.deepcopy(updated)
    old_view['spatial_aim']['layers'][-1] = new_view['spatial_aim']['layers'][-1]
    assert old_view == new_view
    restored_actor = branch(updated, 'actor')
    assert equal_tree(actor.state_dict(), restored_actor.state_dict())
    checks = {}
    with torch.no_grad():
        for split, data in datasets.items():
            after, _ = sequence(restored_actor, data['prepared'])
            before = data['raw']
            keep = list(range(29))+list(range(45, 47))+list(range(63, 81))
            torch.testing.assert_close(after[:, keep], before[:, keep], rtol=0, atol=1e-5)
            checks[split] = float((after[:, keep]-before[:, keep]).abs().max())
    # Retain PPO optimizer structure; invalidate moments only for edited coordinates.
    std = nn.Parameter(cp['log_std'].clone())
    ppo_opt = optimizer(actor, std, cp['config']['actor_lr'], cp['actor_training']['weapon_state_residual']['learning_rate'])
    ppo_opt.load_state_dict(copy.deepcopy(cp['actor_optimizer']))
    for parameter in last.parameters():
        for key, value in ppo_opt.state.get(parameter, {}).items():
            if key != 'step': value[[0, 1, 4, 5]] = 0
    args.out.mkdir(parents=True)
    durable_json(args.out/'weights.json', updated)
    fork = copy.deepcopy(cp)
    fork.update(actor=actor.state_dict(), weights_sha256=sha(args.out/'weights.json'), actor_optimizer=ppo_opt.state_dict(),
                sequence_retention=None, aim_feedback_bc_optimizer=opt.state_dict())
    receipt = dict(version='native_bodypoint_spatial_output_bc_v1', device='cuda', state='complete',
        source_weights_sha256=seal['weights_sha256'], source_checkpoint_sha256=seal['checkpoint_sha256'],
        labels_report_sha256=sha(args.labels/'report.json'), steps=args.steps, learning_rate=args.learning_rate,
        fixed_budget=True, selection_on_development=False, trajectory=trajectory, untouched_logits_max_error=checks,
        trainable_parameters=4*33, training_performed=True, promotion=False,
        trainer_source_sha256=sha(__file__), gpu=torch.cuda.get_device_name(0),
        fresh_own_policy_required_before_ppo=True,
        scope='Geometric body-point/recoil label pilot, only spatial output yaw/pitch rows. Frozen temporal encoder, modes, decisions, movement and sampling noise. Original client inputs. No teacher-optimality claim or native combat quality result.')
    fork['model_migrations'] = cp.get('model_migrations', [])+[copy.deepcopy(receipt)]
    durable_write(args.out/'checkpoint.pt', lambda f: torch.save(fork, f))
    assert equal_tree(torch.load(args.out/'checkpoint.pt', map_location='cuda', weights_only=True), fork)
    for path, digest in pins.items(): assert sha(path) == digest
    receipt['inputs'] = [dict(path=p, sha256=h) for p, h in pins.items()]
    durable_json(args.out/'report.json', receipt)
    durable_json(args.out/'complete.json', dict(version='combat_update_complete_v1',
        weights_sha256=sha(args.out/'weights.json'), checkpoint_sha256=sha(args.out/'checkpoint.pt'), report_sha256=sha(args.out/'report.json')))
    print(json.dumps(dict(trajectory=trajectory, untouched_logits_max_error=checks)))


if __name__ == '__main__':
    main()

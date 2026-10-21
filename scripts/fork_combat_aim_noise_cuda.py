"""Reduce only yaw/pitch Gaussian noise in a sealed CUDA checkpoint."""
import argparse
import copy
import json
import math
import pathlib

from ppo_recurrent import torch, nn, read, durable_json, durable_write, prepare_context, sequence
from migrate_combat_navigation_cuda import branch
from combat_weapon_state_residual import optimizer
from fork_conditional_weapon_exploration_cuda import equal_tree
from process_combat_architecture_pool import sha


def main():
    ap = argparse.ArgumentParser()
    for name in ('source', 'data', 'out'):
        ap.add_argument('--'+name, type=pathlib.Path, required=True)
    ap.add_argument('--scale', type=float, required=True)
    args = ap.parse_args()
    assert torch.cuda.is_available() and 0 < args.scale < 1 and not args.out.exists()
    torch.set_default_device('cuda')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    seal = read(args.source/'complete.json')
    pins = {}
    for name, key in [('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'),
                      ('report.json', 'report_sha256')]:
        p = args.source/name
        assert sha(p) == seal[key]
        pins[p] = sha(p)
    assert read(args.source/'report.json')['device'] == 'cuda'
    model = read(args.source/'weights.json')
    assert not model['deterministic'] and model.get('weapon_state_residual')
    cp = torch.load(args.source/'checkpoint.pt', map_location='cuda', weights_only=True)
    assert cp['weights_sha256'] == seal['weights_sha256']
    assert torch.equal(cp['log_std'], torch.tensor(model['log_std'], dtype=torch.float32, device='cuda'))
    new = copy.deepcopy(model)
    std = nn.Parameter(cp['log_std'].clone())
    with torch.no_grad():
        std[2:] += math.log(args.scale)
    assert bool((std >= -8).all())
    new['log_std'] = std.detach().cpu().tolist()
    before, after = branch(model, 'actor'), branch(new, 'actor')
    for role in ('actor', 'value'):
        assert equal_tree(branch(model, role).state_dict(), cp[role])
        assert equal_tree(branch(new, role).state_dict(), cp[role])
    assert new['log_std'][:2] == model['log_std'][:2]
    old_json, new_json = copy.deepcopy(model), copy.deepcopy(new)
    old_json.pop('log_std'); new_json.pop('log_std')
    assert old_json == new_json
    meta = read(args.data/'report.json')
    assert meta['model_sha256'] == seal['weights_sha256']
    pins[args.data/'report.json'] = sha(args.data/'report.json')
    for name in ('rollout', 'sequence'):
        p = args.data/(name+'.jsonl')
        assert sha(p) == meta[name+'_sha256']
        pins[p] = sha(p)
    with (args.data/'rollout.jsonl').open(encoding='utf-8-sig') as stream:
        rows = [json.loads(line) for line in stream]
    with (args.data/'sequence.jsonl').open(encoding='utf-8-sig') as stream:
        context = [json.loads(line) for line in stream]
    prepared = prepare_context(context, rows, 'cuda')
    with torch.no_grad():
        a, ea = sequence(before, prepared, check_states=True)
        b, eb = sequence(after, prepared, check_states=True)
        assert torch.equal(a, b) and max(ea, eb) < 2e-5
    ao = optimizer(after, std, cp['config']['actor_lr'],
                   cp['actor_training']['weapon_state_residual']['learning_rate'])
    ao.load_state_dict(copy.deepcopy(cp['actor_optimizer']))
    assert equal_tree(ao.state_dict(), cp['actor_optimizer'])
    # Only sampling-coordinate moments change; network tensors and clocks stay.
    moments = ao.state.get(std, {})
    assert moments
    for key, value in moments.items():
        if key != 'step':
            assert value.shape == (4,)
            value[2:].zero_()
    fork = copy.deepcopy(cp)
    previous = copy.deepcopy(fork.get('sequence_retention'))
    fork['sequence_retention'] = None
    fork['log_std'] = std.detach().clone()
    fork['actor_optimizer'] = ao.state_dict()
    args.out.mkdir(parents=True)
    durable_json(args.out/'weights.json', new)
    fork['weights_sha256'] = sha(args.out/'weights.json')
    receipt = dict(state='complete', device='cuda', version='aim_noise_fork_v1',
        scale=args.scale, before_log_std=model['log_std'], after_log_std=new['log_std'],
        source_weights_sha256=seal['weights_sha256'], source_checkpoint_sha256=seal['checkpoint_sha256'],
        real_history_rows=len(rows), data_report_sha256=pins[args.data/'report.json'],
        unchanged_all_mean_logits_exact=True, unchanged_actor_value_tensors_exact=True,
        state_max_error=max(ea, eb), previous_sequence_retention=previous,
        training_performed=False, persisted_optimizer_steps=0, promotion=False,
        scope='Only yaw/pitch latent Gaussian std scaled. All categorical distributions and forward/side std unchanged for fixed histories. Not a deterministic policy. Closed-loop states may differ; fresh own-policy data required before further fitting. Held-out histories used only for CUDA verification.')
    fork['model_migrations'] = cp.get('model_migrations', []) + [copy.deepcopy(receipt)]
    durable_write(args.out/'checkpoint.pt', lambda f: torch.save(fork, f))
    restored = torch.load(args.out/'checkpoint.pt', map_location='cuda', weights_only=True)
    assert equal_tree(restored, fork)
    for p, digest in pins.items():
        assert sha(p) == digest
    receipt.update(weights_sha256=fork['weights_sha256'], checkpoint_sha256=sha(args.out/'checkpoint.pt'))
    durable_json(args.out/'report.json', receipt)
    durable_json(args.out/'complete.json', dict(state='complete', weights_sha256=receipt['weights_sha256'],
        checkpoint_sha256=receipt['checkpoint_sha256'], report_sha256=sha(args.out/'report.json')))
    print(dict(device='cuda', scale=args.scale, rows=len(rows), training=False))


if __name__ == '__main__':
    main()

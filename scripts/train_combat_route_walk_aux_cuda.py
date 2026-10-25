"""Learn route F/S means from observed walk hints; camera and combat body frozen."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from ppo_combat import torch, nn, training_devices
from ppo_recurrent import prepare_context, sequence
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model
from combat_memory_route_movement import VERSION, inputs, export
from train_combat_memory_route_ppo import admit
from combat_checkpoint_scope import require_checkpoint_scope
from evaluate_registered_combat_models import read, save, sha

SCOPE = 'combat_observed_route_walk_aux_v1'
CHECKPOINT = 'combat_route_walk_aux_checkpoint_v1'
CONFIG = dict(version=SCOPE, seed=20261025, learning_rate=.001, planned_steps=300,
              maximum_speed=.5, one_step_distance=40., source_retention=.01,
              unlabelled_active_retention=.1, max_grad_norm=1.,
              actor_scope='full_route_movement_branch', checkpoint_selection='fixed_final_step')


def fixed_identity(model):
    result = copy.deepcopy(model)
    result.pop('memory_route_movement')
    return hashlib.sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_data(report, split, reference):
    entries = []
    for e in report['episodes']:
        if e['split'] != split:
            continue
        assert sha(e['pack']) == e['pack_sha256'] and sha(e['labels']) == e['labels_sha256']
        rows, history, token, _ = admit(Path(e['source_cuda']), e['source_model_sha256'], [])
        assert token == e['native_rollout_sha256']
        import numpy as np
        pack = np.load(e['pack'], allow_pickle=False)
        labels = read(e['labels'])
        assert len(labels) == len(pack['indices']) == e['rows']
        assert all(rows[int(i)]['frame'] == label['frame'] for i, label in zip(pack['indices'], labels))
        with torch.no_grad():
            context = prepare_context(history, rows, 'cuda')
            raw, error = sequence(reference, context, 32, True, 'actor')
            assert error < 2e-5
            features = torch.tensor([r['features'] for r in rows], device='cuda')
            x, active = inputs(features, raw)
            idx = torch.tensor(pack['indices'], device='cuda', dtype=torch.long)
            assert bool(active[idx].all())
            assert bool(features[idx, 2].eq(1).all())
            # No visible target means the existing coarse yaw distribution is
            # used; no sampled-motion fine-aim transform is assumed here.
            assert all(rows[int(i)]['sample'].get('target', 0) == 0 and rows[int(i)]['sample'].get('aim_mode', 0) == 0 for i in pack['indices'])
            turn = torch.tensor(pack['turn'], device='cuda', dtype=torch.float32)
            distance = torch.tensor([sum(v*v for v in label['waypoint_relative'][:2])**.5 for label in labels], device='cuda')
            speed = (distance/CONFIG['one_step_distance']).clamp_max(CONFIG['maximum_speed'])
            camera_delta = torch.pi*raw[idx, 2].tanh()
            relative = turn-camera_delta
            target = speed[:, None]*torch.stack((relative.cos(), -relative.sin()), -1)
            unlabelled = active.clone()
            unlabelled[idx] = False
        entries.append(dict(episode=e, rows=rows, context=context, features=features, raw=raw,
                            x=x.detach(), active=active, idx=idx, unlabelled=unlabelled,
                            target=target.detach(), latent_target=torch.atanh(target).detach()))
    assert entries
    return entries


def prediction(head, e):
    return e['raw'][:, :2] + head(e['x'])*e['active'][:, None]


def metrics(head, entries):
    proposed, targets, original, inactive = [], [], [], []
    for e in entries:
        proposed.append(prediction(head, e)[e['idx']].tanh())
        targets.append(e['target'])
        original.append(e['raw'][e['idx'], :2].tanh())
        if bool(e['unlabelled'].any()):
            inactive.append(head(e['x'])[e['unlabelled']].abs().mean())
    p, y, base = torch.cat(proposed), torch.cat(targets), torch.cat(original)
    alignment = (p*y).sum(-1)/(p.norm(dim=-1)*y.norm(dim=-1)).clamp_min(1e-8)
    return dict(rows=len(p), mean_command_mae=float((p-y).abs().mean()),
                source_command_mae=float((base-y).abs().mean()),
                mean_command_alignment=float(alignment.mean()),
                away_command_rows=int(alignment.lt(-.1).sum()),
                unlabelled_active_mean_correction=float(torch.stack(inactive).mean()) if inactive else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--protocol', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--resume', type=Path)
    a = ap.parse_args()
    assert not a.out.exists() and 1 <= a.steps <= CONFIG['planned_steps']
    protocol = read(a.protocol)
    assert protocol['config'] == CONFIG and protocol['state'] == 'planned'
    training_devices()
    torch.set_default_device('cuda')
    torch.set_num_threads(2)
    torch.manual_seed(CONFIG['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    model, model_hash = read(a.model), sha(a.model)
    parent_path = Path(protocol['source_model'])
    assert sha(parent_path) == protocol['source_sha256'] and sha(a.data) == protocol['data_sha256']
    parent = read(parent_path)
    assert fixed_identity(model) == fixed_identity(parent)
    assert model['memory_route_movement']['version'] == VERSION
    last = parent['memory_route_movement']['layers'][-1]
    assert not any(v for row in last['weight'] for v in row) and not any(last['bias'])
    report = read(a.data)
    assert report['version'] == 'combat_observed_static_route_heading_v1' and report['state'] == 'passed'
    assert report['ground_walk_only'] and not report['ppo_eligible'] and not report['executed_demonstration']
    for path, digest in report['inputs'].items():
        assert sha(path) == digest
    train_worlds = {(e['map'], e['seed']) for e in report['episodes'] if e['split'] == 'train'}
    diagnostic_worlds = {(e['map'], e['seed']) for e in report['episodes'] if e['split'] == 'diagnostic'}
    assert train_worlds and diagnostic_worlds and not train_worlds & diagnostic_worlds
    assert all(e['source_model_sha256'] == sha(parent_path) for e in report['episodes'])
    actor = branch(base_model(model, allow_shared=True), 'actor').eval().requires_grad_(False)
    reference = branch(base_model(parent, allow_shared=True), 'actor').eval().requires_grad_(False)
    actor.memory_route_branch.requires_grad_(True)
    parameters = list(actor.memory_route_branch.parameters())
    assert all(p.is_cuda for p in actor.parameters())
    frozen = {k: v.detach().clone() for k, v in actor.state_dict().items() if not k.startswith('memory_route_branch.')}
    optimizer = torch.optim.Adam(parameters, lr=CONFIG['learning_rate'])
    completed = 0
    if a.resume:
        cp = torch.load(a.resume, map_location='cuda', weights_only=True)
        require_checkpoint_scope(cp, SCOPE)
        assert cp['version'] == CHECKPOINT and cp['weights_sha256'] == model_hash
        assert cp['config'] == CONFIG and cp['data_sha256'] == sha(a.data) and cp['protocol_sha256'] == sha(a.protocol)
        assert cp['parent_weights_sha256'] == sha(parent_path) and cp['frozen_behavior_sha256'] == fixed_identity(model)
        completed = cp['completed_steps']
        assert 0 < completed < a.steps
        actor.memory_route_branch.load_state_dict(cp['route'])
        optimizer.load_state_dict(copy.deepcopy(cp['optimizer']))
        torch.set_rng_state(cp['cpu_rng'].cpu())
        torch.cuda.set_rng_state_all([r.cpu() for r in cp['cuda_rng']])
    else:
        assert model_hash == sha(parent_path) and not optimizer.state
    train, diagnostic = (load_data(report, split, reference) for split in ('train', 'diagnostic'))
    with torch.no_grad():
        before = {s: metrics(actor.memory_route_branch, d) for s, d in [('train', train), ('diagnostic', diagnostic)]}
    losses = []
    for _ in range(completed, a.steps):
        imitation, retention, abstention = [], [], []
        for e in train:
            value = prediction(actor.memory_route_branch, e)
            imitation.append((value[e['idx']]-e['latent_target']).square())
            retention.append((value[e['idx']]-e['raw'][e['idx'], :2]).square())
            if bool(e['unlabelled'].any()):
                abstention.append(actor.memory_route_branch(e['x'])[e['unlabelled']].square())
        loss = torch.cat(imitation).mean() + CONFIG['source_retention']*torch.cat(retention).mean()
        if abstention:
            loss = loss + CONFIG['unlabelled_active_retention']*torch.cat(abstention).mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        norm = nn.utils.clip_grad_norm_(parameters, CONFIG['max_grad_norm'])
        assert torch.isfinite(norm)
        optimizer.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        after = {s: metrics(actor.memory_route_branch, d) for s, d in [('train', train), ('diagnostic', diagnostic)]}
        for e in train+diagnostic:
            full, _ = sequence(actor, e['context'])
            torch.testing.assert_close(full[:, :2], prediction(actor.memory_route_branch, e), atol=2e-5, rtol=1e-5)
            assert torch.equal(full[:, 2:], e['raw'][:, 2:])
            assert torch.equal(full[~e['active']], e['raw'][~e['active']])
    assert all(torch.equal(actor.state_dict()[k], v) for k, v in frozen.items())
    result = copy.deepcopy(model)
    export(result, actor.memory_route_branch)
    assert [k for k in model if model[k] != result[k]] == ['memory_route_movement']
    assert fixed_identity(result) == fixed_identity(parent)
    a.out.mkdir(parents=True)
    save(a.out/'weights.json', result)
    cp = dict(version=CHECKPOINT, checkpoint_training_scope=SCOPE, config=CONFIG,
              weights_sha256=sha(a.out/'weights.json'), parent_weights_sha256=sha(parent_path),
              frozen_behavior_sha256=fixed_identity(parent), data_sha256=sha(a.data), protocol_sha256=sha(a.protocol),
              completed_steps=a.steps, route=actor.memory_route_branch.state_dict(), optimizer=optimizer.state_dict(),
              cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    torch.save(cp, a.out/'checkpoint.pt')
    restored = torch.load(a.out/'checkpoint.pt', map_location='cuda', weights_only=True)
    require_checkpoint_scope(restored, SCOPE)
    assert all(torch.equal(v, restored['route'][k]) for k, v in actor.memory_route_branch.state_dict().items())
    try:
        require_checkpoint_scope(restored)
    except ValueError:
        pass
    else:
        raise AssertionError('Geometric auxiliary checkpoint entered generic PPO')
    assert sha(a.model) == model_hash and all(sha(p) == h for p, h in report['inputs'].items())
    save(a.out/'report.json', dict(version=SCOPE, state='passed', device=torch.cuda.get_device_name(), config=CONFIG,
         training_performed=True, completed_steps=a.steps, resumed_steps=completed,
         trainable_parameters=sum(p.numel() for p in parameters), before=before, after=after,
         loss_first=losses[0], loss_last=losses[-1], frozen_other_parameters_exact=True,
         inactive_and_nonmovement_raw_exact=True, cached_training_verified_against_full_context=True,
         inputs={str(a.model.resolve()): model_hash, str(a.data.resolve()): sha(a.data), str(a.protocol.resolve()): sha(a.protocol)},
         weights_sha256=sha(a.out/'weights.json'), checkpoint_sha256=sha(a.out/'checkpoint.pt'), promotion=False,
         scope='Observed static walk-waypoint auxiliary, not executed expert or optimal actions. Frozen mean camera projection approximates stochastic motion; native quality remains unproven. No runtime route follower or hidden enemy state.'))
    print(json.dumps(dict(before=before, after=after)), flush=True)


if __name__ == '__main__':
    main()

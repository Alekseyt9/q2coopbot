"""CUDA motor-only BC with visible moving/stop groups and typed Adam resume."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from ppo_combat import torch, nn, training_devices
from train_combat_movement_bc import load_data, chosen
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model
from combat_motor_residual import inputs as motor_inputs, export as export_motor
from combat_checkpoint_scope import require_checkpoint_scope
from evaluate_registered_combat_models import read, save, sha

SCOPE = 'combat_visible_balanced_motor_bc_v1'
CHECKPOINT = 'combat_visible_balanced_motor_bc_checkpoint_v1'
CONFIG = dict(version=SCOPE, seed=20261025, learning_rate=.0003, moving_weight=.5,
              visible_retention=.02, no_visible_retention=.1, latent_clip=.95,
              max_grad_norm=1., planned_steps=150, actor_scope='motor_only',
              checkpoint_selection='fixed_final_step', runtime_visibility_gate=False)


def frozen_identity(model):
    fixed = copy.deepcopy(model)
    fixed.pop('motor_residual')
    return hashlib.sha256(json.dumps(fixed, sort_keys=True, allow_nan=False).encode()).hexdigest()


def prepare(reference, data):
    x, positions, y = data
    selected = x[positions[:, 0], positions[:, 1]]
    visible = selected[:, 73:169].reshape(-1, 8, 12)[:, :, 0].ne(0).any(-1)
    moving = y.abs().sum(-1).gt(0)
    with torch.no_grad():
        raw = chosen(reference, data).detach()
        inputs = motor_inputs(selected).detach()
        motor = reference.motor_branch(inputs).detach()
    assert bool((visible & moving).any()) and bool((visible & ~moving).any())
    return dict(data=data, raw=raw, inputs=inputs, motor=motor, target=y,
                latent_target=torch.atanh(y.clamp(-CONFIG['latent_clip'], CONFIG['latent_clip'])),
                visible=visible, moving=moving)


def predict(actor, data):
    return data['raw'][:, :2] + actor.motor_branch(data['inputs']) - data['motor']


def grouped_metrics(raw, data, std):
    visible, moving = data['visible'], data['moving']
    generator = torch.Generator(device='cuda').manual_seed(73471)
    noise = torch.randn((len(raw), 32, 2), generator=generator, device='cuda')*std
    sampled = (raw[:, None, :] + noise).tanh()
    result = {}
    for name, mask in [('moving', visible & moving), ('stop', visible & ~moving), ('no_visible', ~visible)]:
        rows = int(mask.sum())
        result[name] = dict(rows=rows,
            mean_command_mae=float((raw[mask].tanh()-data['target'][mask]).abs().mean()) if rows else None,
            sampled_command_mae=float((sampled[mask]-data['target'][mask, None, :]).abs().mean()) if rows else None,
            always_zero_label_mae=float(data['target'][mask].abs().mean()) if rows else None,
            mean_change_from_source=float((raw[mask]-data['raw'][mask, :2]).abs().mean()) if rows else None)
    result['balanced_visible_mean_mae'] = .5*(result['moving']['mean_command_mae']+result['stop']['mean_command_mae'])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--protocol', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--steps', type=int, default=150)
    ap.add_argument('--resume', type=Path)
    a = ap.parse_args()
    assert not a.out.exists() and 1 <= a.steps <= CONFIG['planned_steps']
    protocol = read(a.protocol)
    assert protocol['training_config'] == CONFIG and protocol['state'] == 'planned'
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
    report = read(a.data)
    assert report['version'] == 'combat_movement_demonstrations_v1' and report['state'] == 'passed'
    assert report['ppo_eligible'] is False
    assert sha(a.data) == protocol['data_sha256']
    for path, digest in report['inputs'].items():
        assert sha(path) == digest
    for split in ('train', 'diagnostic'):
        worlds = {(e['map'], e['seed']) for e in report['episodes'] if e['split'] == split}
        other = {(e['map'], e['seed']) for e in report['episodes'] if e['split'] != split}
        assert worlds and not worlds & other
    model, model_hash = read(a.model), sha(a.model)
    parent_path = Path(protocol['source_model'])
    assert sha(parent_path) == protocol['source_sha256']
    parent = read(parent_path)
    assert frozen_identity(model) == frozen_identity(parent)
    assert model['memory_route_movement']['version'] == 'combat_memory_route_movement_v1'
    route = model['memory_route_movement']['layers'][-1]
    assert not any(v for row in route['weight'] for v in row) and not any(route['bias'])
    assert not model.get('movement_inertia') and not model.get('memory_recovery')
    actor = branch(base_model(model, allow_shared=True), 'actor').eval().requires_grad_(False)
    reference = branch(base_model(parent, allow_shared=True), 'actor').eval().requires_grad_(False)
    assert actor.motor_branch is not None
    actor.motor_branch.requires_grad_(True)
    parameters = list(actor.motor_branch.parameters())
    assert all(p.is_cuda for p in actor.parameters())
    frozen = {k: v.detach().clone() for k, v in actor.state_dict().items() if not k.startswith('motor_branch.')}
    optimizer = torch.optim.Adam(parameters, lr=CONFIG['learning_rate'])
    completed = 0
    if a.resume:
        cp = torch.load(a.resume, map_location='cuda', weights_only=True)
        require_checkpoint_scope(cp, SCOPE)
        assert cp['version'] == CHECKPOINT and cp['weights_sha256'] == model_hash
        assert cp['config'] == CONFIG and cp['data_sha256'] == sha(a.data)
        assert cp['protocol_sha256'] == sha(a.protocol) and cp['parent_weights_sha256'] == sha(parent_path)
        assert cp['frozen_behavior_sha256'] == frozen_identity(model)
        completed = cp['completed_steps']
        assert 0 < completed < a.steps
        actor.motor_branch.load_state_dict(cp['motor'])
        optimizer.load_state_dict(copy.deepcopy(cp['optimizer']))
        torch.set_rng_state(cp['cpu_rng'].cpu())
        torch.cuda.set_rng_state_all([r.cpu() for r in cp['cuda_rng']])
    else:
        assert model_hash == sha(parent_path) and not optimizer.state
    train, diagnostic = (prepare(reference, load_data(report, split)) for split in ('train', 'diagnostic'))
    std = torch.tensor(model['log_std'][:2], device='cuda').exp()
    with torch.no_grad():
        before = {s: grouped_metrics(predict(actor, d), d, std) for s, d in [('train', train), ('diagnostic', diagnostic)]}
    losses = []
    for _ in range(completed, a.steps):
        prediction = predict(actor, train)
        imitation = (prediction-train['latent_target']).square().mean(-1)
        moving, visible = train['moving'], train['visible']
        loss = .5*imitation[visible & moving].mean() + .5*imitation[visible & ~moving].mean()
        retention = (prediction-train['raw'][:, :2]).square().mean(-1)
        loss = loss + CONFIG['visible_retention']*retention[visible].mean()
        if bool((~visible).any()):
            loss = loss + CONFIG['no_visible_retention']*retention[~visible].mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        norm = nn.utils.clip_grad_norm_(parameters, CONFIG['max_grad_norm'])
        assert torch.isfinite(norm)
        optimizer.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        after = {s: grouped_metrics(predict(actor, d), d, std) for s, d in [('train', train), ('diagnostic', diagnostic)]}
        for data in (train, diagnostic):
            full = chosen(actor, data['data'])
            torch.testing.assert_close(full[:, :2], predict(actor, data), atol=2e-5, rtol=1e-5)
            assert torch.equal(full[:, 2:], data['raw'][:, 2:])
    assert all(torch.equal(actor.state_dict()[k], v) for k, v in frozen.items())
    result = copy.deepcopy(model)
    export_motor(result, actor.motor_branch)
    assert [k for k in model if result[k] != model[k]] == ['motor_residual']
    assert frozen_identity(result) == frozen_identity(parent)
    a.out.mkdir(parents=True)
    save(a.out/'weights.json', result)
    checkpoint = dict(version=CHECKPOINT, checkpoint_training_scope=SCOPE, config=CONFIG,
        weights_sha256=sha(a.out/'weights.json'), parent_weights_sha256=sha(parent_path),
        frozen_behavior_sha256=frozen_identity(parent), data_sha256=sha(a.data), protocol_sha256=sha(a.protocol),
        completed_steps=a.steps, motor=actor.motor_branch.state_dict(), optimizer=optimizer.state_dict(),
        cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    torch.save(checkpoint, a.out/'checkpoint.pt')
    restored = torch.load(a.out/'checkpoint.pt', map_location='cuda', weights_only=True)
    require_checkpoint_scope(restored, SCOPE)
    assert all(torch.equal(v, restored['motor'][k]) for k, v in actor.motor_branch.state_dict().items())
    try:
        require_checkpoint_scope(restored)
    except ValueError:
        pass
    else:
        raise AssertionError('Motor BC checkpoint entered generic PPO')
    assert sha(a.model) == model_hash
    save(a.out/'report.json', dict(version=SCOPE, state='passed', device=torch.cuda.get_device_name(),
        training_performed=True, completed_steps=a.steps, resumed_steps=completed,
        trainable_parameters=sum(p.numel() for p in parameters), config=CONFIG, before=before, after=after,
        loss_first=losses[0], loss_last=losses[-1], only_motor_weights_changed=True,
        cached_training_verified_against_full_actor=True, other_raw_outputs_exact=True,
        frozen_other_parameters_exact=True, std_unchanged=True, critic_unchanged=True,
        inputs={str(a.model.resolve()): model_hash, str(a.data.resolve()): sha(a.data), str(a.protocol.resolve()): sha(a.protocol)},
        weights_sha256=sha(a.out/'weights.json'), checkpoint_sha256=sha(a.out/'checkpoint.pt'),
        promotion=False, scope='Visible-context mean movement supervision with original retention; motor applies globally and may change conditional sampled aim. Fixed Gaussian noise is preserved. Not PPO, a runtime visibility gate, executed optimal policy or native quality evidence.'))
    print(json.dumps(dict(before=before, after=after)), flush=True)


if __name__ == '__main__':
    main()

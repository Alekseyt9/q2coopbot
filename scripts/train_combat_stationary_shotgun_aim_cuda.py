"""CUDA-only geometric fine-aim auxiliary; existing motion-aim branch only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from ppo_combat import torch, nn, training_devices
from ppo_recurrent import prepare_context, sequence
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model
from combat_precision_head import selected_means
from combat_sampled_motion_aim import inputs, uses_vertical, export, VERTICAL_CONTROL_VERSION
from combat_motor_residual import reference as motor_reference
from combat_checkpoint_scope import require_checkpoint_scope
from train_combat_memory_route_ppo import admit
from evaluate_registered_combat_models import read, save, sha

SCOPE = 'combat_stationary_shotgun_aim_aux_v1'
CHECKPOINT = 'combat_stationary_shotgun_aim_aux_checkpoint_v1'
CONFIG = dict(version=SCOPE, seed=20261025, planned_steps=200, learning_rate=.0003,
              other_context_retention=1., labelled_retention=.05, max_grad_norm=1.,
              actor_scope='existing_sampled_motion_aim_branch', checkpoint_selection='fixed_final_step')


def fixed_identity(model):
    result = copy.deepcopy(model)
    result.pop('sampled_motion_aim')
    return hashlib.sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load(report, split, actor):
    result = []
    for e in report['episodes']:
        if e['split'] != split:
            continue
        assert sha(e['native']) == e['native_sha256']
        rows, history, _, meta = admit(Path(e['source_cuda']), e['source_model_sha256'], [])
        assert len(rows) == e['rows']
        with torch.no_grad():
            raw, error = sequence(actor, prepare_context(history, rows, 'cuda'), 32, True, 'actor')
            assert error < 2e-5
            features = torch.tensor([r['features'] for r in rows], device='cuda')
            sample = [r['sample'] for r in rows]
            target = torch.tensor([s.get('target', 0) for s in sample], device='cuda', dtype=torch.long)
            mode = torch.tensor([s.get('aim_mode', 0) for s in sample], device='cuda', dtype=torch.long)
            vertical = torch.tensor([s['vertical'] for s in sample], device='cuda', dtype=torch.long)
            movement = torch.tensor([s['latent'][:2] for s in sample], device='cuda')
            reference_raw = motor_reference(actor.motor_branch, features, raw) if actor.motor_branch is not None else raw
            x = inputs(features, reference_raw, movement, target, mode)
            if uses_vertical(actor.sampled_motion_aim_branch):
                extra = torch.nn.functional.one_hot(vertical, 3).to(raw.dtype)
                if actor.sampled_motion_aim_branch.motion_version == VERTICAL_CONTROL_VERSION:
                    extra = torch.zeros_like(extra)
                x = torch.cat((x, extra), -1)
            baseline = actor.sampled_motion_aim_branch(x)
            base = selected_means(raw, target, mode)[:, 2:]
            full = selected_means(raw, target, mode, features, movement,
                                  actor.sampled_motion_aim_branch, vertical, actor.motor_branch)[:, 2:]
            torch.testing.assert_close(full[target > 0], (base+baseline)[target > 0], atol=2e-5, rtol=1e-5)
            idx = torch.tensor([v['index'] for v in e['labels']], device='cuda', dtype=torch.long)
            y = torch.tensor([v['delta_degrees'] for v in e['labels']], device='cuda').reshape(-1, 2)
            if len(idx):
                assert bool(mode[idx].eq(1).all() & target[idx].gt(0).all())
            retain = target.gt(0)
            retain[idx] = False
        result.append(dict(episode=e, x=x.detach(), base=base.detach(), baseline=baseline.detach(),
                           idx=idx, y=y.detach(), retain=retain, labelled=bool(len(idx)),
                           features=features, raw=raw, movement=movement, target=target, mode=mode, vertical=vertical))
    assert result and any(e['labelled'] for e in result)
    return result


def metrics(head, entries):
    predicted, wanted, retained = [], [], []
    for e in entries:
        value = head(e['x'])
        if e['labelled']:
            predicted.append((e['base']+value)[e['idx']].tanh()*15)
            wanted.append(e['y'])
        if bool(e['retain'].any()):
            retained.append((value-e['baseline'])[e['retain']])
    p, y = torch.cat(predicted), torch.cat(wanted)
    return dict(rows=len(p), geometric_delta_mae_degrees=float((p-y).abs().mean()),
                yaw_mae_degrees=float((p-y)[:, 0].abs().mean()), pitch_mae_degrees=float((p-y)[:, 1].abs().mean()),
                other_context_mean_latent_change=float(torch.cat(retained).abs().mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--protocol', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--resume', type=Path)
    a = ap.parse_args()
    assert not a.out.exists() and 1 <= a.steps <= CONFIG['planned_steps']
    training_devices()
    torch.set_default_device('cuda'); torch.set_num_threads(2); torch.manual_seed(CONFIG['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    protocol, report, model = read(a.protocol), read(a.data), read(a.model)
    parent = Path(protocol['source_model'])
    assert protocol['state'] == 'planned' and protocol['config'] == CONFIG
    assert protocol['data_sha256'] == sha(a.data) and sha(parent) == protocol['source_sha256']
    assert report['state'] == 'passed' and report['version'] == 'combat_stationary_shotgun_aim_data_v1'
    assert not report['ppo_eligible'] and not report['executed_demonstration']
    assert all(sha(p) == h for p, h in report['inputs'].items())
    worlds = {s: {(e['map'], e['seed']) for e in report['episodes'] if e['split'] == s} for s in ('train', 'diagnostic')}
    assert not worlds['train'] & worlds['diagnostic'] and fixed_identity(model) == fixed_identity(read(parent))
    actor = branch(base_model(model, allow_shared=True), 'actor').eval().requires_grad_(False)
    reference = branch(base_model(read(parent), allow_shared=True), 'actor').eval().requires_grad_(False)
    head = actor.sampled_motion_aim_branch
    assert head is not None and all(p.is_cuda for p in actor.parameters())
    head.requires_grad_(True)
    frozen = {k: v.detach().clone() for k, v in actor.state_dict().items() if not k.startswith('sampled_motion_aim_branch.')}
    optimizer = torch.optim.Adam(head.parameters(), lr=CONFIG['learning_rate'])
    completed = 0
    if a.resume:
        cp = torch.load(a.resume, map_location='cuda', weights_only=True)
        require_checkpoint_scope(cp, SCOPE)
        assert cp['version'] == CHECKPOINT and cp['weights_sha256'] == sha(a.model)
        assert cp['data_sha256'] == sha(a.data) and cp['protocol_sha256'] == sha(a.protocol)
        assert cp['parent_weights_sha256'] == sha(parent) and cp['config'] == CONFIG
        assert cp['frozen_behavior_sha256'] == fixed_identity(model)
        completed = cp['completed_steps']; assert 0 < completed < a.steps
        head.load_state_dict(cp['aim']); optimizer.load_state_dict(copy.deepcopy(cp['optimizer']))
        torch.set_rng_state(cp['cpu_rng'].cpu()); torch.cuda.set_rng_state_all([v.cpu() for v in cp['cuda_rng']])
    else:
        assert sha(a.model) == sha(parent)
    train, diagnostic = (load(report, s, reference) for s in ('train', 'diagnostic'))
    with torch.no_grad():
        before = {s: metrics(head, d) for s, d in [('train', train), ('diagnostic', diagnostic)]}
    losses = []
    for _ in range(completed, a.steps):
        errors, retain, labelled_retain = [], [], []
        for e in train:
            residual = head(e['x'])
            if e['labelled']:
                desired = torch.atanh(e['y']/15)
                errors.append(((e['base']+residual)[e['idx']]-desired).square())
                labelled_retain.append((residual-e['baseline'])[e['idx']].square())
            if bool(e['retain'].any()):
                retain.append((residual-e['baseline'])[e['retain']].square())
        loss = torch.cat(errors).mean()+CONFIG['other_context_retention']*torch.cat(retain).mean()
        loss = loss+CONFIG['labelled_retention']*torch.cat(labelled_retain).mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad(); loss.backward()
        norm = nn.utils.clip_grad_norm_(head.parameters(), CONFIG['max_grad_norm']); assert torch.isfinite(norm)
        optimizer.step(); losses.append(float(loss.detach()))
    with torch.no_grad():
        after = {s: metrics(head, d) for s, d in [('train', train), ('diagnostic', diagnostic)]}
    assert all(torch.equal(actor.state_dict()[k], v) for k, v in frozen.items())
    result = copy.deepcopy(model); export(result, head)
    assert [k for k in model if model[k] != result[k]] == ['sampled_motion_aim']
    assert fixed_identity(result) == fixed_identity(read(parent))
    a.out.mkdir(parents=True); save(a.out/'weights.json', result)
    cp = dict(version=CHECKPOINT, checkpoint_training_scope=SCOPE, config=CONFIG,
              weights_sha256=sha(a.out/'weights.json'), parent_weights_sha256=sha(parent),
              frozen_behavior_sha256=fixed_identity(model), data_sha256=sha(a.data), protocol_sha256=sha(a.protocol),
              completed_steps=a.steps, aim=head.state_dict(), optimizer=optimizer.state_dict(),
              cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    torch.save(cp, a.out/'checkpoint.pt')
    require_checkpoint_scope(torch.load(a.out/'checkpoint.pt', map_location='cuda', weights_only=True), SCOPE)
    try:
        require_checkpoint_scope(cp)
    except ValueError:
        pass
    else:
        raise AssertionError('Geometric aim checkpoint admitted as generic PPO')
    assert all(sha(p) == h for p, h in report['inputs'].items())
    save(a.out/'report.json', dict(version=SCOPE, state='passed', device=torch.cuda.get_device_name(),
        config=CONFIG, completed_steps=a.steps, resumed_steps=completed, before=before, after=after,
        loss_first=losses[0], loss_last=losses[-1], trainable_parameters=sum(p.numel() for p in head.parameters()),
        frozen_other_parameters_exact=True, cached_means_verified=True, training_performed=True, promotion=False,
        inputs={str(a.model.resolve()): sha(a.model), str(a.data.resolve()): sha(a.data), str(a.protocol.resolve()): sha(a.protocol)},
        weights_sha256=sha(a.out/'weights.json'), checkpoint_sha256=sha(a.out/'checkpoint.pt'),
        scope='Existing shared motion-aim branch: geometric SG stationary fine targets, other-context retention. Changes may generalize to other weapons/modes; no exact SG-only runtime gate, no quality claim.'))
    print(json.dumps(dict(before=before, after=after)), flush=True)


if __name__ == '__main__':
    main()

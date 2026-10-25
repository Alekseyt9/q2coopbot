"""Dedicated CUDA geometric SG fine residual; every legacy parameter frozen."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from ppo_combat import torch, nn, training_devices
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model
from combat_precision_head import selected_means
from combat_fine_weapon_aim import inputs, active, export, VERSION
from train_combat_stationary_shotgun_aim_cuda import load as load_legacy
from combat_checkpoint_scope import require_checkpoint_scope
from evaluate_registered_combat_models import read, save, sha

SCOPE = 'combat_gated_shotgun_fine_aux_v1'
CHECKPOINT = 'combat_gated_shotgun_fine_aux_checkpoint_v1'
CONFIG = dict(version=SCOPE, seed=20261025, planned_steps=200, learning_rate=.001,
              unlabelled_gated_retention=.2, labelled_retention=.02, max_grad_norm=1.,
              actor_scope='fine_weapon_aim_only', checkpoint_selection='fixed_final_step')


def fixed_identity(model):
    result = copy.deepcopy(model); result.pop('fine_weapon_aim', None)
    return hashlib.sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()


def prepare(report, split, reference):
    entries = load_legacy(report, split, reference)
    with torch.no_grad():
        for e in entries:
            e['legacy'] = torch.where(e['target'].gt(0)[:, None], e['base']+e['baseline'], e['base'])
            e['fine_inputs'] = inputs(e['features'], e['raw'], e['movement'], e['target'], e['mode'], e['legacy'], reference.motor_branch)
            e['gate'] = active(e['features'], e['target'], e['mode'])
            assert bool(e['gate'][e['idx']].all())
            e['unlabelled'] = e['gate'].clone(); e['unlabelled'][e['idx']] = False
    return entries


def prediction(head, e):
    return torch.where(e['gate'][:, None], e['legacy']+head(e['fine_inputs']), e['legacy'])


def metrics(head, entries):
    errors, unlabelled = [], []
    gated, off = 0, 0
    for e in entries:
        value = prediction(head, e)
        if e['labelled']:
            errors.append((value[e['idx']].tanh()*15-e['y']).abs())
        assert torch.equal(value[~e['gate']], e['legacy'][~e['gate']])
        gated += int(e['gate'].sum()); off += int((~e['gate']).sum())
        if bool(e['unlabelled'].any()):
            unlabelled.append((value-e['legacy'])[e['unlabelled']].abs())
    err = torch.cat(errors)
    return dict(rows=len(err), geometric_delta_mae_degrees=float(err.mean()),
                yaw_mae_degrees=float(err[:, 0].mean()), pitch_mae_degrees=float(err[:, 1].mean()),
                gated_rows=gated, exact_preserved_outside_gate_rows=off,
                unlabelled_gated_mean_latent_change=float(torch.cat(unlabelled).mean()) if unlabelled else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('model', 'data', 'protocol', 'out'):
        ap.add_argument('--'+key, required=True, type=Path)
    ap.add_argument('--steps', type=int, default=200); ap.add_argument('--resume', type=Path)
    a = ap.parse_args(); assert not a.out.exists() and 1 <= a.steps <= CONFIG['planned_steps']
    training_devices(); torch.set_default_device('cuda'); torch.set_num_threads(2); torch.manual_seed(CONFIG['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    protocol, report, model = read(a.protocol), read(a.data), read(a.model)
    parent, zero = Path(protocol['source_model']), Path(protocol['zero_model'])
    assert protocol['state'] == 'planned' and protocol['config'] == CONFIG
    assert sha(parent) == protocol['source_sha256'] and sha(zero) == protocol['zero_sha256']
    assert sha(a.data) == protocol['data_sha256'] and report['state'] == 'passed'
    assert report['version'] == 'combat_stationary_shotgun_aim_data_v1' and not report['ppo_eligible']
    assert all(sha(p) == h for p, h in report['inputs'].items())
    worlds = {s: {(e['map'], e['seed']) for e in report['episodes'] if e['split'] == s} for s in ('train', 'diagnostic')}
    assert not worlds['train'] & worlds['diagnostic'] and model['fine_weapon_aim']['version'] == VERSION
    assert fixed_identity(model) == fixed_identity(read(parent)) == fixed_identity(read(zero))
    actor = branch(base_model(model, allow_shared=True), 'actor').eval().requires_grad_(False)
    reference = branch(base_model(read(parent), allow_shared=True), 'actor').eval().requires_grad_(False)
    head = actor.fine_weapon_aim_branch; head.requires_grad_(True)
    assert all(p.is_cuda for p in actor.parameters())
    frozen = {k: v.detach().clone() for k, v in actor.state_dict().items() if not k.startswith('fine_weapon_aim_branch.')}
    optimizer = torch.optim.Adam(head.parameters(), lr=CONFIG['learning_rate']); completed = 0
    if a.resume:
        cp = torch.load(a.resume, map_location='cuda', weights_only=True); require_checkpoint_scope(cp, SCOPE)
        assert cp['version'] == CHECKPOINT and cp['weights_sha256'] == sha(a.model)
        assert cp['parent_weights_sha256'] == sha(parent) and cp['zero_weights_sha256'] == sha(zero)
        assert cp['data_sha256'] == sha(a.data) and cp['protocol_sha256'] == sha(a.protocol) and cp['config'] == CONFIG
        assert cp['frozen_behavior_sha256'] == fixed_identity(model)
        completed = cp['completed_steps']; assert 0 < completed < a.steps
        head.load_state_dict(cp['fine']); optimizer.load_state_dict(copy.deepcopy(cp['optimizer']))
        torch.set_rng_state(cp['cpu_rng'].cpu()); torch.cuda.set_rng_state_all([v.cpu() for v in cp['cuda_rng']])
    else:
        assert sha(a.model) == sha(zero)
    train, diagnostic = (prepare(report, s, reference) for s in ('train', 'diagnostic'))
    with torch.no_grad():
        before = {s: metrics(head, d) for s, d in [('train', train), ('diagnostic', diagnostic)]}
    losses = []
    for _ in range(completed, a.steps):
        errors, retain, labelled = [], [], []
        for e in train:
            delta = head(e['fine_inputs'])
            if e['labelled']:
                errors.append(((e['legacy']+delta)[e['idx']]-torch.atanh(e['y']/15)).square())
                labelled.append(delta[e['idx']].square())
            if bool(e['unlabelled'].any()):
                retain.append(delta[e['unlabelled']].square())
        loss = torch.cat(errors).mean()+CONFIG['labelled_retention']*torch.cat(labelled).mean()
        if retain:
            loss = loss+CONFIG['unlabelled_gated_retention']*torch.cat(retain).mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad(); loss.backward()
        norm = nn.utils.clip_grad_norm_(head.parameters(), CONFIG['max_grad_norm']); assert torch.isfinite(norm)
        optimizer.step(); losses.append(float(loss.detach()))
    with torch.no_grad():
        after = {s: metrics(head, d) for s, d in [('train', train), ('diagnostic', diagnostic)]}
        for e in train+diagnostic:
            means = selected_means(e['raw'], e['target'], e['mode'], e['features'], e['movement'],
                                   actor.sampled_motion_aim_branch, e['vertical'], actor.motor_branch)
            torch.testing.assert_close(means[:, 2:], prediction(head, e), atol=2e-5, rtol=1e-5)
            assert torch.equal(means[~e['gate'], 2:], e['legacy'][~e['gate']])
    assert all(torch.equal(actor.state_dict()[k], v) for k, v in frozen.items())
    result = copy.deepcopy(model); export(result, head)
    assert [k for k in model if model[k] != result[k]] == ['fine_weapon_aim'] and fixed_identity(result) == fixed_identity(read(parent))
    a.out.mkdir(parents=True); save(a.out/'weights.json', result)
    cp = dict(version=CHECKPOINT, checkpoint_training_scope=SCOPE, config=CONFIG,
        weights_sha256=sha(a.out/'weights.json'), parent_weights_sha256=sha(parent), zero_weights_sha256=sha(zero),
        frozen_behavior_sha256=fixed_identity(model), data_sha256=sha(a.data), protocol_sha256=sha(a.protocol),
        completed_steps=a.steps, fine=head.state_dict(), optimizer=optimizer.state_dict(),
        cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    torch.save(cp, a.out/'checkpoint.pt'); require_checkpoint_scope(torch.load(a.out/'checkpoint.pt', map_location='cuda', weights_only=True), SCOPE)
    try:
        require_checkpoint_scope(cp)
    except ValueError:
        pass
    else:
        raise AssertionError('Fine auxiliary checkpoint admitted to generic PPO')
    save(a.out/'report.json', dict(version=SCOPE, state='passed', device=torch.cuda.get_device_name(), config=CONFIG,
        completed_steps=a.steps, resumed_steps=completed, before=before, after=after,
        trainable_parameters=sum(p.numel() for p in head.parameters()), loss_first=losses[0], loss_last=losses[-1],
        frozen_legacy_parameters_exact=True, full_selected_means_verified=True, outside_gate_exact=True,
        training_performed=True, promotion=False, weights_sha256=sha(a.out/'weights.json'),
        checkpoint_sha256=sha(a.out/'checkpoint.pt'), inputs={str(p.resolve()): sha(p) for p in (a.model, a.data, a.protocol)},
        scope='Geometric stationary SG fine labels, separate learned residual. No effect outside SGfine visible available target on same states. Native subsequent state trajectories can differ. Not shot-quality evidence.'))
    print(json.dumps(dict(before=before, after=after)), flush=True)


if __name__ == '__main__':
    main()

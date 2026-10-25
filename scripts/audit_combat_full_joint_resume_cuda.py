"""Numerical gate only: real replay, full gradients, gated heads, exact resume."""
import argparse
import copy
import json
from pathlib import Path
from ppo_combat import torch, training_devices
from ppo_recurrent import prepare_context, sequence
from combat_full_joint_ppo import FullJointScope, SCOPE, equal_tree
from combat_joint_aim_policy import events, probabilities
from combat_memory_route_movement import apply as apply_route
from combat_memory_route_features import PARENT_WIDTH
from combat_fine_weapon_aim import apply as apply_fine
from evaluate_registered_combat_models import read, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('model', 'data', 'out'):
        ap.add_argument('--'+key, type=Path, required=True)
    a = ap.parse_args()
    assert not a.out.exists()
    training_devices(); torch.set_default_device('cuda'); torch.set_num_threads(2)
    torch.manual_seed(20261025); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    source_hash = sha(a.model)
    meta = read(a.data/'report.json')
    assert meta['model_sha256'] == source_hash and meta['numerical_verification'] == 'cuda_verified_v1'
    assert sha(a.data/'rollout.jsonl') == meta['rollout_sha256']
    assert sha(a.data/'sequence.jsonl') == meta['sequence_sha256']
    load = lambda p: [json.loads(s) for s in p.read_text().splitlines()]
    rows, history = load(a.data/'rollout.jsonl'), load(a.data/'sequence.jsonl')
    prepared = prepare_context(history, rows, 'cuda')
    model = read(a.model)
    config = dict(version=SCOPE, actor_lr=1e-5, route_lr=1e-4, fine_lr=1e-4,
                  candidate_lr=1e-4, value_lr=1e-4)
    scope = FullJointScope(model, config)
    x = torch.tensor([r['features'] for r in rows], device='cuda', dtype=torch.float32)
    cat = lambda key: torch.tensor([r['sample'].get(key, 0) for r in rows], device='cuda', dtype=torch.long)
    target, mode, vertical, weapon = (cat(k) for k in ('target', 'aim_mode', 'vertical', 'weapon'))
    final = torch.tensor([r['sample']['latent'] for r in rows], device='cuda', dtype=torch.float32)
    attack = torch.tensor([float(r['sample']['attack']) for r in rows], device='cuda')
    source, choice, timing, _ = events(rows, x, target, mode, vertical, weapon)
    score_inputs = None
    if scope.joint.geometry_head:
        score_inputs = (
            torch.tensor([r['features'] for r in rows], device='cuda', dtype=torch.float64),
            torch.tensor([(r['sample'].get('aim_candidate') or {}).get('source_latent',r['sample']['latent']) for r in rows], device='cuda', dtype=torch.float64),
            torch.tensor([(r['sample'].get('aim_candidate') or {}).get('timing',[0.,0.]) for r in rows], device='cuda', dtype=torch.float64))
    def score(s):
        raw, _ = sequence(s.actor, prepared, 32, False, 'actor')
        lp = probabilities(s.joint, raw, s.std, source, final, attack, vertical, weapon,
            x, target, mode, choice, timing, s.actor.sampled_motion_aim_branch,
            s.actor.motor_branch, s.actor.conditional_fire_branch, score_inputs=score_inputs)[0]
        v, _ = sequence(s.value, prepared, 32, False, 'value')
        return raw, lp, v
    with torch.no_grad():
        raw, lp, v = score(scope)
        log_error = float((lp-torch.tensor([r['sample']['log_probability'] for r in rows], device='cuda')).abs().max())
        value_error = float((v[:, 0]-torch.tensor([r['sample']['value'] for r in rows], device='cuda')).abs().max())
        assert log_error < .003 and value_error < 1e-4, (log_error, value_error)
    # Controlled gate fixtures supplement real replay; they are NOT native
    # trajectories, labels, PPO data, or evidence of better combat behavior.
    route_x = x[:4].clone(); route_x[:, 0] = 1
    route_x[:, 73:169] = 0; route_x[:, PARENT_WIDTH] = 1
    route_x[:, 881] = 1; route_x[:, PARENT_WIDTH+1:PARENT_WIDTH+3] = 1
    fine_x = x[:4].clone(); fine_x[:, 0] = 1; fine_x[:, 1123] = 1
    fine_x[:, 73] = 1; fine_x[:, 426] = 1
    tt = torch.ones(4, device='cuda', dtype=torch.long)
    r0 = raw[:4].detach(); movement = source[:4, :2].detach()
    legacy = torch.zeros((4, 2), device='cuda')
    def step(s):
        s.actor_optimizer.zero_grad(); s.value_optimizer.zero_grad()
        rr, ll, vv = score(s)
        route = apply_route(s.actor.memory_route_branch, route_x, r0)
        fine = apply_fine(s.actor.fine_weapon_aim_branch, fine_x, r0, movement, tt, tt, legacy, s.actor.motor_branch)
        loss = -ll.mean() + (route[:, :2]-r0[:, :2]-.125).square().mean() + (fine-.125).square().mean()
        if s.joint.shared is not None:
            width = s.joint.shared[0].in_features
            cx = torch.linspace(-1, 1, 25*width, device='cuda').reshape(25, width)
            logits = s.joint.shared(cx)[:, 0] + s.joint.shared_beta*torch.arange(25, device='cuda').eq(0)
        else:
            cx = torch.zeros((1, 1359), device='cuda')
            logits = s.joint.head(cx)[0]
        loss = loss-logits.log_softmax(-1)[3]
        loss.backward(); (vv.square().mean()).backward()
        gradients = {}
        for group in s.actor_optimizer.param_groups:
            grads = [p.grad for p in group['params'] if p.grad is not None]
            assert grads and all(bool(torch.isfinite(g).all()) for g in grads), group['name']
            gradients[group['name']] = float(sum(g.abs().sum() for g in grads))
            assert gradients[group['name']] > 0, group['name']
        s.actor_optimizer.step(); s.value_optimizer.step(); s.verify()
        return gradients
    gradients = step(scope)
    # Gate invariants are checked after actual nonzero optimization.
    with torch.no_grad():
        off = route_x.clone(); off[:, PARENT_WIDTH] = 0
        assert torch.equal(apply_route(scope.actor.memory_route_branch, off, r0), r0)
        off = fine_x.clone(); off[:, 1123] = 0
        assert torch.equal(apply_fine(scope.actor.fine_weapon_aim_branch, off, r0, movement, tt, tt, legacy, scope.actor.motor_branch), legacy)
    a.out.mkdir(parents=True)
    save(a.out/'diagnostic-weights.json', scope.export())
    token = 'numerical_fixture_only_not_a_consumed_native_rollout'
    cp = scope.snapshot(sha(a.out/'diagnostic-weights.json'), [token])
    torch.save(cp, a.out/'diagnostic-checkpoint.pt')
    restored = FullJointScope(read(a.out/'diagnostic-weights.json'), config)
    assert restored.restore(torch.load(a.out/'diagnostic-checkpoint.pt', map_location='cuda', weights_only=True), cp['weights_sha256']) == [token]
    assert equal_tree(scope.actor_optimizer.state_dict(), restored.actor_optimizer.state_dict())
    assert equal_tree(scope.value_optimizer.state_dict(), restored.value_optimizer.state_dict())
    step(scope); scope_next = scope.snapshot('a'*64, [token])
    restored.restore(cp, cp['weights_sha256']); step(restored)
    resumed_next = restored.snapshot('a'*64, [token])
    assert equal_tree(scope_next, resumed_next), 'Interrupted continuation differs'
    rejected = []
    for label, mutate in (
        ('auxiliary_scope', lambda c: c.update(checkpoint_training_scope='combat_gated_shotgun_fine_aux_v1')),
        ('layout', lambda c: c.update(optimizer_layout=[])),
        ('weights_binding', lambda c: c.update(weights_sha256='b'*64))):
        bad = copy.deepcopy(cp); mutate(bad)
        probe = FullJointScope(read(a.out/'diagnostic-weights.json'), config)
        try: probe.restore(bad, cp['weights_sha256'])
        except (AssertionError, ValueError): rejected.append(label)
        else: raise AssertionError('Invalid checkpoint admitted: '+label)
    assert sha(a.model) == source_hash
    report = dict(state='passed', version=SCOPE, device='cuda', replay_rows=len(rows),
        model_sha256=source_hash, old_log_probability_max_error=log_error, old_value_max_error=value_error,
        nonzero_gradient_groups=gradients, optimizer_layout=scope.layout(), exact_serialized_resume=True,
        frozen_predictor_exact=True, outside_route_and_fine_gates_exact=True,
        rejected_checkpoint_cases=rejected, ordinary_training_admission_changed=False,
        scope='Numerical regression only; validation replay and synthetic gate fixtures. No TRAIN fitting, promotion or combat quality claim.')
    save(a.out/'report.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()

"""CUDA-only geometric step-heading fit; unchanged native action distribution."""
import argparse
import copy
import json
from pathlib import Path
from ppo_combat import torch, training_devices
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model
from combat_memory_route_movement import inputs
from train_combat_route_yaw_ridge import load_data, zeroed, identity, solve
from combat_route_yaw_auxiliary_data import admit
from ppo_recurrent import prepare_context, sequence
from combat_checkpoint_scope import require_checkpoint_scope
from evaluate_registered_combat_models import read, save, sha

SCOPE = 'combat_route_proportional_heading_ridge_v1'
GAIN = .25


def metrics(data, theta):
    yaw = torch.pi * (data['base'] + data['x'] @ theta).tanh()
    error = yaw - data['turn']
    return dict(heading_mae_degrees=float(torch.atan2(error.sin(), error.cos()).abs().mean()*180/torch.pi),
                step_target_mae_degrees=float((yaw-GAIN*data['turn']).abs().mean()*180/torch.pi),
                mean_absolute_turn_degrees=float(yaw.abs().mean()*180/torch.pi),
                maximum_absolute_turn_degrees=float(yaw.abs().max()*180/torch.pi))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    assert not a.out.exists()
    training_devices()
    torch.set_default_device('cuda')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    model = read(a.model)
    fixed = zeroed(model)
    report = read(a.data)
    assert report['version'] == 'combat_observed_static_route_heading_v1' and report['state'] == 'passed'
    assert not report['ppo_eligible'] and not report['executed_demonstration']
    for p, h in report['inputs'].items():
        assert sha(p) == h
    sources = [p for p in report['inputs'] if Path(p).name == 'weights.json']
    assert len(sources) == 1 and zeroed(read(sources[0])) == fixed
    actor = branch(base_model(fixed, allow_shared=True), 'actor').requires_grad_(False)
    train, diagnostic = (load_data(report, split, actor, admit_function=admit) for split in ('train', 'diagnostic'))
    assert not ({tuple(w) for w in train['worlds']} & {tuple(w) for w in diagnostic['worlds']})
    head = model['memory_route_movement']['layers'][-1]
    initial = torch.tensor([*head['weight'][2], head['bias'][2]], device='cuda', dtype=torch.float64)
    # A gradual target is supervision only: no native yaw cap or deterministic
    # route follower is introduced. Sampled actions retain their full support.
    target = torch.atanh(GAIN*train['turn']/torch.pi)-train['base']
    gram, rhs = train['x'].T@train['x'], train['x'].T@target
    theta = solve(gram, rhs)
    assert theta.is_cuda and bool(torch.isfinite(theta).all())
    n = len(train['x'])//2
    x1, x2, y1, y2 = train['x'][:n], train['x'][n:], target[:n], target[n:]
    torch.testing.assert_close(theta, solve(x1.T@x1+x2.T@x2, x1.T@y1+x2.T@y2), atol=1e-8, rtol=1e-8)
    before = {s: metrics(d, initial) for s, d in [('train', train), ('diagnostic', diagnostic)]}
    after = {s: metrics(d, theta) for s, d in [('train', train), ('diagnostic', diagnostic)]}
    result = copy.deepcopy(model)
    last = result['memory_route_movement']['layers'][-1]
    last['weight'][2] = theta[:16].float().cpu().tolist()
    last['bias'][2] = float(theta[16].float())
    assert zeroed(result) == fixed and identity(result) == identity(model)
    changed = branch(base_model(result, allow_shared=True), 'actor').requires_grad_(False)
    for e in report['episodes']:
        rows, history, _, _ = admit(Path(e['source_cuda']), e['source_model_sha256'], [])
        context = prepare_context(history, rows, 'cuda')
        with torch.no_grad():
            old, _ = sequence(actor, context)
            new, _ = sequence(changed, context)
            x = torch.tensor([r['features'] for r in rows], device='cuda')
            _, active = inputs(x, old)
            assert torch.equal(old[~active], new[~active])
            assert torch.equal(old[:, :2], new[:, :2]) and torch.equal(old[:, 3:], new[:, 3:])
    a.out.mkdir()
    save(a.out/'weights.json', result)
    cp = dict(version='combat_route_proportional_heading_checkpoint_v1', checkpoint_training_scope=SCOPE,
              weights_sha256=sha(a.out/'weights.json'), frozen_behavior_sha256=identity(model),
              gain=GAIN, ridge=.001, gram=gram, rhs=rhs, rows=len(train['x']),
              consumed=train['tokens'], worlds=train['worlds'])
    torch.save(cp, a.out/'checkpoint.pt')
    restored = torch.load(a.out/'checkpoint.pt', map_location='cuda', weights_only=True)
    require_checkpoint_scope(restored, SCOPE)
    assert torch.equal(solve(restored['gram'], restored['rhs']), theta)
    try:
        require_checkpoint_scope(restored)
    except ValueError:
        pass
    else:
        raise AssertionError('Auxiliary checkpoint entered generic PPO')
    assert all(sha(p) == h for p, h in report['inputs'].items())
    save(a.out/'report.json', dict(version=SCOPE, state='passed', device=torch.cuda.get_device_name(),
         training_performed=True, trainable_parameters=17, gain=GAIN, ridge=.001,
         rows=len(train['x']), diagnostic_rows=len(diagnostic['x']), before=before, after=after,
         inputs={str(a.model.resolve()): sha(a.model), str(a.data.resolve()): sha(a.data)},
         weights_sha256=sha(a.out/'weights.json'), checkpoint_sha256=sha(a.out/'checkpoint.pt'),
         frozen_other_fields_exact=True, inactive_and_other_raw_outputs_exact=True,
         checkpoint_readback_exact=True, append_resume_implemented=False, promotion=False,
         scope='Gradual static waypoint-heading auxiliary on actual restored-body observations. No executed expert, runtime yaw clamp, PPO or native quality claim.'))
    print(json.dumps(dict(before=before, after=after)), flush=True)


if __name__ == '__main__':
    main()

"""Damped natural-gradient optimization of the native clipped primitive surrogate.

The mean KL Fisher metric handles unequal raw action scales. This is not Adam
PPO or a paper TRPO reproduction: it keeps the existing clipped surrogate and
uses an explicit behavior-relative KL gate after each full-batch update.
"""
import argparse
import copy
import json
import math
import pathlib
import torch

from train_combat_hjepa import export, save, sha
from train_combat_hjepa_actor_ppo import Actor, load_rollouts, log_probability, distribution_kl, fingerprint

VERSION = 'hjepa_primitive_natural_surrogate_v1'
ALGORITHM = dict(max_updates=4, clip=.1, max_mean_kl=.02, damping=.01,
                 cg_iterations=40, cg_relative_tolerance=1e-10, backtrack_power=12,
                 min_gain_fraction=.1, credit='first_life_native_rtg_other_stream_time_baseline_v1',
                 optimizer='damped_fisher_cg_no_persistent_moments_v1')


def flatten(values):
    return torch.cat([value.reshape(-1) for value in values])


def assign(parameters, vector):
    if vector.numel() != sum(p.numel() for p in parameters):
        raise ValueError('parameter vector width')
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            width = parameter.numel()
            parameter.copy_(vector[offset:offset+width].reshape_as(parameter))
            offset += width


def conjugate_gradient(operator, rhs, iterations=40, relative_tolerance=1e-10):
    """SPD linear solve; detect invalid curvature rather than silently use it."""
    if not torch.isfinite(rhs).all():
        raise ValueError('nonfinite natural-gradient right-hand side')
    solution = torch.zeros_like(rhs)
    residual, direction = rhs.clone(), rhs.clone()
    squared = residual.dot(residual)
    initial = float(squared.sqrt())
    if initial == 0:
        return solution, dict(iterations=0, relative_residual=0.)
    completed = 0
    for index in range(iterations):
        product = operator(direction)
        curvature = direction.dot(product)
        if not torch.isfinite(curvature) or float(curvature) <= 0:
            raise ValueError('nonpositive/nonfinite damped Fisher curvature')
        alpha = squared/curvature
        solution += alpha*direction
        residual -= alpha*product
        next_squared = residual.dot(residual)
        completed = index+1
        if float(next_squared.sqrt()) <= max(initial*relative_tolerance, 1e-16):
            squared = next_squared
            break
        direction = residual + (next_squared/squared)*direction
        squared = next_squared
    if not torch.isfinite(solution).all():
        raise ValueError('nonfinite natural-gradient solution')
    return solution, dict(iterations=completed, relative_residual=float(squared.sqrt())/initial)


def fisher_operator(model, features, parameters, damping):
    """Exact Hessian-vector products of local KL(current || proposed)."""
    raw = model(features)
    kl = distribution_kl(raw.detach(), model.log_std.detach(), raw, model.log_std, model.temperature).mean()
    gradient = flatten(torch.autograd.grad(kl, parameters, create_graph=True))

    def apply(vector):
        hessian = torch.autograd.grad(gradient.dot(vector), parameters, retain_graph=True)
        return flatten(hessian).detach() + damping*vector

    return apply


def delta_metrics(before, old_std, after, new_std, temperature):
    delta = after[:, :4]-before[:, :4]
    axis_kl = (new_std-old_std + (old_std.mul(2).exp()+delta.square())
               / (2*new_std.mul(2).exp())-.5).mean(0)
    return dict(delta_in_behavior_sigma_rms=(delta/old_std.exp()).square().mean(0).sqrt().tolist(),
                normal_kl_by_axis=axis_kl.tolist(),
                total_mean_kl=float(distribution_kl(before, old_std, after, new_std, temperature).mean()))


def main(*, version=VERSION, algorithm=None, direction_solver=None, scope=None):
    algorithm = ALGORITHM if algorithm is None else algorithm
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=pathlib.Path, action='append', required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    ap.add_argument('--updates', type=int, default=4)
    ap.add_argument('--resume', type=pathlib.Path)
    a = ap.parse_args()
    if not torch.cuda.is_available() or not 1 <= a.updates <= algorithm['max_updates']:
        raise ValueError('CUDA and 1..4 total updates required')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    rows, parent, binding, proof = load_rollouts(a.data)
    model = Actor(parent).cuda()
    parameters = list(model.parameters())
    tensor = lambda values, dtype=torch.float64: torch.tensor(values, dtype=dtype, device='cuda')
    features = tensor([r['features'] for r in rows])
    latent = tensor([r['sample']['latent'] for r in rows])
    attack = tensor([r['sample']['attack'] for r in rows], torch.bool)
    vertical = tensor([r['sample']['vertical'] for r in rows], torch.long)
    old_raw, old_lp = tensor([r['actor_raw'] for r in rows]), tensor([r['primitive_log_probability'] for r in rows])
    old_std = model.log_std.detach().clone()
    advantage = tensor([r['advantage'] for r in rows])
    with torch.no_grad():
        raw_error = float((model(features)-old_raw).abs().max())
        lp_error = float((log_probability(model(features), old_std, latent, attack, vertical, model.temperature)-old_lp).abs().max())
    if raw_error > 1e-8 or lp_error > 1e-7:
        raise ValueError('initial native/CUDA likelihood mismatch')
    history = []
    if a.resume:
        ck = torch.load(a.resume, map_location='cuda', weights_only=False)
        if (ck['version'] != version or ck['algorithm'] != algorithm or ck['binding'] != binding
                or ck['parent_fingerprint'] != fingerprint(parent) or ck['stopped']):
            raise ValueError('resume source/algorithm mismatch')
        model.load_state_dict(ck['state']); history = ck['history']
        if not len(history) < a.updates:
            raise ValueError('resume must advance same batch')

    def metrics():
        raw = model(features)
        lp = log_probability(raw, model.log_std, latent, attack, vertical, model.temperature)
        ratio = (lp-old_lp).exp()
        surrogate = torch.minimum(ratio*advantage, ratio.clamp(.9, 1.1)*advantage).mean()
        kl = distribution_kl(old_raw, old_std, raw, model.log_std, model.temperature)
        return surrogate, kl.mean(), kl.max(), ((ratio-1).abs() > .1).double().mean()

    stopped, stop_reason = False, None
    for update in range(len(history), a.updates):
        before = flatten(parameters).detach().clone()
        surrogate, _, _, _ = metrics()
        prior = float(surrogate.detach())
        gradient = flatten(torch.autograd.grad(surrogate, parameters)).detach()
        operator = fisher_operator(model, features, parameters, algorithm['damping'])
        if direction_solver is None:
            direction, cg = conjugate_gradient(operator, gradient, algorithm['cg_iterations'], algorithm['cg_relative_tolerance'])
        else:
            direction, cg = direction_solver(operator, gradient, model, features, algorithm)
        product = operator(direction)
        quadratic = float(direction.dot(product))
        if direction_solver is not None:
            cg['verified_relative_residual'] = float((product-gradient).norm()/gradient.norm().clamp_min(1e-30))
        expected = float(gradient.dot(direction))
        if quadratic <= 0 or expected <= 0 or not math.isfinite(quadratic+expected):
            stopped, stop_reason = True, 'no_positive_finite_direction'
            break
        full_step = direction*math.sqrt(2*algorithm['max_mean_kl']/quadratic)
        predicted_gain = float(gradient.dot(full_step))
        # Release the retained local-Hessian graph before changing parameters.
        del operator
        accepted = False
        with torch.no_grad():
            for power in range(algorithm['backtrack_power']+1):
                scale = 2.**(-power)
                assign(parameters, before+scale*full_step)
                model.log_std.clamp_(-10, 1)
                values = [float(v) for v in metrics()]
                required_gain = max(1e-12, algorithm['min_gain_fraction']*scale*predicted_gain)
                if (all(math.isfinite(v) for v in values) and values[1] <= algorithm['max_mean_kl']
                        and values[0]-prior >= required_gain):
                    accepted = True
                    break
        if not accepted:
            assign(parameters, before)
            stopped, stop_reason = True, 'behavior_kl_or_surrogate_line_search_rejected'
            break
        history.append(dict(update=update+1, surrogate=values[0], mean_kl=values[1], max_kl=values[2],
                            clipped_fraction=values[3], step_scale=scale, gradient_norm=float(gradient.norm()),
                            cg=cg, damped_direction_quadratic=quadratic, predicted_gain=scale*predicted_gain,
                            actual_gain=values[0]-prior))
        print(json.dumps(history[-1]), flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    candidate = copy.deepcopy(parent)
    candidate['levels'][0]['encoder'], candidate['actor'] = export(model.encoder), export(model.actor)
    candidate['primitive_sampling']['log_std'] = model.log_std.detach().cpu().tolist()
    source_meta = json.loads(pathlib.Path(str(a.data[0])+'.meta.json').read_text())
    candidate['primitive_sampling']['seed'] = json.loads(pathlib.Path(source_meta['model']).read_text())['primitive_sampling']['seed']
    save(a.out/'model.json', candidate)
    deterministic = copy.deepcopy(candidate); deterministic.pop('primitive_sampling')
    save(a.out/'deterministic.json', deterministic)
    with torch.no_grad():
        raw = model(features)
        delta = delta_metrics(old_raw, old_std, raw, model.log_std, model.temperature)
        save(a.out/'parity.json', dict(features=features[:22].cpu().tolist(), outputs=raw[:22].cpu().tolist()))
    torch.save(dict(version=version, algorithm=algorithm, binding=binding, parent_fingerprint=fingerprint(parent),
                    state=model.state_dict(), history=history, stopped=stopped, stop_reason=stop_reason,
                    model_sha256=sha(a.out/'model.json')), a.out/'checkpoint.pt')
    report = dict(version=version, algorithm=algorithm, admission=proof, initial_parity=dict(raw=raw_error, log_probability=lp_error),
                  history=history, stopped=stopped, stop_reason=stop_reason, policy_delta=delta,
                  model_sha256=sha(a.out/'model.json'), binding=binding,
                  scope=scope or 'Natural-gradient clipped-surrogate pilot; all encoder0/actor/log_std parameters train. Weapon and auxiliary world heads frozen. Training change only; no combat quality or promotion claim.')
    save(a.out/'report.json', report)
    print(json.dumps(dict(admitted=proof['admitted'], policy_delta=delta, stopped=stopped, stop_reason=stop_reason), indent=2))


if __name__ == '__main__':
    main()

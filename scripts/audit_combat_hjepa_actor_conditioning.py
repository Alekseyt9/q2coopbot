"""Read-only comparison of CG directions on the same native TRAIN Fisher system."""
import argparse
import json
import math
import pathlib
import torch
from train_combat_hjepa import save
from train_combat_hjepa_actor_ppo import Actor,load_rollouts,log_probability
from train_combat_hjepa_actor_natural import flatten,fisher_operator,conjugate_gradient
from hjepa_fisher_diagonal import fisher_diagonal,preconditioned_cg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=pathlib.Path,action='append',required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    a = ap.parse_args()
    if not torch.cuda.is_available():raise ValueError('CUDA required')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    rows,parent,binding,admission = load_rollouts(a.data)
    model = Actor(parent).cuda();parameters = list(model.parameters())
    original = flatten(parameters).detach().clone()
    tensor = lambda v,dtype=torch.float64:torch.tensor(v,dtype=dtype,device='cuda')
    x = tensor([r['features'] for r in rows]);latent = tensor([r['sample']['latent'] for r in rows])
    attack = tensor([r['sample']['attack'] for r in rows],torch.bool);pose = tensor([r['sample']['vertical'] for r in rows],torch.long)
    advantage = tensor([r['advantage'] for r in rows]);old_lp = tensor([r['primitive_log_probability'] for r in rows])
    lp = log_probability(model(x),model.log_std,latent,attack,pose,model.temperature,model.fire_temperature)
    ratio = (lp-old_lp).exp();surrogate = torch.minimum(ratio*advantage,ratio.clamp(.9,1.1)*advantage).mean()
    gradient = flatten(torch.autograd.grad(surrogate,parameters)).detach()
    operator = fisher_operator(model,x,parameters,.01)
    diagonal = fisher_diagonal(model,x)+.01
    plain,plain_proof = conjugate_gradient(operator,gradient,40,1e-10)
    conditioned,conditioned_proof = preconditioned_cg(operator,gradient,diagonal,40,1e-10)
    results = {}
    for name,direction,proof in [('cg',plain,plain_proof),('diagonal_pcg',conditioned,conditioned_proof)]:
        product = operator(direction);quadratic = float(direction.dot(product));gain = float(gradient.dot(direction))
        if quadratic<=0 or not math.isfinite(quadratic+gain):raise ValueError('invalid diagnostic direction')
        scaled = direction*math.sqrt(.04/quadratic)
        stats = dict(proof=proof,verified_relative_residual=float((product-gradient).norm()/gradient.norm()),
                     quadratic_objective=gain-.5*quadratic,damped_quadratic=quadratic,
                     linear_gain_at_damped_kl02=float(gradient.dot(scaled)),direction_norm=float(direction.norm()))
        offset = 0
        for parameter_name,p in model.named_parameters():
            part = scaled[offset:offset+p.numel()].reshape_as(p);offset += p.numel()
            if parameter_name=='actor.0.weight':
                stats['scaled_final_actor_row_norms'] = part.norm(dim=1).tolist()
        results[name] = stats
    unchanged = torch.equal(original,flatten(parameters).detach())
    if not unchanged:raise ValueError('read-only audit changed weights')
    report = dict(version='hjepa_native_fisher_conditioning_audit_v1',admission=admission,binding=binding,
        gradient_norm=float(gradient.norm()),diagonal_min=float(diagonal.min()),diagonal_max=float(diagonal.max()),
        results=results,weights_unchanged=unchanged,
        scope='Same source TRAIN batch, same complete primitive clipped-surrogate score and damped Fisher system. 40CG versus40Jacobi-PCG iterations; exact local Fisher diagonal analytically computed and separately tested against full autograd Hessians. Parameter directions and quadratic objectives only, not fitted models, nonlinear trust-region acceptance, runtime actions, or native combat quality. Active categorical curriculum remains unchanged.')
    save(a.out,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['admission','binding']},indent=2))


if __name__=='__main__':main()

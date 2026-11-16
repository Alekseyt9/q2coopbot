"""Exact local primitive Fisher diagonal for Linear/ReLU actors; PCG prototype.

Independent numerical helper. The running natural-surrogate v1 trainer does
not import it. No update, altered loss, sampling rule or game command here.
"""
import torch
from torch import nn


@torch.no_grad()
def fisher_diagonal(model, features):
    sequence = list(model.encoder)+list(model.actor)
    if not sequence or not all(isinstance(m,(nn.Linear,nn.ReLU)) for m in sequence):
        raise ValueError('Linear/ReLU shared actor required')
    if not isinstance(sequence[0],nn.Linear) or not isinstance(sequence[-1],nn.Linear):
        raise ValueError('linear network endpoints required')
    if features.ndim != 2 or len(features)==0:
        raise ValueError('nonempty observation matrix required')
    inputs, raw = [], features
    for module in sequence:
        inputs.append(raw);raw = module(raw)
    if raw.shape[1] != 8 or model.log_std.numel() != 4 or model.temperature <= 0:
        raise ValueError('primitive distribution shape')
    count = len(features)
    metric = raw.new_zeros((count,8,8))
    axis = torch.arange(4,device=raw.device)
    metric[:,axis,axis] = model.log_std.mul(-2).exp()
    fire = (raw[:,4]/model.temperature).sigmoid()
    metric[:,4,4] = fire*(1-fire)/(model.temperature**2)
    pose = (raw[:,5:]/model.temperature).softmax(-1)
    metric[:,5:,5:] = (torch.diag_embed(pose)-pose[:,:,None]*pose[:,None,:])/(model.temperature**2)
    jacobian = torch.eye(8,dtype=raw.dtype,device=raw.device).expand(count,-1,-1)
    diagonal = {id(model.log_std):torch.full_like(model.log_std,2.)}
    for index in range(len(sequence)-1,-1,-1):
        module = sequence[index]
        if isinstance(module,nn.ReLU):
            jacobian = jacobian*(inputs[index]>0)[:,None,:]
            continue
        # For W[k,j], d(raw)/dW is J[:,:,k]*input[:,j]. The complete
        # 8x8 output metric retains pose class correlations, unlike a
        # diagonal-curvature backprop approximation.
        curvature = (jacobian*(metric@jacobian)).sum(1).clamp_min(0)
        diagonal[id(module.weight)] = curvature.T@inputs[index].square()/count
        if module.bias is not None:
            diagonal[id(module.bias)] = curvature.mean(0)
        if index:
            jacobian = jacobian@module.weight
    if set(diagonal) != {id(p) for p in model.parameters()}:
        raise ValueError('unhandled or shared parameters')
    result = torch.cat([diagonal[id(p)].reshape(-1) for p in model.parameters()])
    if not torch.isfinite(result).all() or (result<0).any():
        raise ValueError('invalid local Fisher diagonal')
    return result


def preconditioned_cg(operator, rhs, diagonal, iterations=40, relative_tolerance=1e-10):
    if rhs.shape != diagonal.shape or not torch.isfinite(diagonal).all() or (diagonal<=0).any():
        raise ValueError('finite positive preconditioner required')
    if not torch.isfinite(rhs).all():
        raise ValueError('finite score gradient required')
    result = torch.zeros_like(rhs);residual = rhs.clone()
    initial = float(residual.norm())
    if initial == 0:
        return result,dict(iterations=0,relative_residual=0.)
    z = residual/diagonal;direction = z.clone();rz = residual.dot(z)
    completed = 0
    for index in range(iterations):
        product = operator(direction);curvature = direction.dot(product)
        if not torch.isfinite(curvature) or float(curvature)<=0:
            raise ValueError('nonpositive/nonfinite damped Fisher curvature')
        alpha = rz/curvature
        result += alpha*direction;residual -= alpha*product;completed = index+1
        if float(residual.norm()) <= max(initial*relative_tolerance,1e-16):break
        z = residual/diagonal;next_rz = residual.dot(z)
        direction = z+(next_rz/rz)*direction;rz = next_rz
    if not torch.isfinite(result).all():raise ValueError('nonfinite PCG direction')
    return result,dict(iterations=completed,relative_residual=float(residual.norm())/initial)

"""Small convex logistic diagnostics solved entirely on CUDA."""
from ppo_combat import torch


def fit(offset,design,target,weights,ridge,max_steps=60):
    assert all(v.is_cuda and v.dtype==torch.float64 for v in (offset,design,target,weights))
    assert len(design)==len(offset)==len(target)==len(weights) and bool((weights>0).all())
    assert bool((target==0).any()) and bool((target==1).any()) and ridge>=0
    weights=weights/weights.sum()
    theta=torch.zeros(design.shape[1],device='cuda',dtype=torch.float64)
    penalty=torch.full_like(theta,ridge);penalty[-1]=0
    diagonal=torch.diag(penalty)
    jitter=torch.eye(len(theta),device='cuda',dtype=torch.float64)*1e-10
    def objective(value):
        logits=offset+design@value
        return (weights*torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')).sum()+.5*(penalty*value.square()).sum()
    for iteration in range(max_steps):
        probability=(offset+design@theta).sigmoid()
        gradient=design.T@(weights*(probability-target))+penalty*theta
        if float(gradient.abs().max())<1e-8:break
        curvature=weights*probability*(1-probability)
        hessian=design.T@(design*curvature[:,None])+diagonal+jitter
        direction=torch.linalg.solve(hessian,gradient)
        decrement=gradient@direction;current=objective(theta)
        assert bool(torch.isfinite(decrement)) and float(decrement)>0
        accepted=False
        for power in range(32):
            scale=2.**-power;candidate=theta-scale*direction;loss=objective(candidate)
            if bool(torch.isfinite(loss)) and float(loss)<=float(current-1e-4*scale*decrement)+1e-14:
                theta=candidate;accepted=True;break
        assert accepted,'Convex diagnostic line search failed'
    gradient=design.T@(weights*((offset+design@theta).sigmoid()-target))+penalty*theta
    maximum=float(gradient.abs().max())
    assert maximum<1e-6,('Logistic diagnostic did not converge',maximum)
    return theta,dict(iterations=iteration+1,gradient_maximum=maximum,objective=float(objective(theta)))


def normalizer(values,weights):
    assert values.is_cuda and weights.is_cuda
    weights=weights/weights.sum()
    mean=(values*weights[:,None]).sum(0)
    std=((values-mean).square()*weights[:,None]).sum(0).sqrt().clamp_min(.05)
    return mean,std


def metrics(logits,target,weights):
    assert all(v.is_cuda for v in (logits,target,weights))
    loss=(weights*torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')).sum()/weights.sum()
    recalls=[float((weights[target==c]*((logits[target==c]>=0)==bool(c))).sum()/weights[target==c].sum()) for c in (0,1)]
    return dict(log_loss=float(loss),balanced_accuracy=sum(recalls)/2,
                choice0_recall=recalls[0],choice1_recall=recalls[1])

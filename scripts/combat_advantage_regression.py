"""Detached advantage weights for observed-action regression, CUDA only."""
import math


def weights(returns, baselines, temperature, cap, strength, torch):
    assert returns.is_cuda and baselines.is_cuda and returns.shape==baselines.shape
    assert math.isfinite(temperature) and temperature>0
    assert math.isfinite(cap) and cap>=1 and strength in (0.,1.)
    if strength==0:
        return torch.ones_like(returns),dict(strength=0.,mean=1.,maximum=1.,effective_rows=len(returns))
    with torch.no_grad():
        advantage=returns-baselines
        raw=(advantage/temperature).clamp(max=math.log(cap)).exp()
        assert torch.isfinite(raw).all() and float(raw.mean())>0
        result=raw/raw.mean()
        stats=dict(strength=strength,temperature=temperature,raw_cap=cap,
                   advantage_mean=float(advantage.mean()),advantage_std=float(advantage.std(unbiased=False)),
                   mean=float(result.mean()),minimum=float(result.min()),maximum=float(result.max()),
                   clipped_rows=int((advantage/temperature>=math.log(cap)).sum()),
                   effective_rows=float(result.sum().square()/result.square().sum()))
    return result,stats

"""Independent analytic CUDA likelihood and masked-gradient checks."""
import argparse, json, math, pathlib
from ppo_combat import torch, log_prob, anchor_kl
from combat_weapon_head import feature_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=pathlib.Path, required=True)
    args = ap.parse_args()
    assert torch.cuda.is_available(), 'CUDA required'
    x = torch.zeros((2,845),device='cuda')
    x[:,833] = 1
    x[:,837] = 1
    zero = torch.zeros(2,device='cuda')
    category = torch.zeros(2,dtype=torch.long,device='cuda')
    z = torch.zeros((2,4),device='cuda')
    std = torch.zeros(4,device='cuda')
    results = {}
    for width in (8,20):
        raw = torch.zeros((2,width),device='cuda',requires_grad=True)
        weapon = category if width==20 else None
        lp, entropy = log_prob(lambda _:raw,std,x,z,zero,category,weapon)
        expected = -2*math.log(2*math.pi)-math.log(2)-math.log(3)
        expected_entropy = 2*math.log(2*math.pi*math.e)+math.log(2)+math.log(3)
        if width==20:
            expected-=math.log(2)
            expected_entropy+=math.log(2)
        assert torch.allclose(lp,torch.full_like(lp,expected),atol=1e-6)
        assert torch.allclose(entropy,torch.full_like(entropy,expected_entropy),atol=1e-6)
        (-lp.mean()).backward()
        assert torch.isfinite(raw.grad).all()
        if width==20:
            unavailable = ~feature_mask(x)
            assert torch.equal(raw.grad[:,8:20][unavailable],torch.zeros_like(raw.grad[:,8:20][unavailable]))
            assert raw.grad[:,8].abs().sum()>0 and raw.grad[:,12].abs().sum()>0
            huge=raw.detach().clone();huge[:,9]=10000
            masked_lp,_=log_prob(lambda _:huge,std,x,z,zero,category,weapon)
            assert torch.equal(masked_lp,lp.detach()),'Unavailable weapon changed likelihood'
            bad=torch.ones_like(category)
            try:log_prob(lambda _:raw,std,x,z,zero,category,bad)
            except ValueError:pass
            else:raise AssertionError('Unavailable weapon accepted')
            assert float(anchor_kl(raw.detach(),std,raw.detach(),std,weapon_mask=feature_mask(x)))<1e-7
        results[str(width)]=dict(expected_log_probability=expected,expected_entropy=expected_entropy,
                                max_error=float((lp.detach()-expected).abs().max()))
    args.out.write_text(json.dumps(dict(version='combat_mlp_weapon_cuda_analytic_v1',
        device=torch.cuda.get_device_name(),results=results,masked_gradients_zero=True,
        masked_logit_invariant=True,scope='Analytic CUDA checks, not native experience or gameplay acceptance'),indent=2))
    print(json.dumps(results))


if __name__=='__main__':
    main()

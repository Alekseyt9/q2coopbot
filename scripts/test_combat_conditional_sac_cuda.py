"""Small analytic CUDA checks, not CPU NN tests or game quality evidence."""
import math
import unittest
from combat_conditional_sac import torch, soft_backup, squashed_log_probability


class ConditionalSAC(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        torch.set_default_device('cuda')

    def test_terminal_and_clipped_double_q(self):
        result = soft_backup(torch.tensor([1.,2.]),torch.tensor([True,False]),
            torch.tensor([2.,1000.]),torch.tensor([3.,2000.]),torch.tensor([-.5,-100.]),.9,.2)
        torch.testing.assert_close(result,torch.tensor([2.89,2.]),rtol=1e-6,atol=1e-6)

    def test_tanh_density_and_saturated_gradient(self):
        z=torch.tensor([[0.,0.],[1.,-.5],[20.,-20.]],requires_grad=True)
        normal=torch.distributions.Normal(torch.zeros_like(z),torch.ones_like(z))
        lp=squashed_log_probability(normal,z)
        self.assertAlmostEqual(float(lp[0].detach()),-math.log(2*math.pi),places=5)
        transform=torch.distributions.transforms.TanhTransform(cache_size=0)
        expected=(normal.log_prob(z)-transform.log_abs_det_jacobian(z,z.tanh())).sum(-1)
        torch.testing.assert_close(lp[:2],expected[:2],rtol=1e-6,atol=1e-6)
        self.assertTrue(bool(torch.isfinite(lp).all()))
        lp.sum().backward();self.assertTrue(bool(torch.isfinite(z.grad).all()))


if __name__=='__main__':unittest.main()

"""Small analytic CUDA checks, not CPU NN tests or game quality evidence."""
import math
import unittest
from combat_conditional_sac import torch, soft_backup, squashed_log_probability, behavior_aim_kl


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

    def test_behavior_kl_gradient_and_reference_stop(self):
        teacher=torch.nn.Sequential(torch.nn.Linear(125,2,device='cuda'))
        current=torch.nn.Sequential(torch.nn.Linear(125,2,device='cuda'))
        with torch.no_grad():
            teacher[0].weight.zero_();teacher[0].bias.zero_()
            current[0].weight.zero_();current[0].bias.copy_(torch.tensor([.3,-.2]))
        raw=torch.zeros(1,81);features=torch.zeros(1,1147)
        condition=torch.tensor([[1,1,0,0,1]],dtype=torch.long);movement=torch.zeros(1,2)
        std=torch.zeros(4,requires_grad=True);reference_std=torch.zeros(4,requires_grad=True)
        kl=behavior_aim_kl(raw,features,condition,movement,current,std,teacher,reference_std).mean()
        self.assertAlmostEqual(float(kl.detach()),.065,places=5)
        kl.backward()
        torch.testing.assert_close(current[0].bias.grad,torch.tensor([.3,-.2]),rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(std.grad,torch.zeros(4),rtol=0,atol=1e-6)
        self.assertIsNone(teacher[0].bias.grad);self.assertIsNone(reference_std.grad)


if __name__=='__main__':unittest.main()

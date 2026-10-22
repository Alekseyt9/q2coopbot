"""CUDA weighting and gradient checks; no environment or CPU NN fallback."""
import unittest
import torch
from combat_advantage_regression import weights


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert torch.cuda.is_available()
        torch.set_default_device('cuda')

    def test_uniform_is_exact_and_rng_neutral(self):
        before=torch.cuda.get_rng_state();values=torch.tensor([-2.,0.,4.])
        result,_=weights(values,torch.zeros_like(values),3,20,0,torch)
        self.assertTrue(torch.equal(result,torch.ones_like(values)))
        self.assertTrue(torch.equal(before,torch.cuda.get_rng_state()))

    def test_good_actions_receive_more_detached_gradient(self):
        returns=torch.tensor([-3.,0.,3.],requires_grad=True);baseline=torch.zeros(3,requires_grad=True)
        result,stats=weights(returns,baseline,3,20,1,torch)
        self.assertFalse(result.requires_grad);self.assertAlmostEqual(float(result.mean()),1,places=6)
        likelihood=torch.nn.Parameter(torch.zeros(3));(-(result*likelihood).mean()).backward()
        self.assertGreater(abs(float(likelihood.grad[2])),abs(float(likelihood.grad[0])))
        self.assertIsNone(returns.grad);self.assertIsNone(baseline.grad)
        self.assertGreater(stats['effective_rows'],1)

    def test_cap_and_common_baseline_shift(self):
        values=torch.tensor([-2.,1.,100.]);zero=torch.zeros(3)
        first,stats=weights(values,zero,3,20,1,torch)
        second,_=weights(values+5,zero+5,3,20,1,torch)
        self.assertTrue(torch.equal(first,second));self.assertEqual(stats['clipped_rows'],1)
        self.assertLessEqual(stats['effective_rows'],3)


if __name__=='__main__':unittest.main()

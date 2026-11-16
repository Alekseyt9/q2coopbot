"""Independent distribution formulas and credit boundary checks."""
import math
import unittest
import torch
from torch.distributions import Normal, Bernoulli, Categorical, kl_divergence
from train_combat_hjepa_actor_ppo import network, log_probability, distribution_kl, eligible


class PrimitivePPO(unittest.TestCase):
    def test_float64_import_does_not_round_through_float32(self):
        value = 1.000000000000003
        net = network([dict(weight=[[value]], bias=[-value])])
        self.assertEqual(net[0].weight.dtype, torch.float64)
        self.assertEqual(net[0].weight.item(), value)
        self.assertEqual(net[0].bias.item(), -value)

    def test_joint_latent_likelihood_matches_torch_distributions(self):
        raw = torch.tensor([[.1, -.2, .03, -.04, -2., 1., -3., 2.]], dtype=torch.float64)
        std = torch.tensor([-.5, -1., -3., -4.], dtype=torch.float64)
        latent = torch.tensor([[.2, .1, -.01, .03]], dtype=torch.float64)
        for fire in (False, True):
            for pose in range(3):
                actual = log_probability(raw, std, latent, torch.tensor([fire]), torch.tensor([pose]), .25)
                expected = Normal(raw[:, :4], std.exp()).log_prob(latent).sum(-1)
                expected += Bernoulli(logits=raw[:, 4]/.25).log_prob(torch.tensor([float(fire)], dtype=torch.float64))
                expected += Categorical(logits=raw[:, 5:]/.25).log_prob(torch.tensor([pose]))
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_analytic_kl_matches_independent_factor_kl(self):
        old = torch.tensor([[.1, .3, -.1, .2, -2., 3., -1., 0.]], dtype=torch.float64)
        new = old + torch.tensor([[.3, -.2, .01, -.03, 1., -.3, .8, .2]], dtype=torch.float64)
        s0, s1 = torch.tensor([-.2, -.5, -3., -4.], dtype=torch.float64), torch.tensor([-.3, -.1, -3.2, -3.8], dtype=torch.float64)
        expected = kl_divergence(Normal(old[:, :4], s0.exp()), Normal(new[:, :4], s1.exp())).sum(-1)
        expected += kl_divergence(Bernoulli(logits=old[:, 4]/.25), Bernoulli(logits=new[:, 4]/.25))
        expected += kl_divergence(Categorical(logits=old[:, 5:]/.25), Categorical(logits=new[:, 5:]/.25))
        torch.testing.assert_close(distribution_kl(old, s0, new, s1, .25), expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(distribution_kl(old, s0, old, s0, .25), torch.zeros(1, dtype=torch.float64), atol=1e-12, rtol=0)

    def test_saturated_fire_and_raw_latent_remain_finite(self):
        raw = torch.tensor([[0., 0., 0., 0., 1000., 1000., -1000., 0.]], dtype=torch.float64)
        lp = log_probability(raw, torch.zeros(4, dtype=torch.float64), torch.full((1, 4), 30., dtype=torch.float64),
                             torch.tensor([False]), torch.tensor([1]), .25)
        self.assertTrue(math.isfinite(lp.item()))
        self.assertLess(lp.item(), -10000)

    def test_no_post_ack_credit_even_on_same_frame(self):
        life = dict(start_frame=10, end_frame=20)
        ledger = dict(terminal_step=5)
        row = dict(exclusive=True, identity=dict(frame=20), step_index=5)
        self.assertTrue(eligible(row, ledger, life))
        row['step_index'] = 6
        self.assertFalse(eligible(row, ledger, life))
        ledger['terminal_step'] = None
        self.assertTrue(eligible(row, ledger, life))
        row['identity']['frame'] = 21
        self.assertFalse(eligible(row, ledger, life))
        row['identity']['frame'] = 10; row['exclusive'] = False
        self.assertFalse(eligible(row, ledger, life))


if __name__ == '__main__':
    unittest.main()

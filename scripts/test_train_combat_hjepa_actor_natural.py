"""Independent linear algebra and Fisher geometry checks, no game assets."""
import unittest
import torch
from torch import nn
from train_combat_hjepa_actor_natural import conjugate_gradient, fisher_operator, assign, flatten


class ToyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(8, dtype=torch.float64))
        self.log_std = nn.Parameter(torch.tensor([.15, .15, .003, .003], dtype=torch.float64).log())
        self.temperature = .25

    def forward(self, x):
        return self.mean.expand(len(x), -1)


class NaturalGradient(unittest.TestCase):
    def test_cg_matches_direct_spd_solution(self):
        matrix = torch.tensor([[5., 1., .2], [1., 3., .7], [.2, .7, 2.]], dtype=torch.float64)
        rhs = torch.tensor([.3, -2., 1.], dtype=torch.float64)
        result, proof = conjugate_gradient(lambda x: matrix@x, rhs)
        torch.testing.assert_close(result, torch.linalg.solve(matrix, rhs), atol=1e-12, rtol=1e-12)
        self.assertLess(proof['relative_residual'], 1e-10)

    def test_cg_rejects_nonpositive_curvature(self):
        with self.assertRaisesRegex(ValueError, 'curvature'):
            conjugate_gradient(lambda x: -x, torch.ones(3, dtype=torch.float64))

    def test_fisher_normal_blocks_equal_analytic_metric(self):
        model = ToyActor(); parameters = list(model.parameters())
        operator = fisher_operator(model, torch.zeros(3, 1, dtype=torch.float64), parameters, .01)
        for axis in range(4):
            vector = torch.zeros(12, dtype=torch.float64);vector[axis] = 1.
            actual = operator(vector)
            self.assertAlmostEqual(actual[axis].item(), 1/model.log_std[axis].exp().square().item()+.01, places=7)
            expected = torch.zeros_like(actual);expected[axis] = actual[axis]
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
        for axis in range(8,12):
            vector = torch.zeros(12, dtype=torch.float64);vector[axis] = 1.
            self.assertAlmostEqual(operator(vector)[axis].item(), 2.01, places=12)

    def test_natural_step_balances_movement_and_aim_sigma(self):
        # Equal standardized mean-score signals have raw gradients 1/sigma.
        # The exact Gaussian Fisher solve must produce a mean step proportional
        # to sigma, despite the 50x difference in movement and aim units.
        sigma = torch.tensor([.15, .15, .003, .003], dtype=torch.float64)
        result, _ = conjugate_gradient(lambda x: x/sigma.square(), 1/sigma)
        torch.testing.assert_close(result/sigma, torch.ones(4, dtype=torch.float64), atol=1e-10, rtol=1e-10)

    def test_parameter_assign_roundtrip_and_width_rejection(self):
        model = ToyActor();params = list(model.parameters());original = flatten(params).detach().clone()
        assign(params, original+.1)
        torch.testing.assert_close(flatten(params), original+.1, atol=0, rtol=0)
        assign(params, original)
        torch.testing.assert_close(flatten(params), original, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'width'):
            assign(params, original[:-1])


if __name__ == '__main__':
    unittest.main()

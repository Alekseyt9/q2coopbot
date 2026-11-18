"""Compare analytic Fisher diagonals to independently differentiated Hessians."""
import unittest
import torch
from torch import nn
from hjepa_fisher_diagonal import fisher_diagonal,preconditioned_cg
from train_combat_hjepa_actor_natural import fisher_operator


class SmallActor(nn.Module):
    def __init__(self,temperature):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3,4,dtype=torch.float64),nn.ReLU(),nn.Linear(4,4,dtype=torch.float64))
        self.actor = nn.Sequential(nn.Linear(4,8,dtype=torch.float64))
        self.log_std = nn.Parameter(torch.tensor([.15,.15,.003,.003],dtype=torch.float64).log())
        self.temperature = temperature

    def forward(self,x):return self.actor(self.encoder(x))


class FisherDiagonal(unittest.TestCase):
    def test_every_parameter_matches_full_autograd_hessian_diagonal(self):
        torch.manual_seed(615)
        x = torch.randn(5,3,dtype=torch.float64)
        for temperature,fire_temperature in [(.25,None),(1.,None),(.25,2.)]:
            model = SmallActor(temperature)
            if fire_temperature is not None:model.fire_temperature = fire_temperature
            diagonal = fisher_diagonal(model,x)
            operator = fisher_operator(model,x,list(model.parameters()),0.)
            reference = torch.empty_like(diagonal)
            for index in range(len(diagonal)):
                basis = torch.zeros_like(diagonal);basis[index] = 1.
                reference[index] = operator(basis)[index]
            torch.testing.assert_close(diagonal,reference,rtol=1e-10,atol=1e-9)

    def test_symmetric_correlated_metric_pcg_matches_dense_solve(self):
        # Six orders of raw-coordinate scaling plus off-diagonal coupling.
        scale = torch.tensor([.001,.01,.1,1.,10.,100.],dtype=torch.float64)
        correlation = torch.eye(6,dtype=torch.float64)+torch.ones(6,6,dtype=torch.float64)*.1
        matrix = scale[:,None]*correlation*scale[None,:]
        rhs = torch.tensor([1.,-.3,.2,-2.,3.,-.5],dtype=torch.float64)
        actual,proof = preconditioned_cg(lambda x:matrix@x,rhs,matrix.diag(),iterations=10)
        torch.testing.assert_close(actual,torch.linalg.solve(matrix,rhs),rtol=1e-10,atol=1e-9)
        self.assertLess(proof['relative_residual'],1e-10)

    def test_invalid_diagonal_and_curvature_rejected(self):
        ones = torch.ones(3,dtype=torch.float64)
        with self.assertRaisesRegex(ValueError,'positive'):
            preconditioned_cg(lambda x:x,ones,-ones)
        with self.assertRaisesRegex(ValueError,'curvature'):
            preconditioned_cg(lambda x:-x,ones,ones)

    def test_zero_gradient_has_zero_direction(self):
        zeros = torch.zeros(3,dtype=torch.float64)
        result,proof = preconditioned_cg(lambda x:x,zeros,torch.ones_like(zeros))
        torch.testing.assert_close(result,zeros,atol=0,rtol=0)
        self.assertEqual(proof['iterations'],0)


if __name__=='__main__':unittest.main()

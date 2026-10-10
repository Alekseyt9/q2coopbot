"""Numerical policy-KL tests execute only on CUDA."""
import math
import unittest
from combat_sequence_retention import policy_kl
from train_combat_bc import torch


class SequenceRetentionCUDA(unittest.TestCase):
    def setUp(self):
        assert torch.cuda.is_available(), 'CUDA required; no CPU numerical fallback'
        self.features = torch.zeros(1,854,device='cuda')
        self.features[:,833] = 1
        self.std = torch.zeros(4,device='cuda')

    def test_self_kl_and_gradient(self):
        self.features[:,426] = 1
        teacher = torch.randn(1,45,device='cuda')
        raw = teacher.clone().requires_grad_()
        loss = policy_kl(raw,self.std,teacher,self.std,self.features).sum()
        self.assertLess(abs(float(loss.detach())),1e-12)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))
        self.assertLess(float(raw.grad.abs().max()),1e-8)

    def test_unavailable_target_aim_and_weapon_do_not_contribute(self):
        teacher = torch.zeros(1,45,device='cuda')
        raw = teacher.clone()
        raw[:,21:45] = 100
        raw[:,9:20] = 100
        raw.requires_grad_()
        loss = policy_kl(raw,self.std,teacher,self.std,self.features).sum()
        self.assertLess(abs(float(loss.detach())),1e-12)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))
        self.assertLess(float(raw.grad.abs().max()),1e-8)

    def test_conditional_aim_uses_teacher_target_mass(self):
        self.features[:,426] = 1
        teacher = torch.zeros(1,45,device='cuda')
        raw = teacher.clone()
        raw[:,29] = .2
        result = policy_kl(raw,self.std,teacher,self.std,self.features)
        expected = torch.tensor([.5 * .5 * .2**2],device='cuda',dtype=torch.float64)
        self.assertLess(float((result-expected).abs().max()),1e-8)

    def test_target_category_kl(self):
        self.features[:,426] = 1
        teacher = torch.zeros(1,45,device='cuda',dtype=torch.float64)
        raw = teacher.clone()
        raw[:,20] = math.log(3)
        result = policy_kl(raw,self.std,teacher,self.std,self.features)
        expected = torch.tensor([.5 * math.log(4/3)],device='cuda',dtype=torch.float64)
        self.assertLess(float((result-expected).abs().max()),1e-12)

    def test_precision_mode_conditioned_gaussian(self):
        teacher = torch.zeros(1,81,device='cuda',dtype=torch.float64)
        raw = teacher.clone()
        raw[:,45] = .4
        result = policy_kl(raw,self.std,teacher,self.std,self.features)
        expected = torch.tensor([.5 * .5 * .4**2],device='cuda',dtype=torch.float64)
        self.assertLess(float((result-expected).abs().max()),1e-12)

    def test_log_std_change_has_finite_gradient(self):
        teacher = torch.zeros(1,45,device='cuda')
        std = torch.tensor([.1,-.1,.2,-.2],device='cuda',requires_grad=True)
        loss = policy_kl(teacher,std,teacher,self.std,self.features).sum()
        self.assertGreater(float(loss.detach()),0)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(std.grad).all()))
        self.assertGreater(float(std.grad.abs().max()),0)


if __name__ == '__main__':
    unittest.main()

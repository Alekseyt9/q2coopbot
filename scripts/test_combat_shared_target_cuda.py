import unittest
from ppo_combat import torch, training_devices
from combat_shared_target import initialize, inputs, apply
from combat_shared_target import VERSION
from combat_spatial_aim import SpatialActor
from combat_target_head_scope import freeze_decision_scope
from ppo_combat import nn, layers


class SharedTargetTests(unittest.TestCase):
    def setUp(self):
        training_devices()
        torch.manual_seed(123)
        self.x = torch.randn(5, 3, 1121, device='cuda') * .2
        self.raw = torch.randn(5, 3, 81, device='cuda')
        self.x[..., 73:169].reshape(5, 3, 8, 12)[..., 0] = 1
        self.x[..., 881:1121].reshape(5, 3, 8, 30)[..., 0] = 1

    def test_zero_migration_and_only_target_changes(self):
        branch = initialize()
        self.assertTrue(torch.equal(apply(branch, self.x, self.raw), self.raw))
        optimizer = torch.optim.Adam(branch.parameters(), lr=.003)
        old_first = branch[0].weight.detach().clone()
        # Distinct slot labels exercise sharing and input gradients after the
        # zero final layer receives its first update. This is a synthetic test,
        # not a combat-training or quality result.
        labels = torch.arange(8, device='cuda').expand(5, 3, 8)
        for _ in range(3):
            optimizer.zero_grad()
            logits = apply(branch, self.x, self.raw)[..., 21:29]
            ((logits - labels) ** 2).mean().backward()
            optimizer.step()
        current = apply(branch, self.x, self.raw)
        self.assertTrue(torch.equal(current[..., :21], self.raw[..., :21]))
        self.assertTrue(torch.equal(current[..., 29:], self.raw[..., 29:]))
        self.assertFalse(torch.equal(current[..., 21:29], self.raw[..., 21:29]))
        self.assertFalse(torch.equal(branch[0].weight, old_first))

    def test_enemy_and_memory_permutation_and_empty_summaries(self):
        permutation = torch.tensor([7, 2, 0, 6, 3, 1, 5, 4], device='cuda')
        shuffled = self.x.clone()
        for begin, end, width in ((73,169,12),(426,466,5),(466,786,40)):
            values = self.x[..., begin:end].reshape(5,3,8,width)
            shuffled[..., begin:end] = values[..., permutation, :].flatten(-2)
        shuffled[..., 846:854] = self.x[..., 846:854][..., permutation]
        remembered = self.x[..., 881:1121].reshape(5,3,8,30)
        shuffled[..., 881:1121] = remembered[..., permutation, :].flatten(-2)
        self.assertTrue(torch.allclose(inputs(shuffled,self.raw),
                                      inputs(self.x,self.raw)[...,permutation,:],
                                      atol=1e-6, rtol=1e-6))
        empty = self.x.clone()
        empty[..., 73:169] = 0
        empty[..., 881:1121] = 0
        result = inputs(empty,self.raw)
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertTrue(torch.equal(result[..., -72:],torch.zeros_like(result[..., -72:])))

    def test_integrated_scope_trains_branch_and_preserves_other_outputs(self):
        base=nn.Sequential(nn.Linear(1121,3,device='cuda'),nn.Tanh(),
                           nn.Linear(3,81,device='cuda'))
        spatial=nn.Sequential(nn.Linear(119,2,device='cuda'),nn.ReLU(),
                              nn.Linear(2,2,device='cuda'),nn.ReLU(),
                              nn.Linear(2,6,device='cuda'))
        spec=dict(version='combat_shared_spatial_coarse_fine_aim_v2',layers=layers(spatial))
        model=dict(feature_version='combat_features_v9',spatial_aim=spec,
                   aim_mode_head='combat_precision_aim_v1',
                   shared_target=dict(version=VERSION,layers=layers(initialize())))
        actor=SpatialActor(base,spec,model).to('cuda')
        std=nn.Parameter(torch.zeros(4,device='cuda'))
        optimizer=torch.optim.Adam(list(actor.parameters())+[std],lr=.003)
        initial=actor(self.x).detach().clone()
        frozen_spatial={name:p.detach().clone() for name,p in actor.branch.named_parameters()}
        project,count=freeze_decision_scope(actor,std,optimizer,'decisions')
        self.assertEqual(count,22*4+16193)
        for _ in range(3):
            optimizer.zero_grad()
            output=actor(self.x)
            (output[...,21:29]*torch.arange(8,device='cuda')).mean().backward()
            optimizer.step();project()
        selected=torch.zeros(81,dtype=torch.bool,device='cuda')
        selected[4]=True;selected[8:29]=True
        self.assertTrue(torch.equal(actor(self.x)[...,~selected],initial[...,~selected]))
        for name,p in actor.branch.named_parameters():
            self.assertTrue(torch.equal(p,frozen_spatial[name]))
        self.assertTrue(all(p.requires_grad for p in actor.target_branch.parameters()))
        self.assertTrue(all('exp_avg' in optimizer.state[p] for p in actor.target_branch.parameters()))


if __name__ == '__main__':
    unittest.main()

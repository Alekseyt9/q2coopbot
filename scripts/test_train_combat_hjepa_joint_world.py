import unittest
import torch
from train_combat_hjepa_joint_world import window_targets,effect_predictions


class JointWorldTargets(unittest.TestCase):
    def test_absorbing_terminal_reward_is_not_repeated(self):
        effects=torch.zeros(3,6);effects[2,5]=1
        indexes=torch.tensor([[0,1,2]+[2]*13,[2]*16]);lengths=torch.tensor([3,1])
        result=window_targets(effects,indexes,lengths)
        self.assertEqual(result[0,:,5].tolist(),[0.,1.,1.])
        self.assertEqual(result[1,:,5].tolist(),[1.,1.,1.])

    def test_nonterminal_costs_accumulate_for_each_horizon(self):
        effects=torch.zeros(16,6);effects[:,1]=.01
        result=window_targets(effects,torch.arange(16)[None,:],torch.tensor([16]))
        for i,value in enumerate([.01,.04,.16]):self.assertAlmostEqual(float(result[0,i,1]),value,places=6)

    def test_physical_effect_readout_stays_finite_and_nonnegative(self):
        raw=torch.tensor([[-1000.,-1000.,1000.,0.,1000.,-1000.]])
        result=effect_predictions(raw)
        self.assertTrue(torch.isfinite(result).all());self.assertTrue((result>=0).all())
        self.assertEqual(float(result[0,0]),0.);self.assertEqual(float(result[0,4]),1.)


if __name__=='__main__':unittest.main()

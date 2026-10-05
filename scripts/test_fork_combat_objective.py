import unittest
import test_fork_combat_exploration as exploration
from fork_combat_objective import rebase
from ppo_combat import network, torch


class ObjectiveForkTests(unittest.TestCase):
    def test_reset_critic_and_optimizers_preserve_actor_and_parent(self):
        m,s=exploration.ExplorationForkTests().fixture()
        s['config']['value_lr']=.001
        s['value_optimizer']['param_groups']=[{'lr':.001}]
        config={**s['config'],'objective_reward_sha256':'pinned'}
        child,c=rebase(m,s,config)
        self.assertEqual(child['actor'],m['actor']);self.assertEqual(child['log_std'],m['log_std'])
        self.assertTrue(torch.equal(network(child['value'])(torch.ones(2,386)),torch.zeros(2,1)))
        self.assertEqual(c['value_optimizer']['state'],{});self.assertEqual(c['actor_optimizer']['state'],{})
        self.assertTrue(s['value_optimizer']['state']);self.assertNotIn('objective_reward_sha256',s['config'])
        self.assertEqual(c['consumed_rollouts'],['old'])
        self.assertEqual(c['config'],config)
        self.assertEqual(child['value'][0],m['value'][0])

    def test_cannot_silently_change_optimizer_parameters(self):
        m,s=exploration.ExplorationForkTests().fixture()
        with self.assertRaises(ValueError):rebase(m,s,{**s['config'],'objective_reward_sha256':'new','actor_lr':.01})
        with self.assertRaises(ValueError):rebase(m,s,s['config'])


if __name__=='__main__':unittest.main()

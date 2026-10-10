import copy
import unittest
from fork_combat_gae_checkpoint_cuda import validate_config_fork


class ConfigForkTests(unittest.TestCase):
    def test_only_lambda_can_change(self):
        old=dict(version='combat_ppo_training_v1',gamma=.99,**{'lambda':.95},actor_lr=3e-5,objective_reward_sha256='pinned')
        new={**old,'lambda':.99}
        validate_config_fork(old,new)
        for field,value in [('gamma',.995),('actor_lr',.003),('objective_reward_sha256','other')]:
            invalid={**new,field:value}
            with self.assertRaises(AssertionError):validate_config_fork(old,invalid)
        for value in (.95,float('nan'),float('inf'),1.1,-.1,True):
            with self.assertRaises(AssertionError):validate_config_fork(old,{**old,'lambda':value})
        with self.assertRaises(AssertionError):validate_config_fork(old,{**new,'extra':1})


if __name__=='__main__':unittest.main()

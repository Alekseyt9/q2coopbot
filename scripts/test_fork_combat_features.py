import copy
import unittest
from fork_combat_features import extend
from test_fork_combat_exploration import ExplorationForkTests
from ppo_combat import network, torch


class FeatureForkTests(unittest.TestCase):
    def test_zero_extension_preserves_outputs_and_checkpoint(self):
        model,state=ExplorationForkTests().fixture()
        model['feature_version']='combat_features_v1'
        state['config']['value_lr']=.001
        before=copy.deepcopy(model)
        child,result=extend(model,state)
        x=torch.randn(31,386);extra=torch.randn(31,40)
        for name in ('actor','value'):
            torch.testing.assert_close(network(model[name])(x),network(child[name])(torch.cat((x,extra),1)),rtol=1e-5,atol=1e-6)
            self.assertEqual(result[name].keys(),network(child[name]).state_dict().keys())
        self.assertEqual(model,before)
        self.assertEqual(child['feature_version'],'combat_features_v2')
        for name in ('actor_optimizer','value_optimizer'):
            self.assertEqual(result[name]['state'],{})
        for name in ('config','consumed_rollouts','total_actor_steps','updates_completed'):
            self.assertEqual(result[name],state[name])
        self.assertTrue(torch.equal(result['rng'],state['rng']))
        self.assertEqual(child['log_std'],model['log_std'])
        grandchild,newstate=extend(child,result)
        self.assertEqual(grandchild['feature_version'],'combat_features_v3')
        for name in ('actor','value'):
            torch.testing.assert_close(network(child[name])(torch.cat((x,extra),1)),network(grandchild[name])(torch.cat((x,extra,extra),1)),rtol=1e-5,atol=1e-6)
        with self.assertRaises(ValueError):extend(grandchild,newstate)
        bad=copy.deepcopy(state);bad['actor']['0.bias'][0]+=1
        with self.assertRaises(ValueError):extend(model,bad)


if __name__=='__main__':unittest.main()

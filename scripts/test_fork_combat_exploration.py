import unittest
from fork_combat_exploration import fork
from ppo_combat import layers, torch


class ExplorationForkTests(unittest.TestCase):
    def fixture(self):
        actor = torch.nn.Sequential(torch.nn.Linear(386,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,8))
        value = torch.nn.Sequential(torch.nn.Linear(386,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
        model = dict(kind='combat_ppo_v1', deterministic=False, actor=layers(actor), value=layers(value), log_std=[-2.3,-2.3,-5.3,-5.3])
        state = dict(version='combat_ppo_checkpoint_v2',actor=actor.state_dict(),value=value.state_dict(),log_std=torch.tensor(model['log_std']),
                     actor_optimizer=dict(state={0:dict(step=torch.tensor(1))},param_groups=[dict(lr=.00001)]),
                     value_optimizer=dict(state={0:dict(step=torch.tensor(3))}),config=dict(actor_lr=.0003),
                     consumed_rollouts=['old'],updates_completed=9,total_actor_steps=90,rng=torch.get_rng_state())
        return model,state

    def test_fork_preserves_weights_value_state_history_and_parent(self):
        m,s=self.fixture();child,c=fork(m,s,[-1.2,-1.2,-3.9,-3.9])
        self.assertEqual(child['actor'],m['actor']);self.assertEqual(child['value'],m['value'])
        self.assertEqual(c['actor_optimizer']['state'],{})
        self.assertEqual(c['value_optimizer']['state'][0]['step'],s['value_optimizer']['state'][0]['step'])
        self.assertTrue(torch.equal(c['rng'],s['rng']))
        self.assertEqual(c['consumed_rollouts'],['old']);self.assertEqual(c['total_actor_steps'],90)
        self.assertTrue(s['actor_optimizer']['state']);self.assertEqual(m['log_std'],[-2.3,-2.3,-5.3,-5.3])
        self.assertEqual(c['actor_optimizer']['param_groups'][0]['lr'],.0003)

    def test_reject_nonfinite_range_and_mismatched_parent(self):
        m,s=self.fixture()
        for std in [[0]*3,[0,0,0,float('nan')],[0,0,0,2]]:
            with self.assertRaises(ValueError):fork(m,s,std)
        s['actor']['0.bias'][0]+=1
        with self.assertRaises(ValueError):fork(m,s,[0]*4)


if __name__ == '__main__':unittest.main()

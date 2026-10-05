import copy
import math
import unittest
from fork_combat_aim import aim_labels, fit
from ppo_combat import network, layers, torch


class AimForkTests(unittest.TestCase):
    def fixture(self):
        torch.set_num_threads(2)
        torch.manual_seed(20261006)
        actor=torch.nn.Sequential(torch.nn.Linear(810,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,8))
        value=torch.nn.Sequential(torch.nn.Linear(810,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
        model=dict(kind='combat_ppo_v1',feature_version='combat_features_v4',deterministic=False,actor=layers(actor),value=layers(value),log_std=[-2.3,-2.3,-5.3,-5.3])
        state=dict(version='combat_ppo_checkpoint_v2',actor=actor.state_dict(),value=value.state_dict(),log_std=torch.tensor(model['log_std']),actor_optimizer=dict(state={0:dict(step=torch.tensor(1))},param_groups=[dict(lr=.003)]),value_optimizer=dict(state={0:dict(step=torch.tensor(3))}),config=dict(actor_lr=.0003),consumed_rollouts=['old'],updates_completed=50,total_actor_steps=492,rng=torch.get_rng_state())
        return model,state

    def rows(self):
        rows = []
        for angle in (-50., -10., 0., 10., 50.):
            f = [0.]*810
            f[426:431] = [1, math.sin(math.radians(angle)), math.cos(math.radians(angle)), 0, 1]
            rows.append(dict(features=f))
        return rows

    def test_observed_bbox_masks_signs_limit_and_nearest_slot(self):
        rows = self.rows()
        _, target = aim_labels(rows)
        self.assertEqual(target[:,0].tolist(), [-20., -10., 0., 10., 20.])
        self.assertEqual(target[:,1].tolist(), [0.]*5)
        rows[0]['features'][426] = 0
        rows[0]['features'][431:436] = [1, 0, 1, -1, 0]
        _, target = aim_labels(rows)
        self.assertEqual(target[0].tolist(), [0., -20.])
        rows[0]['features'][431] = 0
        x, _ = aim_labels(rows)
        self.assertEqual(len(x), 4)

    def test_reject_bad_labels(self):
        for limit in (0, 46, float('nan')):
            with self.assertRaises(ValueError): aim_labels(self.rows(), limit)
        for value in (float('nan'), float('inf'), 0):
            rows=self.rows();rows[0]['features'][428]=value
            with self.assertRaises(ValueError): aim_labels(rows)
        with self.assertRaises(ValueError): aim_labels([dict(features=[0.]*810)])

    def test_only_aim_rows_changed_and_parent_checkpoint_preserved(self):
        torch.set_num_threads(2)
        actor=torch.nn.Sequential(torch.nn.Linear(810,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,8))
        value=torch.nn.Sequential(torch.nn.Linear(810,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
        model=dict(kind='combat_ppo_v1',feature_version='combat_features_v4',deterministic=False,actor=layers(actor),value=layers(value),log_std=[-2.3,-2.3,-5.3,-5.3])
        state=dict(version='combat_ppo_checkpoint_v2',actor=actor.state_dict(),value=value.state_dict(),log_std=torch.tensor(model['log_std']),actor_optimizer=dict(state={0:dict(step=torch.tensor(1))},param_groups=[dict(lr=.003)]),value_optimizer=dict(state={0:dict(step=torch.tensor(3))}),config=dict(actor_lr=.0003),consumed_rollouts=['old'],updates_completed=50,total_actor_steps=492,rng=torch.get_rng_state())
        original=copy.deepcopy(model)
        child,updated,report=fit(model,state,self.rows(),20)
        self.assertEqual(model,original)
        self.assertEqual(child['actor'][:2],original['actor'][:2])
        for i in (0,1,4,5,6,7):
            self.assertEqual(child['actor'][2]['weight'][i],original['actor'][2]['weight'][i])
            self.assertEqual(child['actor'][2]['bias'][i],original['actor'][2]['bias'][i])
        self.assertEqual(report['non_aim_raw_max_error'],0)
        self.assertEqual(child['log_std'],model['log_std'])
        self.assertEqual(updated['actor_optimizer']['state'],{})
        self.assertEqual(updated['value_optimizer']['state'],{})
        self.assertTrue(state['actor_optimizer']['state'])
        self.assertEqual(updated['updates_completed'],50)
        self.assertEqual(updated['consumed_rollouts'],['old'])
        self.assertTrue(torch.equal(updated['rng'],state['rng']))
        self.assertEqual(child['value'][-1]['bias'],[0.])
        self.assertTrue(all(x==0 for x in child['value'][-1]['weight'][0]))
        torch.testing.assert_close(network(child['actor'])(aim_labels(self.rows())[0])[:,[0,1,4,5,6,7]],actor(aim_labels(self.rows())[0])[:,[0,1,4,5,6,7]],rtol=0,atol=0)

    def test_joint_encoder_learns_with_other_command_retention(self):
        model,state=self.fixture();original=copy.deepcopy(model)
        child,updated,report=fit(model,state,self.rows(),150,mode='joint')
        self.assertEqual(model,original)
        self.assertNotEqual(child['actor'][0],model['actor'][0])
        self.assertLess(sum(report['train_command_mae_after_degrees']),sum(report['train_command_mae_before_degrees']))
        self.assertLess(max(report['movement_command_mae']),.05)
        self.assertLess(report['attack_probability_mae'],.05)
        self.assertLess(report['vertical_kl'],.02)
        self.assertEqual(updated['actor_optimizer']['state'],{})
        self.assertEqual(updated['value_optimizer']['state'],{})
        self.assertEqual(updated['updates_completed'],50)
        self.assertEqual(updated['consumed_rollouts'],['old'])
        self.assertEqual(child['log_std'],model['log_std'])
        self.assertTrue(torch.equal(updated['rng'],state['rng']))
        for weight in (0,float('nan'),101):
            with self.assertRaises(ValueError):fit(model,state,self.rows(),1,mode='joint',distill_weight=weight)


if __name__=='__main__': unittest.main()

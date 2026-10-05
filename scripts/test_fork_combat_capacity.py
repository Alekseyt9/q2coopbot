import copy
import unittest
from fork_combat_capacity import resize, widen
from test_fork_combat_exploration import ExplorationForkTests
from ppo_combat import network, torch


class CapacityTests(unittest.TestCase):
    def fixture(self):
        m, s = ExplorationForkTests().fixture()
        s['config']['value_lr'] = .001
        return m, s

    def test_preserves_outputs_on_nonzero_inputs_and_state(self):
        torch.manual_seed(17)
        m, s = self.fixture()
        x = torch.randn(127, 386) * 3
        for width, total in ((64,58445),(128,133261),(256,332045)):
            c, state = resize(m, s, width, 19)
            for key in ('actor','value'):
                torch.testing.assert_close(network(m[key])(x), network(c[key])(x), rtol=1e-5, atol=1e-6)
                self.assertEqual(len(c[key][0]['bias']),width)
                self.assertEqual(len(c[key][1]['bias']),width)
                self.assertEqual(state[key].keys(),network(c[key]).state_dict().keys())
            self.assertEqual(sum(p.numel() for key in ('actor','value') for p in network(c[key]).parameters())+4,total)
            for key in ('actor_optimizer','value_optimizer'):
                self.assertEqual(state[key]['state'],{})
            self.assertEqual(state['consumed_rollouts'],s['consumed_rollouts'])
            self.assertEqual(state['total_actor_steps'],s['total_actor_steps'])
            self.assertTrue(torch.equal(state['rng'],s['rng']))
            self.assertEqual(c['log_std'],m['log_std'])
        self.assertTrue(s['actor_optimizer']['state'])

    def test_reproducible_and_rejects_shrink_or_mismatch(self):
        m,s=self.fixture()
        c,_=resize(m,s,128,12);d,_=resize(m,s,128,12)
        self.assertEqual(c,d)
        e,_=resize(m,s,128,13);self.assertNotEqual(c,e)
        with self.assertRaises(ValueError):resize(m,s,512,12)
        with self.assertRaises(ValueError):widen(c['actor'],64,torch.Generator())
        bad=copy.deepcopy(s);bad['actor']['0.bias'][0]+=1
        with self.assertRaises(ValueError):resize(m,bad,128,12)


if __name__ == '__main__':unittest.main()

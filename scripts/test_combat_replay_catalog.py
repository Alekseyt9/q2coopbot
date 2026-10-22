import copy
import unittest
from combat_replay_catalog import policy_family,disjoint_keys


class Catalog(unittest.TestCase):
    def test_duplicate_transition_is_not_extra_experience(self):
        with self.assertRaises(AssertionError):disjoint_keys([[dict(seed=1,index=1)],[dict(seed=1,index=1)]])
        self.assertEqual(len(disjoint_keys([[dict(seed=1,index=1)],[dict(seed=2,index=1)]])),2)

    def test_only_aim_weights_variance_and_value_may_change(self):
        model=dict(actor=[1],value=[2],attention=dict(actor=[3],value=[4]),sampled_motion_aim=dict(version='v1',layers=[5]),log_std=[1,2,3,4],weapon=[6])
        other=copy.deepcopy(model);other['sampled_motion_aim']['layers']=[7];other['log_std'][2:]=[8,9];other['value']=[10];other['attention']['value']=[11]
        self.assertEqual(policy_family(model),policy_family(other))
        for key in ('actor','weapon'):
            bad=copy.deepcopy(other);bad[key]=[99]
            self.assertNotEqual(policy_family(model),policy_family(bad))
        bad=copy.deepcopy(other);bad['log_std'][0]=99
        self.assertNotEqual(policy_family(model),policy_family(bad))


if __name__=='__main__':unittest.main()

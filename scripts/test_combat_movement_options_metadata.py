"""Metadata only, no tensors or neural module construction."""
import copy
import unittest
from combat_movement_options import validate, VERSION, SAMPLER, DISTRIBUTION, SHIFTS


class Metadata(unittest.TestCase):
    def model(self):
        def network(n):
            result=[];width=1635
            for count in (32,16,n):
                result.append(dict(weight=[[0.]*width for _ in range(count)],bias=[0.]*count));width=count
            return result
        return dict(feature_version='combat_features_v13_memory_static_routes',action_distribution=DISTRIBUTION,
            attention={'present':True},aim_candidate_head={'present':True},conditional_fire={'present':True},
            movement_options=dict(version=VERSION,sampler=SAMPLER,enabled=False,duration_frames=5,
                mean_shifts=copy.deepcopy(SHIFTS),actor=network(5),value=network(1)))

    def test_contract(self):
        validate(self.model())
        for mutate in (
            lambda m:m['movement_options'].update(duration_frames=4),
            lambda m:m.update(action_distribution='combat_joint_aim_candidates_v1'),
            lambda m:m.update(feature_version='combat_features_v10'),
            lambda m:m.update(diagnostic_movement_probe={'present':True}),
            lambda m:m.update(wide_aim={'present':True}),
            lambda m:m['movement_options']['actor'][0]['weight'][0].pop(),
        ):
            model=self.model();mutate(model)
            with self.assertRaises(AssertionError):validate(model)


if __name__=='__main__':unittest.main()

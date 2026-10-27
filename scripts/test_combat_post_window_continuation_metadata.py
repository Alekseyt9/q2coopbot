"""Pure metadata checks. No tensor creation or neural inference."""
import copy
import unittest
from combat_continuation_probe import VERSION, POST_WINDOW_VERSION, validate, effective_seed


class PostWindowMetadata(unittest.TestCase):
    def model(self):
        return dict(feature_version='combat_features_v13_memory_static_routes',
            action_distribution='combat_joint_aim_candidates_v1',aim_candidate_head={'present':True},
            diagnostic_movement_probe=dict(version='combat_window_movement_mean_shift_v1',frame=374,duration_frames=5,mean_shift=[-1.,0.]),
            diagnostic_continuation_probe=dict(version=POST_WINDOW_VERSION,fork_frame=379,seed=98765))

    def test_admission_and_version_isolation(self):
        validate(self.model())
        for mutate in (
            lambda m:m['diagnostic_continuation_probe'].update(version=VERSION),
            lambda m:m['diagnostic_continuation_probe'].update(fork_frame=375),
            lambda m:m['diagnostic_continuation_probe'].update(fork_frame=380),
            lambda m:m.update(feature_version='combat_features_v10'),
            lambda m:m.pop('diagnostic_movement_probe'),
            lambda m:m['diagnostic_movement_probe'].update(duration_frames=1),
            lambda m:m.update(diagnostic_anchor_probe={'present':True}),
            lambda m:m.update(diagnostic_weapon_probe={'present':True}),
            lambda m:m.update(deterministic=True),
            lambda m:m['diagnostic_movement_probe'].update(frame=2**31-5),
        ):
            model=self.model();mutate(model)
            with self.assertRaises(AssertionError):validate(model)

    def test_seed_and_original_row_unchanged(self):
        model=self.model()
        for frame,expected in ((374,123),(378,123),(379,98765),(380,98765)):
            row=dict(frame=frame,sample=dict(sampling_seed=123));before=copy.deepcopy(row)
            self.assertEqual(effective_seed(model,row),expected)
            self.assertEqual(row,before)


if __name__=='__main__':unittest.main()

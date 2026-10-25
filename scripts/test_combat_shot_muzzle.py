"""Observation boundaries and unknown geometry; no neural execution."""
import copy
import math
import unittest
from train_combat_shot_muzzle_cuda import state_inputs, local, training_extension


class ObservationBoundary(unittest.TestCase):
    def observation(self):
        return dict(view_angles=[0, 0, 0], velocity=[100, 0, 0], on_ground=True, ducked=False,
            gun_frame=8, kick_angles_degrees=[0, 0, 0], position=[10, 20, 30],
            previous_applied_command=dict(Forward=0, Side=0, Up=0),
            local_geometry=dict(ground_drop=0, up_eye_distance=64, probes=[dict(
                eye_ray_distance=256, standing_hull_clearance=64, ground_drop_at_40=0) for _ in range(8)]))

    def test_unknown_floor_is_distinct_from_flat_floor(self):
        observed = self.observation(); unknown = copy.deepcopy(observed)
        unknown['local_geometry']['ground_drop'] = None
        unknown['local_geometry']['probes'][0]['ground_drop_at_40'] = None
        self.assertNotEqual(state_inputs(observed), state_inputs(unknown))
        self.assertTrue(all(math.isfinite(v) for v in state_inputs(unknown)))

    def test_absolute_position_and_future_labels_are_not_inputs(self):
        observed = self.observation(); contaminated = copy.deepcopy(observed)
        contaminated.update(position=[999, -999, 1234], next_observation={'position': [1, 2, 3]},
                            shot_groundtruth={'start': [5, 6, 7]}, hidden_enemy_pose=[9, 8, 7])
        self.assertEqual(state_inputs(observed), state_inputs(contaminated))

    def test_unknown_hull_and_ray_are_not_obstacles_or_clear_space(self):
        unknown = self.observation()
        unknown['local_geometry']['probes'][0]['standing_hull_clearance'] = None
        unknown['local_geometry']['probes'][0]['eye_ray_distance'] = None
        unknown['local_geometry']['up_eye_distance'] = None
        clear = state_inputs(self.observation())
        encoded = state_inputs(unknown)
        self.assertEqual(len(encoded), len(clear))
        self.assertEqual(encoded.count(-1.), 3)
        self.assertNotEqual(encoded, clear)

    def test_displacement_is_in_current_yaw_frame(self):
        forward = local([0, 19, 14], math.pi/2)
        self.assertAlmostEqual(forward[0], 19)
        self.assertAlmostEqual(forward[1], 0)
        self.assertEqual(forward[2], 14)

    def test_extension_cannot_train_on_diagnostic_or_repeat_worlds(self):
        original = [dict(world='a:1', split='train'), dict(world='b:2', split='diagnostic')]
        for extra in [[dict(world='b:2', split='train')], [dict(world='a:1', split='train')],
                      [dict(world='c:3', split='diagnostic')]]:
            with self.assertRaises(AssertionError):
                training_extension(original, extra)
        self.assertEqual(len(training_extension(original, [dict(world='c:3', split='train')])), 3)


if __name__ == '__main__':
    unittest.main()

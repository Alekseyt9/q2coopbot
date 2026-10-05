import copy
import unittest

from diagnose_combat import summarize


def step(frame=1, yaw=0):
    return dict(owner='provider', observation=dict(identity=dict(life=1, frame=frame),
                position=[0, 0, 0], view_angles=[0, yaw, 0],
                enemies=[dict(clear_shot=True, distance=10, relative=[10, 0, 0])]),
                next_observation=dict(identity=dict(life=1, frame=frame+1), position=[0, 0, 0]),
                action=dict(forward=.2, side=0),
                applied_action=dict(forward=0, side=0, attack=True, yaw_delta_degrees=0))


class DiagnosticsTests(unittest.TestCase):
    def test_runs_break_at_frame_gap_and_life_boundary(self):
        rows = [step(1), step(2), step(4), step(5)]
        other_life = step(6)
        other_life['observation']['identity']['life'] = 2
        r = summarize(rows + [other_life])
        self.assertEqual(r['steps'], 4)
        self.assertEqual(r['longest_stationary_run'], 2)
        self.assertEqual(r['requested_move_stationary'], 4)

    def test_applied_yaw_wrap_and_visibility(self):
        s = step(yaw=-182)
        s['observation']['enemies'][0]['relative'] = [10, -.17455, 0]
        self.assertEqual(summarize([s])['applied_visible_attack_yaw_error_gt15'], 0)
        s['applied_action']['yaw_delta_degrees'] = 30
        self.assertEqual(summarize([s])['applied_visible_attack_yaw_error_gt15'], 1)
        s['observation']['enemies'][0]['clear_shot'] = None
        self.assertEqual(summarize([s])['visible_attack_steps'], 0)

    def test_motion_does_not_cross_missing_frame_or_life(self):
        s = step()
        s['next_observation']['identity']['life'] = 2
        self.assertEqual(summarize([s])['motion_pairs'], 0)
        s['next_observation']['identity']['life'] = 1
        s['next_observation']['identity']['frame'] = 3
        self.assertEqual(summarize([s])['motion_pairs'], 0)
        s['next_observation'] = None
        self.assertEqual(summarize([s])['requested_move_stationary'], 0)

    def test_real_motion_and_ownership(self):
        s = step()
        s['next_observation']['position'] = [3, 4, 0]
        foreign = copy.deepcopy(s)
        foreign['owner'] = 'rules'
        r = summarize([s, foreign])
        self.assertEqual(r['path_length'], 5)
        self.assertEqual(r['steps'], 1)
        self.assertEqual(r['requested_move_stationary'], 0)


if __name__ == '__main__':
    unittest.main()

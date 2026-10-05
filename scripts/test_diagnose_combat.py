import copy
import unittest

from diagnose_combat import summarize, origin_aim_error, bbox_aim_error


def step(frame=1, yaw=0):
    return dict(owner='provider', observation=dict(identity=dict(life=1, frame=frame),
                position=[0, 0, 0], view_angles=[0, yaw, 0], ducked=False,
                enemies=[dict(id=67, **{'class':'monster_parasite'},clear_shot=True, distance=10, relative=[10, 0, 0])]),
                next_observation=dict(identity=dict(life=1, frame=frame+1), position=[0, 0, 0]),
                action=dict(forward=.2, side=0),
                applied_action=dict(forward=0, side=0, attack=True, yaw_delta_degrees=0, pitch_delta_degrees=0))


class DiagnosticsTests(unittest.TestCase):
    def test_parasite_range_separates_enemy_motion_from_retreat(self):
        s=step()
        s['next_observation']['enemies']=[dict(id=67,**{'class':'monster_parasite'},clear_shot=True,relative=[20,0,0])]
        r=summarize([s])
        self.assertEqual(r['parasite_range_increasing'],1)
        self.assertEqual(r['parasite_actual_retreat'],0)
        self.assertEqual(r['parasite_near_stationary'],1)
        s['next_observation']['position']=[-4,0,0]
        self.assertEqual(summarize([s])['parasite_actual_retreat'],1)
        s['next_observation']['identity']['life']=2
        self.assertEqual(summarize([s])['parasite_motion_pairs'],0)
    def test_bbox_aim_matches_go_upper_body_clamp_and_masks(self):
        s=step()
        target=s['observation']['enemies'][0]
        target.update(relative=[100,0,0],observed_solid=8290)
        pitch,angle,_=bbox_aim_error(s['observation'],s['applied_action'],target)
        self.assertAlmostEqual(pitch,0)
        self.assertAlmostEqual(angle,0)
        target['observed_solid']=4194
        self.assertAlmostEqual(bbox_aim_error(s['observation'],s['applied_action'],target)[0],16.69924423,places=6)
        del target['observed_solid']
        self.assertIsNone(bbox_aim_error(s['observation'],s['applied_action'],target))

    def test_eye_height_pitch_sign_and_applied_delta(self):
        s = step()
        target = s['observation']['enemies'][0]
        target['relative'] = [22, 0, 0]
        s['applied_action']['pitch_delta_degrees'] = 45
        pitch, angle, applied = origin_aim_error(s['observation'], s['applied_action'], target)
        self.assertAlmostEqual(pitch, 0)
        self.assertAlmostEqual(angle, 0, places=5)
        self.assertEqual(applied, 45)
        s['observation']['ducked'] = True
        target['relative'] = [2, 0, 0]
        s['applied_action']['pitch_delta_degrees'] = -45
        pitch, angle, _ = origin_aim_error(s['observation'], s['applied_action'], target)
        self.assertAlmostEqual(pitch, 0)
        self.assertAlmostEqual(angle, 0, places=5)

    def test_3d_angle_accounts_for_pitch_and_yaw(self):
        s = step()
        target = s['observation']['enemies'][0]
        target['relative'] = [10, 0, 22]
        s['applied_action'].update(yaw_delta_degrees=90, pitch_delta_degrees=60)
        pitch, angle, _ = origin_aim_error(s['observation'], s['applied_action'], target)
        self.assertAlmostEqual(pitch, 60)
        self.assertAlmostEqual(angle, 90)
        target['relative'] = [0, 0, 22]
        self.assertIsNone(origin_aim_error(s['observation'], s['applied_action'], target))
        self.assertEqual(summarize([s])['origin_aim_samples'], 0)

    def test_pitch_limit_uses_applied_action_and_visible_attack(self):
        s = step()
        s['applied_action']['pitch_delta_degrees'] = 89
        s['interventions'] = [dict(component='pitch', reason='protocol_pitch_limit')]
        self.assertEqual(summarize([s])['applied_attack_pitch_near_limit'], 1)
        self.assertEqual(summarize([s])['pitch_limit_interventions'], 1)
        s['applied_action']['attack'] = False
        self.assertEqual(summarize([s])['origin_aim_samples'], 0)

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

"""Label/identity/clock tests; no neural network execution."""
import io
import math
import unittest

from export_combat_aim_feedback import label, poses


def shot():
    return dict(generation=7, server_frame=100, window=(7, 100, 1, 3), damage=None,
        fields=dict(map='base3', frame='102', shot='1', actor='1', start='10,0,22',
                    aim='1,0,0', view='-5,0,0', recoil='5,0,0', velocity='0,100,0',
                    muzzle_blocked='0', water='0'))


def stream(**changes):
    fields = dict(map='base3', frame='102', shot='1', actor='1', entity='2',
        **{'class': 'turret_driver'}, health='100', solid='2', origin='100,100,0',
        mins='-16,-16,-24', maxs='16,16,32', velocity='0,0,0')
    fields.update(changes)
    return ('sv_test_step version=1 phase=begin spawncount=7 frame=100 seq=3 actor=1\n'
        'sv_test_hitscan spawncount=7 server_frame=100 g_test_hitscan version=1 event=fire shot=1\n'
        'sv_test_aim_pose spawncount=7 server_frame=100 g_test_aim_pose version=1 '+
        ' '.join(k+'='+v for k, v in fields.items())+'\n'
        'sv_test_step version=1 phase=end spawncount=7 frame=101 seq=3 actor=1\n')


class ShotPoseTests(unittest.TestCase):
    def test_accepts_nonprefix_monster_and_exact_join(self):
        result = poses(io.StringIO(stream()), [shot()])
        self.assertEqual(result[(7, 1)][2]['monster_class'], 'turret_driver')

    def test_rejects_wrong_shot_actor_map_game_clock_and_bbox(self):
        for change in [dict(shot='2'), dict(actor='2'), dict(map='base2'), dict(frame='103'),
                       dict(mins='0,0,40'), dict(origin='nan,0,0')]:
            with self.subTest(change=change), self.assertRaises(AssertionError):
                poses(io.StringIO(stream(**change)), [shot()])

    def test_rejects_native_clock_window_and_duplicate_pose(self):
        data = stream()
        for invalid in [data.replace('server_frame=100 g_test_aim_pose', 'server_frame=102 g_test_aim_pose'),
                        data.replace('seq=3 actor=1\n', 'seq=4 actor=1\n', 1),
                        data.replace('sv_test_step version=1 phase=end',
                            data.splitlines()[2]+'\nsv_test_step version=1 phase=end')]:
            with self.assertRaises(AssertionError):
                poses(io.StringIO(invalid), [shot()])

    def test_rejects_pose_before_fire(self):
        lines = stream().splitlines(); lines[1], lines[2] = lines[2], lines[1]
        with self.assertRaises(AssertionError):
            poses(iter(lines), [shot()])

    def test_direct_origin_motion_is_measured_even_with_zero_velocity(self):
        later = stream().replace('seq=3', 'seq=4').replace('frame=101 seq', 'frame=102 seq')
        later = later.replace('frame=100', 'frame=101').replace('frame=102 shot=1', 'frame=103 shot=2')
        later = later.replace('event=fire shot=1', 'event=fire shot=2').replace('origin=100,100,0', 'origin=110,100,0')
        second = shot(); second['fields'] = dict(second['fields'], frame='103', shot='2')
        second.update(server_frame=101, window=(7, 101, 1, 4))
        result = poses(io.StringIO(stream()+later), [shot(), second])
        self.assertIsNone(result[(7, 1)][2]['shot_interval_horizontal_speed'])
        self.assertEqual(result[(7, 2)][2]['velocity'], (0, 0, 0))
        self.assertAlmostEqual(result[(7, 2)][2]['shot_interval_horizontal_speed'], 100)

    def test_label_includes_muzzle_parallax_and_recoil(self):
        native = poses(io.StringIO(stream()), [shot()])[(7, 1)][2]
        step = dict(owner='provider', action=dict(target_entity=2, target_track=8),
            observation=dict(position=[0, 0, 0], view_angles=[0, 0, 0],
                enemies=[dict(id=2, observed_track=8, relative=[100, 100, 0],
                              clear_shot=True, observed_solid=8290)]),
            applied_action=dict(pitch_delta_degrees=-5, yaw_delta_degrees=0))
        result = label(shot(), native, step)
        self.assertAlmostEqual(result['desired_view_degrees'][0], -5)
        self.assertAlmostEqual(result['desired_view_degrees'][1], math.degrees(math.atan2(100, 90)))
        self.assertEqual(result['observed_origin_error_units'], 0)
        self.assertEqual(result['native_recoil'], (5, 0, 0))
        step['action']['target_track'] = 9
        with self.assertRaises(StopIteration):
            label(shot(), native, step)


if __name__ == '__main__':
    unittest.main()

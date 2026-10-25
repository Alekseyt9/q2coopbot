"""Geometry/filter tests only; does not import or execute neural models."""
import copy
import unittest
from prepare_combat_stationary_shotgun_aim import label


class Labels(unittest.TestCase):
    def fixture(self):
        o = dict(weapon='Shotgun', on_ground=True, ducked=False, position=[0, 0, 0], velocity=[0, 0, 0],
                 view_angles=[0, 0, 0], identity=dict(frame=100),
                 enemies=[dict(id=9, clear_shot=True, observed_solid=8290, relative=[64, 0, -64])])
        return dict(observation=o, next_observation=copy.deepcopy(o), action=dict(target_entity=9),
                    sample=dict(target=1, aim_mode=1))

    def test_elevation_and_ducked_muzzle(self):
        s = self.fixture()
        standing = label(s)
        self.assertEqual(standing['delta_degrees'], [0., 13.5])
        s['observation']['ducked'] = s['next_observation']['ducked'] = True
        self.assertLess(label(s)['unclipped_delta_degrees'][1], standing['unclipped_delta_degrees'][1])

    def test_stance_changed_or_moving_is_rejected(self):
        s = self.fixture(); s['next_observation']['ducked'] = True
        self.assertIsNone(label(s))
        s = self.fixture(); s['next_observation']['position'][0] = 1
        self.assertIsNone(label(s))

    def test_no_hidden_or_unavailable_target(self):
        s = self.fixture(); s['observation']['enemies'][0]['clear_shot'] = False
        self.assertIsNone(label(s))
        s = self.fixture(); s['observation']['weapon'] = 'Machinegun'
        self.assertIsNone(label(s))


if __name__ == '__main__':
    unittest.main()

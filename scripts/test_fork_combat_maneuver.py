import copy
import math
import unittest
from fork_combat_maneuver import movement_label, fit
import test_fork_combat_aim


class ManeuverTests(unittest.TestCase):
    def features(self, directions=(0,)):
        f = [0.]*810
        f[2] = f[21] = 1
        f[9] = 1
        f[808] = 2/8
        for direction in directions:
            f[25+6*direction:31+6*direction] = [1, 1, 1, 1, 1, .01]
        return f

    def test_yaw_and_quake_side_sign(self):
        f = self.features()
        self.assertEqual(movement_label(f, [0.]*8), [.75, -0.])
        raw = [0.]*8
        raw[2] = math.atanh(.5)  # Simultaneous 90-degree left turn.
        forward, side = movement_label(f, raw)
        self.assertAlmostEqual(forward, 0.)
        self.assertAlmostEqual(side, .75)
        forward, side = movement_label(self.features((2,)), [0.]*8)
        self.assertAlmostEqual(forward, 0.)
        self.assertAlmostEqual(side, -.75)

    def test_unknown_ground_blocked_and_missing_probe_are_not_safe(self):
        for index in (2, 21, 25, 27, 29):
            f = self.features(); f[index] = 0
            self.assertIsNone(movement_label(f, [0.]*8))
        f = self.features(); f[28] = 39/64
        self.assertIsNone(movement_label(f, [0.]*8))
        f[28] = 40/64
        self.assertIsNotNone(movement_label(f, [0.]*8))

    def test_observed_barrel_and_parasite_choose_clear_retreat(self):
        f = self.features((0, 4))
        f[259] = 1
        f[260:269] = [1, 100/512, 100/512, 0, 0, 1, 0, 0, 0]
        forward, _ = movement_label(f, [0.]*8)
        self.assertAlmostEqual(forward, -.75)
        f[259] = 0
        f[73:85] = [1, 100/512, 100/512, 0, 0, 0, 0, 0, 0, 1, 0, 1]
        self.assertAlmostEqual(movement_label(f, [0.]*8)[0], -.75)
        # Unknown props do not create an observed barrel.
        f[73] = 0
        self.assertAlmostEqual(movement_label(self.features(), [0.]*8)[0], .75)

    def test_invalid_observations(self):
        for value in (float('nan'), float('inf')):
            f = self.features(); f[0] = value
            with self.assertRaises(ValueError): movement_label(f, [0.]*8)
        with self.assertRaises(ValueError): movement_label([0.], [0.]*8)
        with self.assertRaises(ValueError): movement_label(self.features(), [0.])

    def test_fork_retains_parent_checkpoint_and_solo_commands(self):
        model, state = test_fork_combat_aim.AimForkTests().fixture()
        original = copy.deepcopy(model)
        rows = [dict(features=self.features((direction,)), seed=1) for direction in (0, 2, 4)]
        solo = [dict(features=self.features((direction,)), seed=2) for direction in (1, 3)]
        for row in solo: row['features'][808] = 1/8
        child, updated, report = fit(model, state, rows, solo, epochs=200)
        self.assertEqual(model, original)
        self.assertEqual(report['labels'], 3)
        self.assertLess(sum(report['movement_mae_after']), sum(report['movement_mae_before']))
        self.assertLess(max(report['solo_movement_mae']), .05)
        self.assertEqual(updated['updates_completed'], state['updates_completed'])
        self.assertEqual(updated['consumed_rollouts'], state['consumed_rollouts'])
        self.assertEqual(updated['actor_optimizer']['state'], {})
        self.assertEqual(updated['value_optimizer']['state'], {})
        self.assertEqual(child['log_std'], model['log_std'])


if __name__ == '__main__': unittest.main()

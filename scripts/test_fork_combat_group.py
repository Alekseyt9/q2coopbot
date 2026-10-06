import copy
import math
import unittest
from fork_combat_group import labels, fit
from test_fork_combat_maneuver import ManeuverTests
from test_fork_combat_aim import AimForkTests


class GroupTests(unittest.TestCase):
    def features(self, directions=range(8)):
        f = ManeuverTests().features(directions)
        for slot, parasite, x, angle in [(0, False, 100., 0.), (1, True, 180., 45.)]:
            at = 73+12*slot
            f[at:at+12] = [1,x/512,x/512,0,0,0,0,0,0,int(parasite),int(not parasite),1]
            at = 426+5*slot
            f[at:at+5] = [1,math.sin(math.radians(angle)),math.cos(math.radians(angle)),0,1]
        return f

    def test_non_nearest_threat_and_bbox_masks(self):
        f = self.features()
        _, aim, slot = labels(f,[0.]*8)
        self.assertEqual(slot,1);self.assertEqual(aim,[20.,0.])
        f[431] = 0
        self.assertEqual(labels(f,[0.]*8)[2],0)
        f[426] = 0
        self.assertIsNone(labels(f,[0.]*8)[1])

    def test_ground_clearance_drop_and_solo_masks(self):
        f = self.features((0,))
        f[30] = 49/32
        self.assertIsNone(labels(f,[0.]*8)[0])
        f[30] = 48/32
        self.assertIsNotNone(labels(f,[0.]*8)[0])
        f[2] = 0
        self.assertIsNone(labels(f,[0.]*8)[0])
        self.assertIsNotNone(labels(f,[0.]*8)[1])
        f[808] = 1/8
        self.assertEqual(labels(f,[0.]*8),(None,None,None))

    def test_approaching_projectile_changes_direction_only_with_known_velocity(self):
        f = self.features()
        # Remove enemies to isolate projectile scoring; composition mask remains group.
        f[73:169] = [0.]*96;f[426:466] = [0.]*40
        baseline = labels(f,[0.]*8)[0]
        f[169] = 1
        f[170:182] = [1,120/512,120/512,0,0,1,-400/400,0,0,0,0,1]
        self.assertNotEqual(labels(f,[0.]*8)[0],baseline)
        f[175] = 0
        self.assertEqual(labels(f,[0.]*8)[0],baseline)
        f[175] = 1;f[176] = 1
        self.assertEqual(labels(f,[0.]*8)[0],baseline)

    def test_command_compensates_selected_target_yaw(self):
        f = self.features((0,))
        move,aim,_ = labels(f,[0.]*8)
        self.assertAlmostEqual(move[0],.75*math.cos(math.radians(aim[0])))
        self.assertAlmostEqual(move[1],.75*math.sin(math.radians(aim[0])))
        f[431:436] = [1,0,1,0,1]
        self.assertAlmostEqual(labels(f,[0.]*8)[0][1],0.)

    def test_reject_nonfinite_and_invalid_bbox(self):
        for value in (float('nan'),float('inf')):
            f = self.features();f[0] = value
            with self.assertRaises(ValueError):labels(f,[0.]*8)
        f = self.features();f[433] = 0
        with self.assertRaises(ValueError):labels(f,[0.]*8)

    def test_range_band_returns_to_visible_remaining_parasite(self):
        f = self.features((0,4));f[73:169] = [0.]*96;f[426:466] = [0.]*40
        f[808] = 1/8
        f[73:85] = [1,400/512,400/512,0,0,0,0,0,0,1,0,1]
        f[426:431] = [1,0,1,0,1]
        self.assertEqual(labels(f,[0.]*8),(None,None,None))
        move,aim,slot = labels(f,[0.]*8,range_band=True)
        self.assertGreater(move[0],0);self.assertEqual(aim,[0.,0.]);self.assertEqual(slot,0)
        f[74] = f[75] = 200/512
        self.assertEqual(labels(f,[0.]*8,range_band=True),(None,None,None))
        f[73] = 0
        self.assertEqual(labels(f,[0.]*8,range_band=True),(None,None,None))

    def test_fit_preserves_lineage_and_uses_only_features(self):
        model,state = AimForkTests().fixture();original = copy.deepcopy(model)
        rows = [dict(features=self.features((d,)),seed=7,hidden_reward=1e20) for d in (0,2,4)]
        retain = [dict(features=ManeuverTests().features((d,)),seed=8) for d in (1,3)]
        for row in retain:row['features'][808] = 1/8
        child,new,report = fit(model,state,rows,retain,epochs=300)
        self.assertEqual(model,original)
        self.assertEqual(report['nonnearest_targets'],3)
        self.assertEqual(new['updates_completed'],state['updates_completed'])
        self.assertEqual(new['total_actor_steps'],state['total_actor_steps'])
        self.assertEqual(new['consumed_rollouts'],state['consumed_rollouts'])
        self.assertEqual(child['log_std'],model['log_std'])
        self.assertEqual(new['actor_optimizer']['state'],{})
        self.assertEqual(new['value_optimizer']['state'],{})
        self.assertLess(sum(report['movement_mae_after']),sum(report['movement_mae_before']))

    def test_finish_only_never_labels_two_threats_or_close_solo(self):
        f = self.features((0,4))
        self.assertEqual(labels(f,[0.]*8,finish_only=True),(None,None,None))
        f[73:169] = [0.]*96;f[426:466] = [0.]*40;f[808] = 1/8
        f[73:85] = [1,400/512,400/512,0,0,0,0,0,0,1,0,1]
        f[426:431] = [1,0,1,0,1]
        move,aim,slot = labels(f,[0.]*8,finish_only=True)
        self.assertGreater(move[0],0);self.assertEqual(aim,[0.,0.]);self.assertEqual(slot,0)
        f[75] = 320/512
        self.assertEqual(labels(f,[0.]*8,finish_only=True),(None,None,None))
        f[75] = 400/512;f[82] = 0;f[83] = 1
        self.assertEqual(labels(f,[0.]*8,finish_only=True),(None,None,None))
        with self.assertRaises(ValueError):labels(f,[0.]*8,range_band=True,finish_only=True)

    def test_barrel_escape_learns_distance_not_aim_and_masks_unknown_props(self):
        f = self.features((0,4));f[73:169] = [0.]*96
        f[73:85] = [1,400/512,400/512,0,0,0,0,0,0,1,0,1];f[808] = 1/8
        self.assertEqual(labels(f,[0.]*8,barrel_escape=True),(None,None,None))
        f[259] = 1;f[260:269] = [1,100/512,100/512,0,0,1,0,0,0]
        move,aim,slot = labels(f,[0.]*8,barrel_escape=True)
        self.assertLess(move[0],0);self.assertIsNone(aim);self.assertIsNone(slot)
        f[265] = 0
        self.assertEqual(labels(f,[0.]*8,barrel_escape=True),(None,None,None))
        f[265] = 1;f[259] = 0
        self.assertEqual(labels(f,[0.]*8,barrel_escape=True),(None,None,None))


if __name__ == '__main__':unittest.main()

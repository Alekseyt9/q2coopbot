"""Geometry and common-support tests only; no NN execution."""
import copy
import math
import unittest
from prepare_combat_bodypoint_labels import center_label,usable


class BodypointTests(unittest.TestCase):
    def row(self):
        return dict(version='combat_aim_feedback_labels_v2',sample=dict(aim_mode=1),offline_label=dict(
            native_pose=dict(origin=[100.,0.,0.],mins=[-16.,-16.,-24.],maxs=[16.,16.,32.]),
            native_muzzle=[0.,0.,22.],native_recoil=[5.,0.,0.],native_aim=[1.,0.,0.],
            desired_view_degrees=[-5.,0.],desired_command_delta_degrees=[-5.,0.],
            correction_from_executed_view_degrees=[-5.,0.],observed_bodypoint_error_degrees=0.,
            muzzle_blocked=False,water=False))

    def test_center_recoil_parallax_and_source_immutable(self):
        row=self.row();before=copy.deepcopy(row);result=center_label(row);label=result['offline_label']
        expected=math.degrees(math.atan2(18,100))-5
        self.assertEqual(row,before)
        self.assertEqual(label['desired_point'],[100.,0.,4.])
        self.assertAlmostEqual(label['desired_view_degrees'][0],expected)
        self.assertAlmostEqual(label['desired_command_delta_degrees'][0],expected)
        self.assertIsNone(label['observed_bodypoint_error_degrees'])
        self.assertFalse(label['point_visibility_verified'])

    def test_common_support_rejects_unreachable_fine_water_block(self):
        row=self.row();self.assertTrue(usable(row));self.assertTrue(usable(center_label(row)))
        row['offline_label']['native_muzzle'][2]=100.
        self.assertTrue(usable(row));self.assertFalse(usable(center_label(row)))
        for key in ('water','muzzle_blocked'):
            row=self.row();row['offline_label'][key]=True;self.assertFalse(usable(row))

    def test_invalid_bbox_rejected(self):
        row=self.row();row['offline_label']['native_pose']['maxs'][2]=-30
        with self.assertRaises(AssertionError):center_label(row)


if __name__=='__main__':unittest.main()

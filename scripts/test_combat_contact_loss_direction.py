"""Native right-positive sidemove regression; scalar geometry, no neural math."""
import unittest
from select_combat_contact_loss_roots import camera_relative_mean_shift


class DirectionTest(unittest.TestCase):
    def test_native_cardinal_axes(self):
        for yaw,world,expected in [(0,[100,0,0],[1.5,0]),(0,[0,100,0],[0,-1.5]),
                (90,[100,0,0],[0,1.5]),(90,[0,100,0],[1.5,0]),(180,[0,100,0],[0,1.5])]:
            with self.subTest(yaw=yaw,world=world):
                actual=camera_relative_mean_shift(world,yaw)
                for a,b in zip(actual,expected):self.assertAlmostEqual(a,b,places=12)

    def test_no_direction_for_coincident_or_invalid_memory(self):
        for relative in ([0,0,0],[1,2,0],[float('nan'),0,0]):
            with self.assertRaises(ValueError):camera_relative_mean_shift(relative,0)


if __name__=='__main__':unittest.main()

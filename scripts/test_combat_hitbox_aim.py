import math
import unittest
from audit_combat_hitbox_aim import ballistic_basis, intersects, spread_ray


class HitboxRayTests(unittest.TestCase):
    def test_segment_limit_parallel_miss_and_behind(self):
        lower,upper=[10,-1,-1],[12,1,1]
        self.assertTrue(intersects([0,0,0],[1,0,0],lower,upper))
        self.assertFalse(intersects([0,0,0],[1,0,0],lower,upper,maximum=9))
        self.assertTrue(intersects([0,0,0],[1,0,0],lower,upper,maximum=11))
        self.assertFalse(intersects([0,0,0],[-1,0,0],lower,upper))
        self.assertFalse(intersects([0,2,0],[1,0,0],lower,upper))

    def test_truncates_towards_zero_like_native_vectoangles(self):
        for yaw,elevation in [(23.8,7.9),(-23.8,-7.9)]:
            y,e=math.radians(yaw),math.radians(elevation)
            direction=[math.cos(e)*math.cos(y),math.cos(e)*math.sin(y),math.sin(e)]
            forward,_,_=ballistic_basis(direction)
            self.assertAlmostEqual(math.degrees(math.atan2(forward[1],forward[0])),math.trunc(yaw))
            self.assertAlmostEqual(math.degrees(math.atan2(forward[2],math.hypot(*forward[:2]))),math.trunc(elevation))

    def test_native_spread_basis_and_upper_point_margin(self):
        self.assertEqual(spread_ray([1,0,0],[300,500]),[8192,-300,500])
        start=[0,0,14];lower=[390,-16,-24];upper=[410,16,32]
        upper_ray=spread_ray([400,0,8],[0,500]);center_ray=spread_ray([400,0,-10],[0,500])
        self.assertFalse(intersects(start,upper_ray,lower,upper,maximum=1))
        self.assertTrue(intersects(start,center_ray,lower,upper,maximum=1))


if __name__=='__main__':unittest.main()

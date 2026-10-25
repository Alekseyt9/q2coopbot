"""No engine or NN execution: strict telemetry admission and trigger grouping."""
import unittest
from read_combat_shotgun_telemetry import parse


class Stream(unittest.TestCase):
    def example(self):
        prefix = 'sv_test_sg spawncount=9 server_frame=98 g_test_sg version=1 '
        common = 'map=base3 frame=100 trigger=1 actor=1 '
        return ['sv_test_step version=1 phase=begin spawncount=9 frame=98 seq=8 actor=1',
                prefix+'event=begin '+common+'mod=2 count=2 damage=4 start=0,0,14 aim=1,0,0 origin=0,0,0 velocity=0,0,0 view=0,0,0 viewheight=22 grounded=1 ducked=0 gunframe=8 ammo_before=20',
                prefix+'event=pose '+common+'entity=5 class=monster_parasite health=3 solid=2 origin=64,0,0 mins=-16,-16,-24 maxs=16,16,24 velocity=0,0,0',
                prefix+'event=pellet '+common+'index=0 count=2 mod=2 start=0,0,14 aim=1,0,0 impact=48,0,14 fraction=0.1 spread=0,0 nominal=500,500 water=0 muzzle_blocked=0 sky=0 target=5 target_class=monster_parasite damageable=1 health_before=3 damage=4',
                'sv_test_damage spawncount=9 server_frame=98 g_test_damage version=1 attacker=1 target=5 mod=2 target_class=monster_parasite health_before=3 health_after=-1',
                prefix+'event=pellet '+common+'index=1 count=2 mod=2 start=0,0,14 aim=1,0,0 impact=48,0,14 fraction=0.1 spread=0,0 nominal=500,500 water=0 muzzle_blocked=0 sky=0 target=5 target_class=monster_parasite damageable=1 health_before=-1 damage=4',
                prefix+'event=end '+common+'mod=2 count=2',
                'sv_test_step version=1 phase=end spawncount=9 frame=99 seq=8 actor=1']

    def test_group_not_pellet_reward(self):
        rows = parse(self.example())
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]['pellets']), 2)
        self.assertEqual(rows[0]['direct_monster_damage'], 3)

    def test_bad_or_partial_stream_rejected(self):
        for mutate in [lambda x: x[:-1], lambda x: x[:3]+x[4:],
                       lambda x: [s.replace('index=1', 'index=0') for s in x],
                       lambda x: [s.replace('impact=48,0,14', 'impact=nan,0,14') for s in x],
                       lambda x: x[1:]]:
            with self.assertRaises((ValueError, KeyError)):
                parse(mutate(self.example()))

    def test_empty_old_stream(self):
        self.assertEqual(parse(['ordinary log line']), [])

    def test_weapon_can_fire_at_native_end_phase(self):
        rows = parse([s.replace('server_frame=98', 'server_frame=99') for s in self.example()])
        self.assertEqual(rows[0]['native_phase_offset'], 1)
        with self.assertRaises(ValueError):
            parse([s.replace('server_frame=98', 'server_frame=100') for s in self.example()])


if __name__ == '__main__':
    unittest.main()

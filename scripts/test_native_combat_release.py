import io
import unittest
from native_combat_release import release_boundary


class ReleaseClockTests(unittest.TestCase):
    def test_uses_actual_network_clock_not_game_clock(self):
        manifest=dict(synchronous=True,fixed_world_hold=True,release_game_frame=100)
        for frame in (98,70):
            line=f'sv_test_combat spawncount=7 server_frame={frame} g_test_combat_start game_frame=100 ready=1 seed=12\n'
            self.assertEqual(release_boundary(io.StringIO(line),manifest,12),(7,frame,100))

    def test_rejects_missing_duplicate_wrong_seed_and_game_frame(self):
        manifest=dict(synchronous=True,fixed_world_hold=True,release_game_frame=100)
        line='sv_test_combat spawncount=7 server_frame=98 g_test_combat_start game_frame=100 ready=1 seed=12\n'
        for data in ('',line+line,line.replace('seed=12','seed=13'),line.replace('game_frame=100','game_frame=99'),line.replace('ready=1','ready=0')):
            with self.subTest(data=data),self.assertRaises(AssertionError):
                release_boundary(io.StringIO(data),manifest,12)


if __name__=='__main__':unittest.main()

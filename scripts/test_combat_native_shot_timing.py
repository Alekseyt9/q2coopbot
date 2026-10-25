"""Frame-domain regression tests; scalar only, no neural computation."""
import unittest
from combat_native_shot_timing import physical_shots


def step(frame, ammo, gun, matched=True):
    return dict(observation=dict(identity=dict(frame=frame, actor=1, life=1), health=100,
        weapon='Shotgun', ammo=ammo, gun_frame=gun), native_step=dict(begin_frame=frame,end_frame=frame+1),
        server_execution=dict(matched=matched,window_exclusive=True,recovery_commands=0))


class TimingTest(unittest.TestCase):
    def test_explicit_clocks(self):
        log='sv_test_projectile spawncount=7 server_frame=98 g_test_projectile version=1 event=spawn frame=100 actor=1 mod=1'
        events=physical_shots([step(98,20,19),step(99,19,9)],log,98,110)
        self.assertEqual([x['server_command_frame'] for x in events],[98,98])
        self.assertEqual(events[0]['native_game_frame'],100)
        self.assertEqual(events[1]['observation_after_server_frame'],99)

    def test_rejects_gaps_mismatched_windows_other_actor_and_outside_life(self):
        log='sv_test_projectile spawncount=7 server_frame=98 g_test_projectile version=1 event=spawn frame=100 actor=2 mod=1'
        self.assertEqual(physical_shots([step(98,20,19,False),step(99,19,9)],log,98,110),[])
        self.assertEqual(physical_shots([step(98,20,19),step(100,19,9)],log,98,110),[])
        self.assertEqual(physical_shots([step(98,20,19),step(99,19,9)],log,100,110),[])


if __name__=='__main__': unittest.main()

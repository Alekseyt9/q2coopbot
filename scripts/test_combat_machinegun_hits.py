"""Pure telemetry contract tests. No training or NN execution."""
import io,unittest
from report_combat_machinegun_hits import scan,summarize

FIRE='sv_test_hitscan spawncount=7 server_frame=102 g_test_hitscan version=1 event=fire map=base1 frame=102 shot=3 actor=1 mod=4 start=0,0,0 aim=1,0,0 view=0,0,0 recoil=0,0,0 velocity=20,0,0 spread=5,-2 nominal=300,500 water=0 muzzle_blocked=0 fraction=0.01 impact=100,0,0 sky=0 target=8 target_class=monster_parasite damageable=1 health_before=20 burst=1 gunframe=4 ammo_before=30 damage=8'
DAMAGE='sv_test_damage spawncount=7 server_frame=102 g_test_damage version=1 map=base1 frame=102 attacker=1 target=8 inflictor=1 mod=4 health_before=20 health_after=12 take=8 armor=0 power=0 protection=0 target_class=monster_parasite attacker_class=player shot=0'
BEGIN='sv_test_step version=1 phase=begin spawncount=7 frame=101 seq=9 actor=1'
END='sv_test_step version=1 phase=end spawncount=7 frame=102 seq=9 actor=1'

class HitscanTests(unittest.TestCase):
    def read(self,*lines):return scan(io.StringIO('\n'.join(lines)),{(7,101,1,9)})
    def test_damage_is_joined_by_order_and_health(self):
        shots=self.read(BEGIN,FIRE,DAMAGE,END);report=summarize(shots)
        self.assertEqual(report['counts']['health_damage'],8)
        self.assertEqual(report['live_monster_damage_fraction'],1)
        self.assertEqual(report['counts']['moving_shots'],1)
    def test_missing_damage_is_explicit_contact(self):
        report=summarize(self.read(BEGIN,FIRE,END))
        self.assertEqual(report['counts']['damageable_contact_without_health_damage'],1)
        self.assertEqual(report['live_monster_damage_fraction'],0)
    def test_post_window_damage_cannot_credit_shot(self):
        shots=self.read(BEGIN,FIRE,END,DAMAGE)
        self.assertIsNone(shots[0]['damage'])
    def test_mismatched_damage_target_is_rejected(self):
        with self.assertRaises(AssertionError):self.read(BEGIN,FIRE,DAMAGE.replace('target=8','target=9'),END)
    def test_inconsistent_recoil_direction_is_rejected(self):
        with self.assertRaises(AssertionError):self.read(BEGIN,FIRE.replace('recoil=0,0,0','recoil=-9,0,0'),END)
    def test_duplicate_shot_is_rejected(self):
        with self.assertRaises(AssertionError):self.read(BEGIN,FIRE,FIRE,END)
    def test_invalid_generation_is_rejected(self):
        with self.assertRaises(AssertionError):self.read(BEGIN,FIRE.replace('spawncount=7','spawncount=8'),END)
    def test_truncation_is_rejected(self):
        with self.assertRaises((AssertionError,KeyError)):self.read(BEGIN,FIRE.rsplit(' damage=',1)[0],END)
    def test_outside_window_is_ineligible(self):
        self.assertEqual(summarize(self.read(FIRE))['counts'].get('shots',0),0)
    def test_spread_out_of_range_is_rejected(self):
        with self.assertRaises(AssertionError):self.read(BEGIN,FIRE.replace('spread=5,-2','spread=301,-2'),END)
    def test_missing_fire_for_native_damage_is_rejected(self):
        with self.assertRaises(AssertionError):self.read(BEGIN,DAMAGE,END)
    def test_damage_on_different_map_is_rejected(self):
        with self.assertRaises(AssertionError):self.read(BEGIN,FIRE,DAMAGE.replace('map=base1','map=base2'),END)

if __name__=='__main__':unittest.main()

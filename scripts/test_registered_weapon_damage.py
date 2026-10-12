import unittest
from report_registered_weapon_damage import weapon_damage, physical_events
from test_combat_machinegun_hits import FIRE


class Tests(unittest.TestCase):
    def test_effects_exclude_other_actor_life_and_prerelease_and_cap_overkill(self):
        life=dict(map='base1',spawncount=7,start_frame=100,end_frame=110,bot_entity=1)
        event=dict(map='base1',spawncount=7,frame=102,game_frame=40,attacker=1,
            attacker_class='player',target_class='monster_parasite',mod=1,
            health_before=20,health_after=-80)
        events=[event,dict(event,frame=99),dict(event,frame=111),dict(event,game_frame=39),
            dict(event,attacker=2),dict(event,spawncount=8),dict(event,map='base2'),
            dict(event,target_class='player'),dict(event,health_before=-5,health_after=-105)]
        self.assertEqual(weapon_damage(events,life,40,103),
            {'1':dict(health_damage=20,positive_damage_contacts=1,kills=1)})
        self.assertEqual(weapon_damage([event],life,40,102)['1']['health_damage_after_supported_observation'],20)

    def test_machinegun_count_is_native_event_with_clock_bounds_and_unique_id(self):
        life=dict(map='base1',spawncount=7,start_frame=100,end_frame=110,bot_entity=1)
        lines=['g_test_projectile ready version=1',FIRE]
        self.assertEqual(physical_events(lines,life,102)['machinegun_fire_events'],1)
        self.assertEqual(physical_events(lines,life,103),{})
        with self.assertRaises(AssertionError):physical_events(lines+[FIRE],life,102)


if __name__=='__main__':unittest.main()

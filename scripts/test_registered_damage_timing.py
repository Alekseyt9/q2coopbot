import unittest

from audit_registered_damage_timing import damage_events, summarize_events


def event(**changes):
    value=dict(map='jail1', spawncount=7, frame=98, game_frame=100,
               attacker=1, attacker_class='player', target_class='monster_parasite',
               health_before=10, health_after=-5, take=15)
    value.update(changes)
    return value


class DamageTimingTests(unittest.TestCase):
    life=dict(map='jail1', spawncount=7, start_frame=8, end_frame=150, bot_entity=1)

    def test_release_uses_game_clock_not_server_clock_and_caps_overkill(self):
        result=summarize_events([event(),event(frame=99,game_frame=101,health_before=-5,health_after=-20)],self.life,100)
        self.assertEqual(result['first_monster_health_damage_delay_native_frames'],0)
        self.assertEqual(result['post_release_monster_health_damage'],10)
        self.assertEqual(result['post_release_monster_kills'],1)

    def test_excludes_setup_wrong_world_actor_life_and_zero_health_damage(self):
        records=[event(game_frame=99),event(map='base1'),event(spawncount=8),
                 event(attacker=2),event(attacker_class='monster_gunner'),
                 event(frame=151,game_frame=153),event(target_class='player'),
                 event(health_before=10,health_after=10,take=0)]
        result=summarize_events(records,self.life,100)
        self.assertTrue(result['no_monster_health_damage'])
        self.assertIsNone(result['first_monster_health_damage_game_frame'])
        self.assertEqual(result['post_release_monster_health_damage'],0)

    def test_delayed_damage_remains_an_effect_not_an_intent(self):
        result=summarize_events([event(frame=109,game_frame=111)],self.life,100)
        self.assertEqual(result['first_monster_health_damage_delay_native_frames'],11)

    def test_native_parser_requires_ready_valid_health_and_order(self):
        record=('sv_test_damage spawncount=7 server_frame=98 g_test_damage version=1 '
                'map=jail1 frame=100 attacker=1 target=2 inflictor=1 mod=4 '
                'health_before=10 health_after=-5 take=15 armor=0 power=0 protection=0 '
                'target_class=monster_parasite attacker_class=player shot=3')
        values=list(damage_events(['g_test_damage ready version=1',record]))
        self.assertEqual((values[0]['frame'],values[0]['game_frame'],values[0]['shot']),(98,100,3))
        for invalid in ([record],['g_test_damage ready version=1',record.replace('take=15','take=14')],
                        ['g_test_damage ready version=1',record,record.replace('server_frame=98','server_frame=97')],
                        ['g_test_damage ready version=1',record+' extra=1']):
            with self.assertRaises(AssertionError):list(damage_events(invalid))


if __name__ == '__main__':unittest.main()

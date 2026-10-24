import unittest

from audit_combat_projectile_training_exposure import monster_damage_kind


class InfightingTests(unittest.TestCase):
    def fixture(self,before,after,attacker=467,target=477):
        return dict(attacker_class='monster_gunner',target_class='monster_soldier_light',
                    attacker=attacker,target=target,health_before=before,health_after=after,take=17)

    def test_real_corpse_contact_is_not_live_infighting(self):
        self.assertEqual(monster_damage_kind(self.fixture(-36,-53)),'corpse_contact')
        self.assertEqual(monster_damage_kind(self.fixture(0,-17)),'corpse_contact')

    def test_even_one_live_hp_is_infighting(self):
        self.assertEqual(monster_damage_kind(self.fixture(1,-16)),'infighting')
        self.assertEqual(monster_damage_kind(self.fixture(40,23)),'infighting')

    def test_self_and_player_events_are_distinct(self):
        self.assertEqual(monster_damage_kind(self.fixture(40,23,target=467)),'self_damage')
        fields=self.fixture(40,23);fields['attacker_class']='player'
        self.assertIsNone(monster_damage_kind(fields))


if __name__=='__main__':
    unittest.main()

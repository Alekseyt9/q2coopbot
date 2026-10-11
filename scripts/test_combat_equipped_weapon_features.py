"""Feature-contract checks only, without model execution."""
import unittest

from combat_equipped_weapon_features import (CATEGORIES, PREFIX, VIEW_MODELS, WIDTH,
    equipped_weapon_index, equipped_weapon_tail, append_equipped_weapon, weapon_request_tail)


class EquippedWeaponFeaturesTests(unittest.TestCase):
    def test_every_label_and_exact_view_model(self):
        for index, (name, model) in enumerate(zip(CATEGORIES[1:], VIEW_MODELS), 1):
            for alias in (name, f'models/weapons/{model}/tris.md2',
                          f'MODELS\\WEAPONS\\{model.upper()}\\TRIS.MD2'):
                with self.subTest(alias=alias):
                    tail=equipped_weapon_tail(dict(weapon=alias))
                    self.assertEqual(len(tail),12)
                    self.assertEqual(sum(tail),1.)
                    self.assertEqual(tail[index],1.)

    def test_unknown_never_guesses_from_stock_or_action(self):
        for weapon in ('',None,'weapons/v_machn/tris.md2','models/weapons/v_machn/other.md2',
                       'models/weapons/v_machn/tris.md2.extra','modded railgun'):
            self.assertEqual(equipped_weapon_index(dict(weapon=weapon,
                inventory=[dict(name='Machinegun',count=1)],ammo=100,
                previous_command=dict(weapon='Machinegun'))),0)
        self.assertEqual(equipped_weapon_index({}),0)

    def test_normalizes_case_and_whitespace(self):
        self.assertEqual(equipped_weapon_index(dict(weapon='  SUPER SHOTGUN  ')),3)

    def test_three_previously_aliased_weapons_are_distinct(self):
        tails=[tuple(equipped_weapon_tail(dict(weapon=w)))
               for w in ('Machinegun','Super Shotgun','Grenades')]
        self.assertEqual(len(set(tails)),3)

    def test_prefix_preserved_exactly(self):
        prefix=[i / 1000 for i in range(PREFIX)]
        result=append_equipped_weapon(prefix,dict(weapon='Machinegun'))
        self.assertEqual(len(result),WIDTH)
        self.assertEqual(result[:PREFIX],prefix)
        self.assertEqual(prefix,[i / 1000 for i in range(PREFIX)])

    def test_request_history_is_separate_and_age_bounded(self):
        tail=weapon_request_tail(dict(weapon='Machinegun',previous_weapon_request='Shotgun',previous_weapon_request_age_frames=300))
        self.assertEqual(tail[2],1.)
        self.assertEqual(tail[12:], [1.,4.])
        self.assertEqual(sum(tail[:12]),1.)
        self.assertEqual(weapon_request_tail({})[0],1.)
        for observation in (dict(previous_weapon_request='Shotgun'),
                            dict(previous_weapon_request_age_frames=1),
                            dict(previous_weapon_request='Shotgun',previous_weapon_request_age_frames=-1),
                            dict(previous_weapon_request='modded',previous_weapon_request_age_frames=1)):
            with self.assertRaises(ValueError): weapon_request_tail(observation)

    def test_invalid_prefix_or_observed_type_rejected(self):
        for prefix in ([0.]*(PREFIX-1),[float('nan')]*PREFIX,[float('inf')]*PREFIX,[True]*PREFIX):
            with self.assertRaises(ValueError): append_equipped_weapon(prefix,{})
        for weapon in (1,False,[],{}):
            with self.assertRaises(ValueError): equipped_weapon_tail(dict(weapon=weapon))


if __name__ == '__main__':
    unittest.main()

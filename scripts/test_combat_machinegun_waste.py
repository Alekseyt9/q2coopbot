"""Native outcome accounting only; no neural policy execution."""
import unittest

from audit_combat_machinegun_waste import confirmed_waste, summarize_bucket


def shot(**changes):
    fields = dict(sky='0', fraction='0.5', damageable='1', health_before='100',
                  muzzle_blocked='0', water='0', target_class='monster_soldier')
    fields.update(changes)
    return dict(fields=fields, damage=None, eligible=True, creditable=True,
                metrics=dict(speed=0, recoil_pitch=0))


class NativeWasteAccounting(unittest.TestCase):
    def test_ambiguous_damageable_contact_is_not_a_miss(self):
        s = shot()
        self.assertFalse(confirmed_waste(s))
        s['damage'] = dict(health_before='100', health_after='92')
        result = summarize_bucket([s])
        self.assertEqual(result['counts']['live_monster_damage'], 1)
        self.assertEqual(result['creditable_confirmed_waste'], 0)

    def test_confirmed_endings_and_nonexclusive_credit(self):
        shots = [shot(sky='1'), shot(fraction='1'), shot(damageable='0'),
                 shot(health_before='0'), shot()]
        shots[0]['creditable'] = False
        result = summarize_bucket(shots)
        self.assertEqual(result['counts']['shots'], 5)
        self.assertEqual(result['confirmed_waste'], 4)
        self.assertEqual(result['creditable_confirmed_waste'], 3)
        self.assertAlmostEqual(result['proposed_reward_deltas']['-0.02'], -.06)

    def test_empty_bucket_has_no_imputed_accuracy(self):
        result = summarize_bucket([])
        self.assertIsNone(result['live_monster_damage_fraction'])
        self.assertEqual(result['creditable_confirmed_waste'], 0)


if __name__ == '__main__':
    unittest.main()

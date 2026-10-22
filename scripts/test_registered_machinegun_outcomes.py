"""Telemetry outcome classification; no NN execution."""
import unittest
from report_registered_machinegun_outcomes import outcome,summary


class OutcomeTests(unittest.TestCase):
    def shot(self,**changes):
        fields=dict(sky='0',fraction='.2',damageable='1',health_before='100',target_class='monster_infantry',muzzle_blocked='0')
        fields.update(changes)
        return dict(fields=fields,damage=dict(health_before='100',health_after='92'))

    def test_exact_outcomes_and_no_contact_priority(self):
        self.assertEqual(outcome(self.shot()),'live_monster_damage')
        for fields,expected in [(dict(sky='1'),'sky'),(dict(fraction='1'),'no_contact'),
                (dict(damageable='0'),'solid_contact'),(dict(health_before='0'),'corpse_contact'),
                (dict(target_class='misc_explobox'),'other_damage')]:
            with self.subTest(fields=fields):self.assertEqual(outcome(self.shot(**fields)),expected)

    def test_ambiguous_damageable_contacts_not_counted_as_non_damage_rays(self):
        s=self.shot();s['damage']=None
        self.assertEqual(outcome(s),'damageable_without_health_damage')
        self.assertEqual(summary([s])['instantaneous_non_damage_ray_count'],0)
        self.assertIsNone(summary([])['live_monster_damage_fraction'])


if __name__=='__main__':unittest.main()

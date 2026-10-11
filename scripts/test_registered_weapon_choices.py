"""Observed switching audit: censor gaps and exclude setup/rules attribution."""
import unittest
from audit_registered_weapon_choices import audit_frames


def frame(number,current='Grenades',request='',owner='provider',visible=True):
    return (dict(identity=dict(map='jail1',spawncount=1,connection=1,actor=1,life=1,frame=number),
                 weapon=current,enemies=[dict(clear_shot=visible)]),dict(weapon=request),owner)


class WeaponChoiceAuditTests(unittest.TestCase):
    def test_repeated_requests_keep_first_observed_delay(self):
        result=audit_frames([frame(1),frame(2,request='Machinegun'),frame(3,request='Machinegun'),
                             frame(4,current='Machinegun')])
        self.assertEqual(result['request_events'][0]['observed_delay_frames'],2)
        self.assertEqual(result['request_events'][0]['status'],'observed_equipped')
        self.assertEqual(result['counts']['request_frames'],2)

    def test_keep_streak_requires_visible_enemy_and_contiguous_frames(self):
        result=audit_frames([frame(1),frame(2),frame(3,visible=False),frame(4),frame(6),frame(7)])
        self.assertEqual(result['max_outside_pilot_keep_clear_streak'],2)
        self.assertEqual(result['counts']['outside_pilot_fire_set_keep_clear_enemy_frames'],5)

    def test_gap_and_rules_frames_censor_observed_match(self):
        for frames,status in (([frame(1,request='Machinegun'),frame(3,current='Machinegun')],'observation_gap'),
                              ([frame(1,request='Machinegun'),frame(2,owner='rules'),frame(3,current='Machinegun')],'ownership_changed')):
            with self.subTest(status=status):
                event=audit_frames(frames)['request_events'][0]
                self.assertEqual(event['status'],status)
                self.assertIsNone(event['observed_delay_frames'])

    def test_superseded_and_unfinished_are_not_successes(self):
        result=audit_frames([frame(1,request='Machinegun'),frame(2,request='Shotgun')])
        self.assertEqual([e['status'] for e in result['request_events']],['superseded','capture_ended'])

    def test_unknown_current_is_not_inferred_as_unsupported(self):
        result=audit_frames([frame(1,current='modded')])
        self.assertEqual(result['counts']['unknown_equipped_frames'],1)
        self.assertEqual(result['counts']['outside_pilot_fire_set_frames'],0)


if __name__=='__main__':unittest.main()

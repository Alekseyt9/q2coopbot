import copy
import unittest
from prepare_registered_joint_mobile_labels import explicit_targetless_release


class TargetlessReleaseTests(unittest.TestCase):
    def pair(self):
        r=dict(native_matched=True,native_exclusive=True,features=[0.]*1147)
        c=dict(observation=dict(observation_age_ms=0),teacher_aim_source='none',teacher_aim_entity=0,
            proposed=dict(attack=False),applied=dict(attack=False))
        return r,c

    def test_explicit_none_with_dispatch_is_a_release(self):
        r,c=self.pair();self.assertTrue(explicit_targetless_release(r,c))
        r['native_exclusive']=False;self.assertFalse(explicit_targetless_release(r,c))

    def test_other_aim_or_observed_target_never_becomes_none(self):
        r,c=self.pair()
        for source in ('navigation','barrel','grenade_release_pose','enemy'):
            c['teacher_aim_source']=source;self.assertFalse(explicit_targetless_release(r,c))
        c['teacher_aim_source']='none';r['features'][426]=1
        self.assertFalse(explicit_targetless_release(r,c))

    def test_guarded_attack_and_stale_frames_are_not_negative(self):
        r,c=self.pair();c['proposed']['attack']=True;self.assertFalse(explicit_targetless_release(r,c))
        c['proposed']['attack']=False;c['observation']['observation_age_ms']=301
        self.assertFalse(explicit_targetless_release(r,c))


if __name__=='__main__':unittest.main()

"""Counterfactual metadata admission only; no neural forward or game assets."""
import copy,unittest
from combat_native_teacher_registry import VERSION,validate_record

class TeacherAdmissionTest(unittest.TestCase):
    def record(self):
        return dict(version=VERSION,id='a'*64,source_model_sha256='b'*64,seed=123,frame=110,map='base3',diagnostic_only=True,ordinary_ppo_eligible=False,training_performed=False,coupled_fire_verified=True,same_continuation_policy=True,
            input=dict(feature_version='combat_features_v10',features=[0.],pre_gate_context187=[0.]*187,source_latent=[0.]*4,target=1,aim_mode=0,vertical=0,weapon=4),
            outcomes={k:dict(discounted_observed_return=v,native=dict(rules_combat_rows=0)) for k,v in (('choice0',0.),('choice1',1.))},
            target=dict(eligible_for_teacher_supervision=True,ineligible_reasons=[],preference=1,utility_delta=1.,tie_margin=.05))
    def test_rejects_future_input_and_incompatible_actuation(self):
        r=self.record();validate_record(r)
        for mutate in (
            lambda r:r['input'].update(reward=99),
            lambda r:r.update(coupled_fire_verified=False),
            lambda r:r.update(same_continuation_policy=False),
            lambda r:r.update(ordinary_ppo_eligible=True),
            lambda r:r['target'].update(preference=0),
            lambda r:r['outcomes']['choice1']['native'].update(rules_combat_rows=1),
        ):
            bad=copy.deepcopy(r);mutate(bad)
            with self.assertRaises(AssertionError):validate_record(bad)
    def test_ties_and_ineligible_cases_remain_unlabelled(self):
        r=self.record();r['outcomes']['choice1']['discounted_observed_return']=.01;r['target'].update(utility_delta=.01,preference=None);validate_record(r)
        r=self.record();r['target'].update(eligible_for_teacher_supervision=False,ineligible_reasons=['rules_owned_continuation'],preference=None);r['outcomes']['choice1']['native']['rules_combat_rows']=1;validate_record(r)

if __name__=='__main__':unittest.main()

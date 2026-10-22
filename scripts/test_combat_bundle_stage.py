import copy
import unittest
from report_combat_hypothesis_bundle import validated_stage


class BundleStageTests(unittest.TestCase):
    def fixture(self, components=False):
        labels=['control','movement','aim','full'] if components else ['control','candidate']
        variants=[dict(label=l,mode='learned',cuda_replay=True,deterministic=False) for l in labels]
        variants.append(dict(label='rules',mode='rules',cuda_replay=False,deterministic=False))
        stage=dict(version='combat_component_ablation_stage_v1' if components else 'combat_hypothesis_stage_v1',
            stage='component-ablation' if components else 'smoke',cases_per_variant=8,native_budget=len(variants)*8,
            ordinary_cases=4,recovery_cases=4,automatic_promotion=False,training_performed=False,component_labels=labels)
        spec=dict(variants=variants,**{('component_ablation' if components else 'hypothesis_stage'):stage})
        seal=dict(native_cases=len(variants)*8,cuda_cases=len(labels)*8,processed=[dict(cases=8) for _ in labels])
        return spec,seal

    def test_existing_three_way_budget(self):
        spec,seal=self.fixture();self.assertEqual(validated_stage(spec,seal)['stage'],'smoke')

    def test_component_budget_requires_all_four_replays(self):
        spec,seal=self.fixture(True);self.assertEqual(validated_stage(spec,seal)['native_budget'],40)
        bad=copy.deepcopy(seal);bad['cuda_cases']=24;bad['processed']=bad['processed'][:3]
        with self.assertRaises(AssertionError):validated_stage(spec,bad)

    def test_budget_and_component_identity_cannot_drift(self):
        spec,seal=self.fixture(True);spec['component_ablation']['component_labels'][-1]='other'
        with self.assertRaises(AssertionError):validated_stage(spec,seal)
        spec,seal=self.fixture(True);seal['native_cases']=39
        with self.assertRaises(AssertionError):validated_stage(spec,seal)


if __name__=='__main__':unittest.main()

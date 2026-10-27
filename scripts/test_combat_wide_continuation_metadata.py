"""Metadata/seed routing checks only; no neural model construction or forward."""
import copy
import unittest

from combat_continuation_probe import VERSION,validate as continuation,effective_seed
from combat_wide_aim import VERSION as WIDE_VERSION,SAMPLER,DISTRIBUTION,PROBE_VERSION,validate as wide


class WideContinuationMetadata(unittest.TestCase):
    def model(self):
        layers=[];width=187
        for n in (32,16,1):
            layers.append(dict(weight=[[0.]*width for _ in range(n)],bias=[0.]*n));width=n
        return dict(feature_version='combat_features_v13_memory_static_routes',deterministic=False,
            action_distribution=DISTRIBUTION,combat_execution_contract='stock_weapons_fire_v1',
            conditional_fire={'present':True},aim_candidate_head={'present':True},
            wide_aim=dict(version=WIDE_VERSION,sampler=SAMPLER,enabled=True,layers=layers),
            diagnostic_wide_gate_probe=dict(version=PROBE_VERSION,frame=123,choice=1),
            diagnostic_continuation_probe=dict(version=VERSION,fork_frame=124,seed=170001))

    def test_combination_is_diagnostic_and_root_precedes_fork(self):
        model=self.model();continuation(model);wide(model,allow_probe=True)
        with self.assertRaises(AssertionError):wide(model)
        for mutate in (
            lambda m:m['diagnostic_continuation_probe'].update(fork_frame=123),
            lambda m:m['diagnostic_continuation_probe'].update(fork_frame=125),
            lambda m:m.update(feature_version='unknown'),
            lambda m:m['wide_aim'].update(enabled=False),
            lambda m:m.update(diagnostic_anchor_probe={'present':True}),
        ):
            bad=copy.deepcopy(model);mutate(bad)
            with self.assertRaises(AssertionError):continuation(bad)
        model.pop('diagnostic_wide_gate_probe');continuation(model);wide(model,allow_probe=True)

    def test_effective_seed_does_not_rewrite_initial_seed(self):
        model=self.model()
        for frame,expected in ((122,7),(123,7),(124,170001),(125,170001)):
            row=dict(frame=frame,sample=dict(sampling_seed=7));before=copy.deepcopy(row)
            self.assertEqual(effective_seed(model,row),expected)
            self.assertEqual(row,before)


if __name__=='__main__':unittest.main()

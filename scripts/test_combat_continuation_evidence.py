"""Causal-input, stream-identity, ownership and manifest integrity checks."""
import copy
import tempfile
import unittest
from pathlib import Path

from combat_continuation_evidence import VERSION,INDEX_VERSION,summary,validate_record,load_index
from process_combat_architecture_pool import save,sha


class ContinuationEvidenceTests(unittest.TestCase):
    def fixture(self):
        streams=[]
        for seed,delta in ((None,-1.),(101,2.),(202,0.)):
            arms={m:dict(utility=delta if m=='choice1' else 0.,rules_combat_rows=0,
                goal=False,death=True,fork_observed=None if seed is None else True,data='test-'+m) for m in ('choice0','choice1')}
            streams.append(dict(suffix_seed=seed,fork_frame=None if seed is None else 51,outcomes=arms))
        return dict(version=VERSION,id='a'*64,source_model_sha256='b'*64,map='jail4',seed=123,frame=50,
            discount=.99,reward_version='combat_reward_v4',reward_config_sha256='c'*64,combat_execution_contract='stock_weapons_fire_v1',
            input=dict(feature_version='combat_features_v13_memory_static_routes',features=[0.,1.],source_latent=[0.]*4,
                pre_gate_context187=[0.]*187,target=1,aim_mode=0,vertical=1,weapon=2),
            streams=streams,summary=summary(streams),diagnostic_only=True,ordinary_ppo_eligible=False,training_target_emitted=False)

    def test_reject_leak_duplicate_stream_and_changed_summary(self):
        record=self.fixture();validate_record(record)
        for change in (
            lambda r:r['input'].update(future_goal=True),
            lambda r:r['streams'][2].update(suffix_seed=101),
            lambda r:r['summary'].update(mean_delta=42.),
            lambda r:r.update(ordinary_ppo_eligible=True),
        ):
            bad=copy.deepcopy(record);change(bad)
            with self.assertRaises(AssertionError):validate_record(bad)

    def test_hybrid_is_retained_without_filtered_autonomous_mean(self):
        record=self.fixture();record['streams'][1]['outcomes']['choice1']['rules_combat_rows']=1
        record['summary']=summary(record['streams']);validate_record(record)
        self.assertEqual(record['summary']['streams'],3)
        self.assertEqual(record['summary']['hybrid_streams'],1)
        self.assertIsNone(record['summary']['mean_delta'])
        self.assertIsNone(record['summary']['sample_stddev'])
        self.assertFalse(record['summary']['observed_preference_flip'])

    def test_changed_corpus_cannot_be_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);folder=repo/'workspace/artifacts/evidence';folder.mkdir(parents=True)
            record=self.fixture();path=folder/'record.json';save(path,record)
            manifest=folder/'manifest.json'
            save(manifest,dict(version=VERSION,state='complete',contexts=1,ordinary_ppo_eligible=False,training_targets_emitted=False,
                entries=[dict(id=record['id'],map=record['map'],seed=record['seed'],path=str(path),sha256=sha(path))]))
            index=folder/'index.json'
            save(index,dict(version=INDEX_VERSION,ordinary_ppo_eligible=False,training_targets_emitted=False,split_unit='map',
                corpora=[dict(manifest='workspace/artifacts/evidence/manifest.json',manifest_sha256=sha(manifest))]))
            self.assertEqual(len(load_index(repo,index)),1)
            record['streams'][0]['outcomes']['choice0']['utility']=7.;save(path,record)
            with self.assertRaises(AssertionError):load_index(repo,index)


if __name__=='__main__':unittest.main()

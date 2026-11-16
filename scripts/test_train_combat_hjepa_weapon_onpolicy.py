"""Admission boundaries and episodic credit; no CUDA or game assets required."""
import json
import pathlib
import tempfile
import unittest
import copy
import torch
from train_combat_hjepa import sha
from train_combat_hjepa_weapon_onpolicy import load_rollouts, episode_return, previous_batch, VERSION, ALGORITHM, RTG_VERSION, RTG_ALGORITHM
from train_combat_hjepa_branch_preferences import network


class RolloutAdmission(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.paths = [self.make_stream(11, True), self.make_stream(12, False)]

    def make_stream(self, stream, win):
        model = self.root / f'model-{stream}.json'
        model.write_text(json.dumps(dict(weapon_sampling_seed=stream, weapon_actor='same frozen head')))
        steps = self.root / f'steps-{stream}.jsonl'
        identity = dict(frame=98, life=1)
        steps.write_text(json.dumps(dict(index=0, owner='provider', observation=dict(identity=identity),
                                        action=dict(weapon=''), server_execution=dict(matched=True)))+'\n')
        path = self.root / f'rollout-{stream}.jsonl'
        path.write_text(json.dumps(dict(seed=123, split='train', features=[0.]*871,
            identity=identity, source=str(steps), step_index=0, slot=0,
            availability=[True]+[False]*11, behavior_probabilities=[1.]+[0.]*11,
            behavior_log_probability=0.))+'\n')
        result = dict(seed=123, capture_valid=True, generated_fixture=dict(split='train'),
                      generated_start=dict(confirmed=True,entities=[dict(health=100,**{'class':'monster_soldier'})]),
                      first_life=dict(damage=dict(received_health_damage=0,self_health_damage=0,
                                                 teammate_health_damage=0),outgoing_by_target_class=[]),
                      goal_stop=win,death_stop=False)
        meta = dict(version='hjepa_onpolicy_weapon_rollout_v1',split='train',rows=1,
                    data_sha256=sha(path),model=str(model),model_sha256=sha(model),
                    sources_sha256={str(model):sha(model),str(steps):sha(steps)},
                    episodes=[dict(seed=123,rows=1,native_result=result)])
        self.meta_path(path).write_text(json.dumps(meta))
        return path

    def meta_path(self, path):
        return pathlib.Path(str(path)+'.meta.json')

    def rewrite_row(self, mutate):
        path = self.paths[0]
        row = json.loads(path.read_text())
        mutate(row)
        path.write_text(json.dumps(row)+'\n')
        meta_path = self.meta_path(path)
        meta = json.loads(meta_path.read_text())
        meta['data_sha256'] = sha(path)
        meta_path.write_text(json.dumps(meta))

    def test_independent_group_baseline_excludes_own_return(self):
        rows, episodes, _, _ = load_rollouts(self.paths)
        self.assertEqual(len(rows),2)
        self.assertEqual(episodes[(11,123)]['advantage'],1)
        self.assertEqual(episodes[(12,123)]['advantage'],-1)

    def test_modified_export_cannot_reuse_receipt(self):
        self.paths[0].write_text(self.paths[0].read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'rollout changed'):
            load_rollouts(self.paths)

    def test_hypothetical_action_is_not_on_policy_sample(self):
        self.rewrite_row(lambda r:r.update(slot=1))
        with self.assertRaisesRegex(ValueError,'actual native choice'):
            load_rollouts(self.paths)

    def test_equal_seed_with_different_initial_observation_is_not_paired(self):
        self.rewrite_row(lambda r:r['features'].__setitem__(0,1.))
        with self.assertRaisesRegex(ValueError,'initial observations differ'):
            load_rollouts(self.paths)

    def test_validation_is_not_training(self):
        path = self.meta_path(self.paths[0])
        meta = json.loads(path.read_text());meta['split']='validation';path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,'TRAIN'):
            load_rollouts(self.paths)

    def test_native_source_change_is_rejected(self):
        (self.root/'steps-11.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'native source or receipt changed'):
            load_rollouts(self.paths)

    def test_death_overrides_goal_and_self_damage_is_penalized(self):
        meta = json.loads(self.meta_path(self.paths[0]).read_text())
        result = meta['episodes'][0]['native_result']
        self.assertEqual(episode_return(result),100)
        result['first_life']['damage']['self_health_damage']=36
        self.assertEqual(episode_return(result),64)
        result['death_stop']=True
        self.assertEqual(episode_return(result),-136)


class BatchContinuation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.parent = dict(weapon_actor=[dict(weight=[[0.,0.] for _ in range(12)],bias=[0.]*12)],kind='frozen')
        model = self.root/'model.json';model.write_text(json.dumps(self.parent))
        self.state = dict(version=VERSION,algorithm=ALGORITHM,steps=4,used_seeds=[123],
                          model_sha256=sha(model),weapon_actor=network(self.parent['weapon_actor']).state_dict(),
                          cumulative_inputs={},ancestry=[])
        self.path = self.root/'checkpoint.pt'
        torch.save(self.state,self.path)

    def test_fresh_batch_can_continue_exact_behavior_head(self):
        result = previous_batch(self.path,self.parent,[124])
        self.assertEqual(result['used_seeds'],[123])

    def test_old_world_cannot_be_reused_as_fresh(self):
        with self.assertRaisesRegex(ValueError,'already used'):
            previous_batch(self.path,self.parent,[123])

    def test_different_behavior_head_cannot_inherit_optimizer(self):
        parent = copy.deepcopy(self.parent);parent['weapon_actor'][0]['bias'][0]=1.
        with self.assertRaisesRegex(ValueError,'behavior model differs'):
            previous_batch(self.path,parent,[124])

    def test_partial_batch_is_not_next_batch(self):
        self.state['steps']=2;torch.save(self.state,self.path)
        with self.assertRaisesRegex(ValueError,'not completed'):
            previous_batch(self.path,self.parent,[124])

    def test_checkpoint_weights_must_match_serialized_behavior(self):
        self.state['weapon_actor']['0.bias'][0]=1.;torch.save(self.state,self.path)
        with self.assertRaisesRegex(ValueError,'does not bind behavior head'):
            previous_batch(self.path,self.parent,[124])

    def test_rtg_requires_explicit_credit_mode(self):
        self.state.update(version=RTG_VERSION,algorithm=RTG_ALGORITHM);torch.save(self.state,self.path)
        with self.assertRaisesRegex(ValueError,'algorithm mismatch'):
            previous_batch(self.path,self.parent,[124])
        self.assertEqual(previous_batch(self.path,self.parent,[124],'return-to-go')['version'],RTG_VERSION)

    def test_episode_to_rtg_keeps_same_completed_source(self):
        self.assertEqual(previous_batch(self.path,self.parent,[124],'return-to-go')['version'],VERSION)


if __name__ == '__main__':
    unittest.main()

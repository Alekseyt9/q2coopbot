"""Provenance checks only: no network evaluation or numerical model execution."""
import pathlib
import tempfile
import unittest

from combat_registered_retention import bindings
from process_combat_architecture_pool import read, save, sha


class RegisteredRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        model = self.root / 'weights.json'
        model.write_text('{}')
        self.plan_path = self.root / 'plan.json'
        save(self.plan_path, dict(model_path=str(model), model_sha256=sha(model),
                                 tasks=[dict(split='train', seeds=[11,12], episode=dict(id='family'))]))
        self.pool_path = self.root / 'pool.json'
        save(self.pool_path, dict(state='complete',source_unchanged=True,usable_captures=3,
             plans=[{},dict(path=str(self.plan_path),sha256=sha(self.plan_path),model_sha256=sha(model))],
             jobs=[dict(job=0,plan_index=0,task_index=0,seed=99,mode='rules',error=''),
                   dict(job=1,plan_index=1,task_index=0,seed=11,mode='learned',error=''),
                   dict(job=2,plan_index=1,task_index=0,seed=12,mode='learned',error='')]))
        processing = self.root / 'processing'
        data = processing / 'merged'
        data.mkdir(parents=True)
        for name in ('rollout','sequence'):
            (data / (name+'.jsonl')).write_text('{}\n')
        save(data/'report.json',dict(model_sha256=sha(model),rows=2,
             rollout_sha256=sha(data/'rollout.jsonl'),sequence_sha256=sha(data/'sequence.jsonl')))
        self.receipt_path = processing / 'report.json'
        save(self.receipt_path,dict(state='complete',device='cuda',pool_sha256=sha(self.pool_path),
             model_sha256=sha(model),selected_plan_index=1,rows=2,
             cases=[dict(job=1,seed=11,split='train'),dict(job=2,seed=12,split='train')]))

    def check(self):
        return bindings(self.pool_path,1,self.receipt_path)

    def test_selected_train_plan_in_multi_controller_pool(self):
        self.assertEqual(self.check()[2],{11:'family',12:'family'})

    def test_holdout_plan_is_rejected_even_with_updated_plan_hash(self):
        plan=read(self.plan_path); plan['tasks'][0]['split']='test'; save(self.plan_path,plan)
        pool=read(self.pool_path); pool['plans'][1]['sha256']=sha(self.plan_path); save(self.pool_path,pool)
        with self.assertRaisesRegex(AssertionError,'Train-only'):
            self.check()

    def test_error_in_other_controller_rejects_whole_pool(self):
        pool=read(self.pool_path); pool['jobs'][0]['error']='failed'; save(self.pool_path,pool)
        with self.assertRaisesRegex(AssertionError,'Whole pool'):
            self.check()

    def test_wrong_receipt_selection_and_missing_cases(self):
        original=read(self.receipt_path)
        for mutation in (dict(selected_plan_index=0),dict(cases=original['cases'][:1])):
            with self.subTest(mutation=mutation):
                save(self.receipt_path,dict(original,**mutation))
                with self.assertRaises(AssertionError): self.check()

    def test_changed_sequence_is_rejected(self):
        (self.receipt_path.parent/'merged/sequence.jsonl').write_text('changed')
        with self.assertRaises(AssertionError): self.check()


if __name__ == '__main__':
    unittest.main()

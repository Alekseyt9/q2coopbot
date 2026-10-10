import copy
import unittest
from process_registered_combat_pool_cuda import select_learned_jobs


class LearnedSelectionTests(unittest.TestCase):
    def test_explicit_selection_excludes_rules_but_requires_whole_pool(self):
        report=dict(state='complete',source_unchanged=True,usable_captures=3,jobs=[
            dict(job=2,plan_index=1,mode='learned',error=''),
            dict(job=0,plan_index=0,mode='rules',error=''),
            dict(job=1,plan_index=1,mode='learned',error='')])
        self.assertEqual([j['job'] for j in select_learned_jobs(report,1)],[1,2])
        for index in (None,0,2):
            with self.assertRaises(AssertionError):select_learned_jobs(report,index)
        bad=copy.deepcopy(report);bad['jobs'][1]['error']='rules capture failed'
        with self.assertRaises(AssertionError):select_learned_jobs(bad,1)
        bad=copy.deepcopy(report);bad['source_unchanged']=False
        with self.assertRaises(AssertionError):select_learned_jobs(bad,1)


if __name__=='__main__':unittest.main()

"""Small-set budgeting and stratification checks, without neural execution."""
import unittest
from prepare_combat_hypothesis_stage import STAGES, make_spec, select_families


def episodes():
    return [dict(id=f'{m}-ordinary-{i}',map=m,status='runnable',timescale=2)
            for m in ('base3','ware2','jail1','city1') for i in range(5)] + [
        dict(id=f'{m}-{i}-recovery',map=m,status='runnable',timescale=2)
        for m in ('base3','ware2','jail1','city1') for i in range(2)]


class Tests(unittest.TestCase):
    def test_exact_stage_budgets_and_map_coverage(self):
        for stage, budget in zip(STAGES, (8,24,56,112)):
            ids, count = select_families(episodes(), stage, 17)
            self.assertEqual(len(ids)*count,budget)
            self.assertEqual(len(ids),len(set(ids)))
            self.assertEqual({s.split('-')[0] for s in ids},{'base3','ware2','jail1','city1'})
            self.assertEqual(ids,select_families(list(reversed(episodes())),stage,17)[0])

    def test_strata_not_replaced_by_easy_only_selection(self):
        ids,count=select_families(episodes(),'screen',17)
        self.assertEqual(sum(s.endswith('-recovery') for s in ids)*count,8)
        with self.assertRaises(AssertionError):select_families(episodes()[:20],'screen',17)

    def test_matching_modes_and_no_mutation_of_template(self):
        t=dict(version='registered_combat_model_comparison_v1',variants=[
            dict(label='control',mode='learned',deterministic=False),
            dict(label='candidate',mode='learned',deterministic=False),dict(label='rules',mode='rules')])
        s=make_spec(t,episodes(),'smoke','control','candidate',468,17,41700)
        self.assertEqual(s['hypothesis_stage']['native_budget'],24)
        self.assertEqual(s['split'],'validation')
        confirmation=make_spec(t,episodes(),'confirmation','control','candidate',476,17,41700)
        self.assertEqual(confirmation['split'],'confirmation')
        self.assertFalse(s['hypothesis_stage']['automatic_promotion'])
        self.assertNotIn('episodes',t)
        t['variants'][1]['deterministic']=True
        with self.assertRaises(AssertionError):make_spec(t,episodes(),'smoke','control','candidate',468,17,41700)


if __name__=='__main__':unittest.main()

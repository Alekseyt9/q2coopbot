import copy
import unittest
from prepare_combat_replay_cuda import adjacent_rows


def fixture():
    context = [dict(seed=1, index=i, frame=10+i, features=[i], memory=dict(reset=i==0)) for i in range(2)]
    row = dict(seed=1,index=0,frame=10,next_frame=11,features=[0],next_features=[1],terminal=False,truncated=False,bootstrap_zero=False,next_reset=False)
    return row, context


class ReplayBoundaries(unittest.TestCase):
    def test_missing_successor_does_not_invent_terminal(self):
        row, context = fixture()
        terminal = copy.deepcopy(row); terminal.update(index=1,frame=11,next_frame=12,features=[1],terminal=True,bootstrap_zero=True)
        row.update(index=2,frame=12,next_frame=13,features=[2])
        context.append(dict(seed=1,index=2,frame=12,features=[2],memory=dict(reset=False)))
        pairs, missing = adjacent_rows([terminal,row], context)
        self.assertEqual(pairs, [(0,1,1,False)])
        self.assertEqual(missing, [(1,2)])

    def test_causal_successor(self):
        row, context = fixture()
        self.assertEqual(adjacent_rows([row], context), ([(0,0,1,True)], []))

    def test_rejects_other_observation_and_reset(self):
        for mode in ('features','reset','frame'):
            row, context = fixture()
            context[1]['features' if mode=='features' else 'frame'] = [99] if mode=='features' else 99
            if mode=='reset': context[1]['frame']=11;context[1]['memory']['reset']=True
            with self.assertRaises(AssertionError): adjacent_rows([row], context)

    def test_zero_bootstrap_requires_explicit_boundary(self):
        row, context = fixture(); row['bootstrap_zero']=True
        with self.assertRaises(AssertionError): adjacent_rows([row], context)


if __name__=='__main__': unittest.main()

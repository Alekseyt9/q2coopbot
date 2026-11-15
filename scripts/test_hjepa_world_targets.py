import json
import pathlib
import tempfile
import unittest
from train_combat_hjepa import load,sha


class WorldTargetsTest(unittest.TestCase):
    def test_policy_trajectories_stay_separate_and_death_target_is_trained(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);source=root/'source';source.write_text('native fixture')
            path=root/'rows.jsonl';rows=[]
            for namespace in ('policy-a','policy-b'):
                for i in range(16):
                    x=[1.]+[0.]*844;next_x=x.copy()
                    if i==15:next_x[0]=-.02
                    rows.append(dict(episode='seed-1@'+namespace,frame=i+1,features=x,next_features=next_x,
                                     action=[0.,0.,0.,0.,1.,1.,0.,0.],utility=i*.01,
                                     next_utility=-5. if i==15 else (i+1)*.01,behavior_weight=0.))
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            pathlib.Path(str(path)+'.meta.json').write_text(json.dumps(dict(version='hjepa_data_v1',feature_version='combat_features_v6',rows=32,sources_sha256={str(source):sha(source)},data_sha256=sha(path))))
            (x,actions,utility,behavior),ids,_=load(path,'cpu')
            self.assertEqual(tuple(x.shape),(2,17,845))
            self.assertEqual(tuple(utility.shape),(2,17))
            self.assertTrue((utility[:,-1]==-5).all())
            self.assertAlmostEqual(float(x[0,-1,0]),-.02,places=6)
            self.assertEqual(len(ids),2)
            self.assertEqual(float(behavior.sum()),0.)


if __name__=='__main__':unittest.main()

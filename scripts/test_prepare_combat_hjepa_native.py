import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

from prepare_combat_hjepa_native import native_increment, sha


class NativeLabelsTest(unittest.TestCase):
    def fixture(self, root, missing=False):
        identity = dict(map='base1',connection=1,spawncount=2,actor=0,life=1,frame=10)
        steps, outcomes, rows = [], [], []
        for i in range(3):
            ident = dict(identity, frame=10+i)
            obs = dict(identity=ident,ammo=100-i,weapon='Machinegun')
            native = dict(spawncount=2,actor=0,begin_frame=10+i,end_frame=11+i)
            steps.append(dict(worker='w',episode='seed-1',index=i,observation=obs,
                              next_observation=dict(obs,ammo=99-i),native_step=native,owner='rules'))
            outcomes.append(dict(worker='w',episode='seed-1',step=i,available=not(missing and i==1),
                                 version='server_step_effects_v1',monster_health_damage=10,
                                 monster_kills=int(i==1),received_health_damage=0,self_health_damage=0,
                                 teammate_health_damage=0,deaths=0))
            rows.append(dict(episode='seed-1:'+json.dumps(dict(identity,frame=0)),frame=10+i,utility=-999,
                             features=[0]*845,next_features=[0]*845,action=[0,0,0,0,0,1,0,0]))
        for name, records in [('steps.jsonl',steps),('server_outcomes.jsonl',outcomes),('raw.jsonl',rows)]:
            (root/name).write_text('\n'.join(json.dumps(r) for r in records))
        (root/'report.json').write_text(json.dumps(dict(command_proof=dict(accepted=True),synchronous_confirmed=True,observed_reset_confirmed=True)))
        (root/'raw.jsonl.meta.json').write_text(json.dumps(dict(version='hjepa_data_v1',rows=3,
            feature_version='combat_features_v6',sources_sha256={str(root/'steps.jsonl'):sha(root/'steps.jsonl')})))

    def run_prepare(self, root, terminal=False):
        script=pathlib.Path(__file__).with_name('prepare_combat_hjepa_native.py')
        command=[sys.executable,str(script),'--data',str(root/'raw.jsonl'),'--out',str(root/'native.jsonl')]
        if terminal:command.append('--terminal-targets')
        result=subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        return [json.loads(line) for line in (root/'native.jsonl').read_text().splitlines()]

    def test_labels_only_use_past_effects(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);self.fixture(root);rows=self.run_prepare(root)
            self.assertEqual(len(rows),3)
            self.assertEqual(rows[0]['utility'],0)
            self.assertAlmostEqual(rows[1]['utility'],.015)
            self.assertAlmostEqual(rows[2]['utility'],2.03)
            self.assertTrue(all(r['behavior_weight']==1 for r in rows))
            self.assertTrue(all(r['features']==[0]*845 for r in rows))

    def test_unavailable_native_effect_breaks_future_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);self.fixture(root,missing=True);rows=self.run_prepare(root)
            self.assertEqual([r['frame'] for r in rows],[10])

    def test_negative_native_counts_rejected(self):
        with self.assertRaises(ValueError):
            native_increment(dict(monster_health_damage=-1,monster_kills=0,received_health_damage=0,self_health_damage=0,teammate_health_damage=0,deaths=0),0)

    def test_death_cost_reaches_next_state_target_without_changing_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);self.fixture(root)
            p=root/'server_outcomes.jsonl'
            outcomes=[json.loads(line) for line in p.read_text().splitlines()]
            outcomes[1]['deaths']=1
            p.write_text('\n'.join(json.dumps(r) for r in outcomes))
            rows=self.run_prepare(root,terminal=True)
            self.assertAlmostEqual(rows[1]['utility'],.015)
            self.assertAlmostEqual(rows[1]['next_utility'],-2.97)
            self.assertEqual(rows[1]['features'],[0]*845)
            self.assertEqual(rows[1]['next_features'],[0]*845)


if __name__=='__main__':
    unittest.main()

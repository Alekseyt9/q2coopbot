"""Scientific audit boundaries: first life, provider ownership and actual HP."""
import json
import pathlib
import tempfile
import unittest
from report_combat_hjepa_categorical import native_cycles
from run_combat_hjepa_actor_curriculum import immutable_json


class NativeCycles(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        case = self.root/'policy'/'case-0';dataset = case/'worker-0-episode-0'/'dataset'
        dataset.mkdir(parents=True);self.case,self.dataset = case,dataset
        result = dict(worker=0,episode=0,first_life=dict(start_frame=10,end_frame=12,bot_entity=1,
            damage=dict(self_health_damage=5)))
        (case/'report.json').write_text(json.dumps(dict(results=[result])))
        rows = []
        for frame in [9,10,11,12]:
            rows.append(dict(owner='rules' if frame==11 else 'provider',
                observation=dict(identity=dict(frame=frame),weapon='models/weapons/v_handgr/tris.md2',gun_frame=11),
                server_execution=dict(window_exclusive=True),action=dict(attack=False),applied_action=dict(attack=False)))
        (dataset/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        events = [dict(attacker=1,target=1,frame=11,health_before=3,take=10,mod=24),
                  dict(attacker=1,target=1,frame=11,health_before=2,take=2,mod=9),
                  dict(attacker=1,target=1,frame=9,health_before=90,take=80,mod=24)]
        (dataset/'server_outcomes.jsonl').write_text(json.dumps(dict(events=events))+'\n')

    def test_only_exclusive_provider_decisions_before_endpoint(self):
        totals, sources = native_cycles(self.root,'policy')
        self.assertEqual(totals['hold_decisions'],1)
        self.assertEqual(totals['proposed_releases'],1)
        self.assertEqual(totals['applied_releases'],1)
        self.assertEqual(len(sources),2)

    def test_overkill_and_outside_first_life_do_not_inflate_held_damage(self):
        totals,_ = native_cycles(self.root,'policy')
        self.assertEqual(totals['held_grenade_hp'],3)
        self.assertEqual(totals['held_grenade_events'],1)

    def test_missing_native_self_damage_is_rejected(self):
        path = self.case/'report.json';report = json.loads(path.read_text())
        report['results'][0]['first_life']['damage']['self_health_damage'] = 4
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'ledger mismatch'):
            native_cycles(self.root,'policy')

    def test_stage_resume_cannot_replace_existing_inputs(self):
        path = self.root/'prepared.json'
        immutable_json(path,dict(seed=1));immutable_json(path,dict(seed=1))
        with self.assertRaisesRegex(ValueError,'changed'):
            immutable_json(path,dict(seed=2))
        self.assertEqual(json.loads(path.read_text()),dict(seed=1))


if __name__ == '__main__':
    unittest.main()

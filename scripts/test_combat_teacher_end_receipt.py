import copy
import json
import tempfile
import unittest
from pathlib import Path

from combat_teacher_end_receipt import admitted_terminal_receipt_v3
from process_combat_architecture_pool import save,sha


class NativeEndReceiptTests(unittest.TestCase):
    def fixture(self,root):
        rows=[dict(frame=f,map='base1',spawncount=4,self_entity=1,arbitration=dict(move_source='policy'),
                   enemies=[],sent_command=dict(Buttons=0),self=[0,0,0]) for f in (1,2,3)]
        (root/'bot.jsonl').write_text('\n'.join(map(json.dumps,rows)),encoding='utf-8')
        result=dict(root=str(root),death_stop=None,goal_stop=None,first_life=dict(start_frame=1,end_frame=3,
            end_reason='first_observed_death',minimum_health=0,map='base1',spawncount=4,bot_entity=1,
            aim_by_class=[],movement=[dict(source='policy',command_frames=3,actual_next_frame_steps=0)]))
        identity=dict(map='base1',spawncount=4,actor=1,life=1,frame=2)
        step=dict(index=7,terminal=True,truncated=False,reason='observed_death',
            observation=dict(identity=identity,health=1),next_observation=dict(identity=dict(identity,frame=3),health=0),
            native_step=dict(begin_frame=2,end_frame=3,actor=1,spawncount=4,damage_indexes=[1]),
            server_execution=dict(matched=True,window_exclusive=True,recovery_commands=0))
        effect=dict(version='server_step_effects_v1',step=7,available=True,deaths=1,
            events=[dict(target=1,target_class='player',map='base1',spawncount=4,health_before=1,health_after=0,frame=3)])
        for name,row in (('steps.jsonl',step),('server_outcomes.jsonl',effect)):
            (root/name).write_text(json.dumps(row)+'\n',encoding='utf-8')
        save(root/'report.json',dict(source_sha256={str(root/name):sha(root/name) for name in ('steps.jsonl','server_outcomes.jsonl')}))
        return result

    def test_missing_optional_tag_requires_exact_native_death(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);result=self.fixture(root)
            receipt,proof=admitted_terminal_receipt_v3(result,2,root)
            self.assertFalse(proof['native_death']['original_stop_tag_present'])
            present=copy.deepcopy(result);present['death_stop']=receipt['death_stop']
            self.assertEqual(receipt,admitted_terminal_receipt_v3(present,2,root)[0])
            bad=copy.deepcopy(present);bad['death_stop']['health']=-1
            with self.assertRaises(AssertionError):admitted_terminal_receipt_v3(bad,2,root)
            path=root/'server_outcomes.jsonl';path.write_text('{}\n',encoding='utf-8')
            with self.assertRaises(AssertionError):admitted_terminal_receipt_v3(result,2,root)


if __name__=='__main__':unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from combat_teacher_movement_receipt import admitted_terminal_receipt_v2


class MovementReceiptTests(unittest.TestCase):
    def fixture(self, root):
        rows = [dict(frame=f,map='mine3',spawncount=4,self_entity=1,
                     arbitration=dict(move_source='prepare' if f==1 else 'policy'),
                     enemies=[],sent_command=dict(Buttons=0),self=[3*(f-1),0,0]) for f in (1,2,3)]
        self.write(root,rows)
        result = dict(root=str(root),goal_stop=None,death_stop=None,
            first_life=dict(start_frame=1,end_frame=3,map='mine3',spawncount=4,bot_entity=1,
                aim_by_class=[],movement=[dict(source='prepare',command_frames=1,actual_next_frame_steps=1),
                                         dict(source='policy',command_frames=2,actual_next_frame_steps=1)]))
        return rows,result

    def write(self,root,rows):
        (root/'bot.jsonl').write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')

    def test_only_preparation_movement_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,result=self.fixture(root)
            receipt,proof=admitted_terminal_receipt_v2(result,2)
            self.assertEqual(receipt['first_life']['movement'],[dict(source='policy',command_frames=2,actual_next_frame_steps=1)])
            self.assertEqual(proof['preparation_movement'][0]['command_frames'],1)
            rows[2]['self']=[3,0,0];self.write(root,rows)
            result['first_life']['movement'][1]['actual_next_frame_steps']=0
            other,_=admitted_terminal_receipt_v2(result,2)
            self.assertNotEqual(receipt,other)

    def test_unexplained_movement_total_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,result=self.fixture(root)
            result['first_life']['movement'][1]['command_frames']=3
            with self.assertRaises(AssertionError):
                admitted_terminal_receipt_v2(result,2)


if __name__=='__main__':
    unittest.main()

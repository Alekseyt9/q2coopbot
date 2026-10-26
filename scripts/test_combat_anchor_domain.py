"""Metadata-only regression checks; no neural execution."""
import copy
import unittest
from recover_combat_anchor_screen import first_life,terminal_receipt
from finish_combat_anchor_probe import first_diff,normalized


class AnchorDomainTest(unittest.TestCase):
    def fixture(self):
        r=dict(first_life=dict(start_frame=98,end_frame=101,last_frames=[dict(frame=101,visible_enemies=[dict(id=2,health=8),dict(id=1,health=3)])]),
            goal_stop=dict(observed_frame=101,native_objective=dict(frame=101,alive=0,pending=0)),death_stop=None)
        rows=[dict(observation=dict(identity=dict(life=1,frame=f),health=100),
            server_execution=dict(matched=True),sent_command=dict(Yaw=f)) for f in range(98,105)]
        return r,rows

    def test_debug_grenade_order_only(self):
        r,_=self.fixture();r['first_life']['last_frames'][0]['grenades']=[dict(id=7,origin=[1,2,3]),dict(id=6,origin=[4,5,6])]
        v=copy.deepcopy(r);v['first_life']['last_frames'][0]['grenades'].reverse()
        self.assertEqual(terminal_receipt(r),terminal_receipt(v))
        v['first_life']['last_frames'][0]['grenades'][0]['origin'][0]+=1
        self.assertNotEqual(terminal_receipt(r),terminal_receipt(v))

    def test_only_post_terminal_tail_removed(self):
        r,rows=self.fixture();other=copy.deepcopy(rows[:5]);other[-1]['sent_command']['Yaw']=999
        self.assertEqual(first_life(r,rows),first_life(r,other))
        other[1]['sent_command']['Yaw']=123
        self.assertIsNotNone(first_diff(first_life(r,rows),first_life(r,other)))
        r['goal_stop']['native_objective']['pending']=1
        with self.assertRaises(AssertionError):first_life(r,rows)

    def test_debug_order_only_and_all_ids_retained(self):
        r,rows=self.fixture();v=copy.deepcopy(r)
        v['first_life']['last_frames'][0]['visible_enemies'].reverse()
        self.assertEqual(terminal_receipt(r),terminal_receipt(v))
        self.assertNotEqual(r,v)  # Receipt preparation must not mutate inputs.
        v['first_life']['last_frames'][0]['visible_enemies'][0]['id']=3
        self.assertNotEqual(terminal_receipt(r),terminal_receipt(v))
        v=copy.deepcopy(r);v['first_life']['last_frames'][0]['visible_enemies'][0]['health']=7
        self.assertNotEqual(terminal_receipt(r),terminal_receipt(v))
        # Actual observations retain order: sorting is never part of normalization.
        a=copy.deepcopy(rows[0]);b=copy.deepcopy(a)
        a['observation']['enemies']=[dict(id=2),dict(id=1)]
        b['observation']['enemies']=[dict(id=1),dict(id=2)]
        self.assertIsNotNone(first_diff(normalized(a),normalized(b)))


if __name__=='__main__':unittest.main()

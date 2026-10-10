"""Protect distinctions between actor requests, final guards and observations."""
import collections,unittest
from report_combat_visible_target_decisions import count_row

def row(owner='provider',attack=False,sent=False,life=1,health=100):
    return dict(owner=owner,server_execution=dict(matched=True),observation=dict(identity=dict(life=life),health=health,enemies=[dict(clear_shot=True)],velocity=[0,0,0]),action=dict(attack=attack,target_entity=7,forward=1,side=0,vertical='crouch'),applied_action=dict(attack=sent,forward=0,side=0))

class DecisionsTest(unittest.TestCase):
    def test_actor_withholds_fire_distinct_from_guard(self):
        counts=collections.Counter();count_row(row(),1,counts)
        self.assertEqual(counts['clear_no_attack_requested'],1)
        self.assertEqual(counts['clear_request_blocked'],0)
        count_row(row(attack=True,sent=False),1,counts)
        self.assertEqual(counts['clear_no_attack_requested'],1)
        self.assertEqual(counts['clear_request_blocked'],1)
        self.assertEqual(counts['clear_move_reduced'],2)
    def test_no_clear_target_and_later_life_do_not_contaminate(self):
        counts=collections.Counter();x=row();x['observation']['enemies'][0]['clear_shot']=False
        count_row(x,1,counts);count_row(row(life=2),1,counts);count_row(row(health=0),1,counts)
        self.assertEqual(counts['frames'],1);self.assertEqual(counts['clear_frames'],0)
    def test_rules_missing_target_is_not_provider_no_target(self):
        counts=collections.Counter();x=row(owner='rules',attack=True,sent=True);x['action'].pop('target_entity')
        count_row(x,1,counts)
        self.assertEqual(counts['provider_clear_without_selected_target'],0)
        self.assertEqual(counts['clear_attack_sent'],1)
    def test_unmatched_native_dispatch_rejected(self):
        x=row();x['server_execution']['matched']=False
        with self.assertRaises(AssertionError):count_row(x,1,collections.Counter())

if __name__=='__main__':unittest.main()

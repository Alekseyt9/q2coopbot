"""Label geometry tests; no model execution."""
import copy,unittest
from export_combat_machinegun_aim import labels
from report_combat_machinegun_aim import measure

class LabelTests(unittest.TestCase):
    def setUp(self):
        self.shot={'fields':{'start':'0,0,0','recoil':'-13.5,0,0'}}
        self.step={'owner':'provider','interventions':[], 'observation':{'position':[0,0,0],'view_angles':[0,0,0], 'kick_angles_degrees':[-2,0,0], 'enemies':[{'id':4,'distance':100,'relative':[100,0,-22],'observed_solid':2|(3<<5)|(8<<10),'observed_track':3,'clear_shot':True}]}}
    def test_actual_fresh_recoil_is_label_only(self):
        before=copy.deepcopy(self.step);q=labels(self.shot,self.step)[0]
        self.assertAlmostEqual(q['pitch_delta_degrees'],13.5);self.assertAlmostEqual(q['yaw_delta_degrees'],0)
        self.assertEqual(self.step,before)
    def test_actual_muzzle_translation_changes_geometry(self):
        self.shot['fields']['start']='0,10,0'
        self.assertLess(labels(self.shot,self.step)[0]['yaw_delta_degrees'],-5)
    def test_camera_kick_is_not_fresh_weapon_recoil(self):
        self.step['observation']['kick_angles_degrees']=[12,8,3]
        self.assertAlmostEqual(labels(self.shot,self.step)[0]['pitch_delta_degrees'],13.5)
    def test_unknown_bbox_is_masked(self):
        self.step['observation']['enemies'][0]['observed_solid']=0
        self.assertEqual(labels(self.shot,self.step),[])
    def test_occluded_target_is_masked(self):
        self.step['observation']['enemies'][0]['clear_shot']=False
        self.assertEqual(labels(self.shot,self.step),[])
    def test_guarded_pitch_is_masked(self):
        self.step['interventions']=[{'component':'pitch'}]
        self.assertEqual(labels(self.shot,self.step),[])
    def test_absent_empty_interventions_is_supported(self):
        del self.step['interventions']
        self.assertEqual(len(labels(self.shot,self.step)),1)
    def test_no_target_is_not_replaced_with_visible_enemy(self):
        self.step['action']={'target_entity':0}
        self.assertEqual(measure(self.shot,self.step)['status'],'provider_no_target_with_visible_bbox')
    def test_unknown_bbox_no_target_remains_distinct(self):
        self.step['action']={'target_entity':0}
        self.step['observation']['enemies'][0]['observed_solid']=0
        self.assertEqual(measure(self.shot,self.step)['status'],'provider_no_target_without_visible_bbox')

if __name__=='__main__':unittest.main()

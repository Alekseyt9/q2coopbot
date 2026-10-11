"""Boundary/censoring tests for observed phases; no model execution."""
import unittest
from audit_registered_weapon_phases import observed_phase,audit

def row(frame,weapon='Machinegun',gun=6,request='',owner='provider'):
    observation=dict(identity=dict(map='base3',spawncount=1,connection=1,actor=1,life=1,frame=frame),
        weapon=weapon,gun_frame=gun,enemies=[])
    return observation,dict(attack=True,weapon=request),owner

class Phases(unittest.TestCase):
    def test_boundaries_not_shots(self):
        self.assertEqual(observed_phase('Machinegun',3),'activation_animation')
        self.assertEqual(observed_phase('Machinegun',4),'firing_animation')
        self.assertEqual(observed_phase('Machinegun',45),'idle_animation')
        self.assertEqual(observed_phase('Machinegun',46),'lowering_animation')
        self.assertEqual(observed_phase('Grenades',16),'unsupported_weapon')
        result=audit([row(98,gun=3),row(99,gun=4)])
        self.assertNotIn('shots',result['counts'])
        self.assertEqual(result['counts']['applied_attack/activation_animation'],1)
        alias=audit([row(98,weapon='models/weapons/v_machn/tris.md2',gun=4)])
        self.assertEqual(alias['counts']['phase/firing_animation'],1)
    def test_observed_change_and_interrupted_request(self):
        result=audit([row(98,request='Shotgun'),row(99,weapon='Shotgun',gun=0),row(100,weapon='Shotgun',gun=1)])
        self.assertEqual(result['counts']['observed_equipped_changes'],1)
        self.assertEqual(result['weapon_requests']['request_events'][0]['status'],'observed_equipped')
        result=audit([row(98,request='Shotgun'),row(99,request='Blaster')])
        self.assertEqual(result['weapon_requests']['request_events'][0]['status'],'superseded')
    def test_gaps_and_ownership_do_not_invent_transitions(self):
        result=audit([row(98),row(100,weapon='Shotgun',gun=0),row(101,weapon='Blaster',owner='rules'),row(102)])
        self.assertEqual(result['counts'].get('observed_equipped_changes',0),0)
        with self.assertRaises(AssertionError): audit([row(98),row(98)])

if __name__=='__main__': unittest.main()

"""Metadata tests for release clocks and intent/execution distinctions; no NN."""
import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]/'workspace/artifacts'
ROOT.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('executed',pathlib.Path(__file__).with_name('audit_registered_execution_actions.py'))
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def row(frame,owner='provider',forward=.3331,applied=.3325,attack=False,blocked=False):
    identity=dict(map='jail1',spawncount=7,connection=1,actor=1,life=1,frame=frame)
    def action(fire,movement):
        return dict(identity=identity,forward=movement,side=0,yaw_delta_degrees=0,pitch_delta_degrees=0,attack=fire,vertical='release')
    return dict(combat_policy=dict(selection=dict(owner=owner,interventions=[]),
        observation=dict(identity=identity,health=100,weapon='Blaster',on_ground=False,
            velocity=[200,0,0],enemies=[dict(clear_shot=True)]),
        proposed=action(attack,forward),applied=action(attack and not blocked,applied)))

class Tests(unittest.TestCase):
    def audit(self,rows):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path=pathlib.Path(directory)/'trace.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            return module.audit(path,98,100,7)

    def test_first_released_frames_and_rounding(self):
        result=self.audit([row(97),row(98),row(99),row(100)])
        self.assertEqual(result['counts']['provider_frames'],3)
        self.assertEqual(result['counts']['legacy_native_frame_cutoff_provider_frames'],1)
        self.assertEqual(result['counts']['movement_changed_beyond_rounding_frames'],0)

    def test_blocked_intent_does_not_imply_stationary_or_shot(self):
        blocked=row(98,forward=.5,applied=0,attack=True,blocked=True)
        blocked['combat_policy']['selection']['interventions']=[dict(component='attack',reason='pilot_fixed_weapon'),dict(component='movement',reason='unsupported_motion_guard')]
        result=self.audit([blocked,row(99,forward=0,applied=0)])
        self.assertEqual(result['counts']['attack_suppressed_frames'],1)
        self.assertEqual(result['counts']['applied_attack_frames'],0)
        self.assertEqual(result['counts']['movement_command_fully_suppressed_frames'],1)
        self.assertEqual(result['counts']['airborne_movement_suppressed_frames'],1)
        self.assertEqual(result['reasons']['attack/pilot_fixed_weapon'],1)
        self.assertNotIn('stationary_frames',result['counts'])
        self.assertNotIn('shots',result['counts'])

    def test_early_rules_are_visible_and_duplicates_rejected(self):
        result=self.audit([row(98,owner='rules')])
        self.assertEqual(result['counts']['visible_enemy_rules_frames'],1)
        with self.assertRaises(AssertionError):
            self.audit([row(98),row(98,owner='rules')])
        with self.assertRaises(AssertionError):
            self.audit([row(98,forward=float('nan'))])

if __name__=='__main__':
    unittest.main()

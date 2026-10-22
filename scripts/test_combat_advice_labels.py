import copy
import unittest

from prepare_combat_advice_labels import label


def example():
    identity = dict(map='base3',connection=1,spawncount=2,actor=1,life=1,frame=101)
    return dict(sent_command=dict(Up=0),
        combat_policy=dict(selection=dict(owner='provider'),observation=dict(identity=identity,
            health=100,observation_age_ms=0,on_ground=True,ducked=False,enemies=[])),
        combat_advice=dict(version='reactive_combat_advice_v2',identity=identity,available=True,
            recommended_action=dict(identity=identity,weapon='',forward=0,side=0,attack=False,
                yaw_delta_degrees=0,pitch_delta_degrees=-30),
            recommended_command=dict(Forward=0,Side=0,Up=0,Buttons=0,Impulse=0,Msec=100),
            decision=dict(aim_source='none')))


class AdviceLabelsTest(unittest.TestCase):
    def test_targetless_corrective_aim_is_preserved(self):
        row = example(); before = copy.deepcopy(row); value = label(row)
        self.assertEqual(value['ground']['aim_degrees'],[0,-30])
        self.assertEqual(value['ground']['target'],0)
        self.assertTrue(value['decision']['targetless_release'])
        self.assertEqual(row,before)

    def test_airborne_or_vertical_mismatch_keeps_only_decisions(self):
        for field in ('airborne','learner_jump','teacher_crouch'):
            row = example()
            if field == 'airborne': row['combat_policy']['observation']['on_ground'] = False
            if field == 'learner_jump': row['sent_command']['Up'] = 400
            if field == 'teacher_crouch': row['combat_advice']['recommended_command']['Up'] = -400
            value = label(row)
            self.assertIsNone(value['ground'])
            self.assertTrue(value['decision']['attack_mask'])

    def test_stale_target_and_setup_are_not_teacher_samples(self):
        row = example(); row['combat_policy']['selection']['owner'] = 'rules'
        self.assertIsNone(label(row))
        row = example(); row['combat_advice']['available'] = False
        self.assertIsNone(label(row))
        row = example(); row['combat_advice']['recommended_action'].update(target_entity=3,target_track=1)
        with self.assertRaises(AssertionError): label(row)


if __name__ == '__main__': unittest.main()

import unittest

from audit_registered_joint_teacher import previous_intent, target_choice


def identity(frame=100, life=1):
    return dict(map='base3', connection=1, spawncount=4, actor=0, life=life, frame=frame)


class TeacherHistoryTests(unittest.TestCase):
    def test_past_only_and_gap_life_reset(self):
        past = dict(identity=identity(), entity=12, track=2, slot=1)
        self.assertEqual(previous_intent(past, dict(identity=identity(101)))['entity'], 12)
        for current in (identity(100), identity(102), identity(101, 2)):
            self.assertIsNone(previous_intent(past, dict(identity=current)))

    def test_explicit_target_stable_slots_and_unknown_track(self):
        policy = dict(teacher_aim_source='enemy', teacher_aim_entity=12,
            observation=dict(identity=identity(), enemies=[
                dict(id=11, distance=30, clear_shot=True, observed_track=1),
                dict(id=12, distance=30, clear_shot=True, observed_track=2)]))
        self.assertEqual(target_choice(policy)[0]['slot'], 2)
        policy['observation']['enemies'][1]['observed_track'] = None
        self.assertEqual(target_choice(policy), (None, 'enemy_track_unknown'))

    def test_no_nearest_target_or_false_none_label(self):
        policy = dict(teacher_aim_source='navigation', teacher_aim_entity=0,
            observation=dict(identity=identity(), enemies=[dict(id=11, distance=1)]))
        self.assertEqual(target_choice(policy), (None, 'non_enemy_aim'))
        policy.update(teacher_aim_source='enemy', teacher_aim_entity=99)
        self.assertEqual(target_choice(policy), (None, 'enemy_outside_observed_slots'))


if __name__ == '__main__':
    unittest.main()

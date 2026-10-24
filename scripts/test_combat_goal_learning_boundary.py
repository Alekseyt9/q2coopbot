import unittest

from audit_combat_blocked_ground_native_credit import goal_learning_boundary


class GoalBoundaryTests(unittest.TestCase):
    def fixture(self):
        goal=dict(kill_frame=10,observed_frame=12)
        rows=[dict(index=3,next_frame=10,bootstrap_zero=True)]
        steps={3:dict(reason='control_handoff',terminal=False,goal_handoff_verified=True,
                      native_step=dict(end_frame=10),next_observation=dict(health=50))}
        return goal,rows,steps

    def test_verified_finite_task_ends_before_observation_latency(self):
        goal,rows,steps=self.fixture()
        boundary=goal_learning_boundary(goal,rows,steps)
        self.assertEqual(boundary,10)
        self.assertEqual([f for f in (9,10,11,12,13) if f>boundary],[11,12,13])

    def test_unconfirmed_or_bootstrapped_handoff_does_not_move_boundary(self):
        goal,rows,steps=self.fixture()
        steps[3]['goal_handoff_verified']=False
        self.assertEqual(goal_learning_boundary(goal,rows,steps),12)
        steps[3]['goal_handoff_verified']=True;rows[0]['bootstrap_zero']=False
        self.assertEqual(goal_learning_boundary(goal,rows,steps),12)
        self.assertIsNone(goal_learning_boundary(None,rows,steps))

    def test_bad_goal_endpoints_fail_closed(self):
        for frame,health in ((9,50),(13,50),(10,0)):
            goal,rows,steps=self.fixture()
            rows[0]['next_frame']=steps[3]['native_step']['end_frame']=frame
            steps[3]['next_observation']['health']=health
            with self.subTest(frame=frame,health=health),self.assertRaises(AssertionError):
                goal_learning_boundary(goal,rows,steps)


if __name__=='__main__':
    unittest.main()

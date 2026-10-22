import copy
import unittest
from audit_registered_stationary_transitions import classify


class StationaryTimingTests(unittest.TestCase):
    def pair(self):
        identity=dict(map='base3',connection=1,spawncount=3,actor=0,life=1,frame=100)
        before=dict(observation=dict(identity=identity,health=100,velocity=[0,0,0],position=[0,0,0]),
            proposed=dict(forward=1,side=0,vertical='release'),applied=dict(forward=0,side=0,vertical='release'),
            selection=dict(interventions=[dict(component='movement',reason='no_ground_support')]))
        after=copy.deepcopy(before);after['observation']['identity']['frame']=101
        after['observation']['enemies']=[dict(clear_shot=True)]
        return before,after

    def test_command_is_associated_with_next_not_current_speed(self):
        before,after=self.pair();before['observation']['velocity']=[200,0,0]
        self.assertEqual(classify(before,after)[0],'suppressed_motion_next_stationary')
        before['observation']['velocity']=[0,0,0];after['observation']['velocity']=[200,0,0]
        self.assertIsNone(classify(before,after))

    def test_no_cross_gap_or_life_attribution(self):
        before,after=self.pair();after['observation']['identity']['frame']=102
        self.assertIsNone(classify(before,after))
        after['observation']['identity']['frame']=101;after['observation']['identity']['life']=2
        self.assertIsNone(classify(before,after))


if __name__=='__main__':unittest.main()

import copy
import math
import unittest

from prepare_combat_future_dataset import example, local


def fixture():
    ident = dict(map='base3',connection=1,spawncount=2,actor=1,life=1,frame=10)
    enemy = dict(id=3,observed_track=5,**{'class':'monster_parasite'},distance=100,relative=[100,0,0],observed_velocity=[0,0,0],clear_shot=True)
    o = dict(identity=ident,health=80,position=[0,0,0],velocity=[0,0,0],view_angles=[0,0,0],enemies=[enemy])
    n = copy.deepcopy(o); n['identity']['frame']=11; n['position']=[10,0,0];n['enemies'][0]['relative']=[90,0,0]
    cmd = dict(Msec=100,Forward=400,Side=0,Up=0,Buttons=0)
    a = dict(identity=ident,forward=1,side=0,yaw_delta_degrees=0,pitch_delta_degrees=0,attack=False)
    step = dict(owner='provider',observation=o,next_observation=n,client_sequence=40,sent_command=cmd,applied_action=a,
                native_step=dict(begin_frame=10,end_frame=11,spawncount=2,actor=1,sequence=40),
                server_execution=dict(matched=True,window_exclusive=True,recovery_commands=0))
    trace = dict(combat_policy=dict(observation=o),sent_command=cmd)
    return step,trace


class FutureDatasetTest(unittest.TestCase):
    def test_ego_motion_is_not_monster_motion_and_future_actions_are_ignored(self):
        step,trace=fixture(); row,reason=example(step,trace,[0.]*1147)
        self.assertIsNone(reason);self.assertEqual(row['target'][0],[10/64,0,0]);self.assertEqual(row['target'][1],[0,0,0])
        before=copy.deepcopy(row);step['next_observation']['invented_future_action']='must not become input'
        self.assertEqual(example(step,trace,[0.]*1147)[0],before)

    def test_slot_reordering_and_track_recycling(self):
        step,trace=fixture(); other=copy.deepcopy(step['observation']['enemies'][0]);other.update(id=4,observed_track=6,distance=200,relative=[200,0,0])
        step['observation']['enemies'].append(other)
        nxt=copy.deepcopy(other);nxt['relative']=[195,0,0]
        step['next_observation']['enemies'].insert(0,nxt)
        row,_=example(step,trace,[0.]*1147)
        self.assertEqual(row['target'][1],[0,0,0]);self.assertEqual(row['target'][2],[5/64,0,0])
        step['next_observation']['enemies'][1]['observed_track']=99
        row,_=example(step,trace,[0.]*1147)
        self.assertFalse(row['motion_mask'][1]);self.assertEqual(row['present'][0],0)
        self.assertTrue(row['present_mask'][0]);self.assertFalse(row['clear_mask'][0])

    def test_reset_gap_death_and_dispatch_rejected(self):
        for change in ('life','frame','health','exclusive','recovery','msec','missing'):
            step,trace=fixture()
            if change=='life':step['next_observation']['identity']['life']=2
            if change=='frame':step['next_observation']['identity']['frame']=12
            if change=='health':step['next_observation']['health']=0
            if change=='exclusive':step['server_execution']['window_exclusive']=False
            if change=='recovery':step['server_execution']['recovery_commands']=1
            if change=='msec':step['sent_command']['Msec']=50
            if change=='missing':step['next_observation']=None
            row,reason=example(step,trace,[0.]*1147)
            self.assertIsNone(row,change);self.assertIsNotNone(reason,change)

    def test_rotation_uses_initial_forward_left_axes(self):
        result=local([10,0,2],math.pi/2)
        self.assertAlmostEqual(result[0],0);self.assertAlmostEqual(result[1],-10);self.assertEqual(result[2],2)


if __name__ == '__main__':unittest.main()

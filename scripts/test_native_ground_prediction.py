"""Reject misleading next-frame evidence, without neural execution."""
import unittest
from audit_native_ground_prediction import measure


def row(frame):
    return dict(observation_frame=frame,combat_policy=dict(observation=dict(identity=dict(life=1,frame=frame,spawncount=7),health=100)),
        pm_state_known=True,pm_type=0,pm_time=0,on_ground=True,self=[0,0,24.125],self_velocity=[0,0,0],
        connection=1,spawncount=7,map='base1',health=100,armor=0,sent_command=dict(Msec=100,Forward=0,Side=0),
        pmove_ground_prediction=dict(position=[0,0,24.125],velocity=[0,0,0],contacts=[]))


class Tests(unittest.TestCase):
    def test_unique_grounded_pair(self):
        x=measure([row(100),row(101)],100,7)
        self.assertEqual(x['all']['position_error']['maximum'],0)
        self.assertEqual(x['counts']['measured_pairs'],1)

    def test_duplicates_and_gap_not_treated_as_one_tick(self):
        self.assertEqual(measure([row(100),row(100),row(101)],100,7)['all']['position_error']['count'],0)
        self.assertEqual(measure([row(100),row(102)],100,7)['all']['position_error']['count'],0)

    def test_damage_and_airborne_excluded(self):
        a,b=row(100),row(101);b['health']=99
        self.assertEqual(measure([a,b],100,7)['counts']['external_damage_or_reset'],1)
        b=row(101);b['on_ground']=False
        self.assertEqual(measure([a,b],100,7)['counts']['next_frame_airborne'],1)

    def test_pre_release_and_next_life_excluded(self):
        b=row(101);b['combat_policy']['observation']['identity']['life']=2
        self.assertEqual(measure([row(99),row(100),b],100,7)['all']['position_error']['count'],0)


if __name__=='__main__':unittest.main()

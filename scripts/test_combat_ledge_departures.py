"""Pure geometry/data tests; no neural computation."""
import copy
import unittest
from audit_combat_ledge_departures import departures, warning


def observation(frame,z,grounded):
    return dict(identity=dict(frame=frame,map='base3',connection=1,spawncount=1,actor=1,life=1),
                position=[0,0,z],velocity=[0,0,0],on_ground=grounded,health=80,view_angles=[0,0,0],
                local_geometry=dict(frame=frame,probes=[dict(yaw_offset_degrees=i*45,ground_drop_at_40=0,
                                                 standing_hull_clearance=64) for i in range(8)]))


class DepartureTests(unittest.TestCase):
    def test_side_sign_and_world_velocity_rotation(self):
        obs=observation(10,0,True);action=dict(forward=0,side=1)
        self.assertEqual(warning(obs,action)['probe_degrees'],270)
        action['side']=-1
        self.assertEqual(warning(obs,action)['probe_degrees'],90)
        action['yaw_delta_degrees']=90
        self.assertEqual(warning(obs,action)['probe_degrees'],180)
        obs['view_angles'][1]=16384;obs['velocity']=[0,100,0]
        self.assertEqual(warning(obs,action)['probe_degrees'],0)
        self.assertEqual(warning(obs,action)['direction_source'],'observed_velocity')
        obs['local_geometry']['frame']=9
        self.assertIsNone(warning(obs,action))

    def test_factual_drop_gap_and_causal_warning(self):
        observations=[observation(10,0,True),observation(11,-10,False),
                      observation(12,-70,False),observation(13,-100,True)]
        rows=[dict(index=i+1,observation=observations[i],next_observation=observations[i+1],
                   applied_action=dict(forward=1,side=0,vertical='release'),terminal=False,truncated=False)
              for i in range(3)]
        rows[0]['observation']['local_geometry']['probes'][0]['ground_drop_at_40']=None
        diagnostics={10:dict(prediction=None,ground_surface='dry_flat')}
        original=copy.deepcopy(rows)
        event=departures(rows,diagnostics)[0]
        self.assertTrue(event['deep_drop']);self.assertEqual(event['drop'],100)
        self.assertTrue(event['landed']);self.assertTrue(event['precommand_warning']['ground_missing'])
        self.assertEqual(rows,original)
        observations[2]['local_geometry']['probes'][0]['ground_drop_at_40']=None
        self.assertEqual(departures(rows,diagnostics)[0]['precommand_warning'],event['precommand_warning'])
        short=departures([rows[0],rows[2]],diagnostics)[0]
        self.assertFalse(short['deep_drop']);self.assertTrue(short['right_censored'])

    def test_cross_life_successor_not_used(self):
        before=observation(10,0,True);after=observation(11,-200,False);after['identity']['life']=2
        row=dict(index=1,observation=before,next_observation=after,
                 applied_action=dict(forward=1,side=0,vertical='jump'),terminal=False,truncated=False)
        result=departures([row],{10:dict(prediction=None,ground_surface='dry_flat')})[0]
        self.assertFalse(result['deep_drop']);self.assertEqual(result['future_steps'],0)


if __name__=='__main__':unittest.main()

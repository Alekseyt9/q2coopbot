"""Native telemetry boundaries and ambiguous command matching, without NN."""
import unittest
from audit_native_motion_collisions import native_calls,match_motion

def line(kind,fields):
    return 'sv_test_motion spawncount=7 server_frame=100 g_test_motion version=1 frame=50 actor=1 call=1 kind='+kind+' '+fields

BEGIN=line('begin','msec=100 forward=10 side=0 up=0 pitch=0 yaw=0 pm_type=0 pm_time=0 pm_flags=4 snapinitial=0 origin=0,0,192 velocity=0,0,0')
TRACE=line('trace','index=2 fraction=0.5 startsolid=0 allsolid=0 blocker=8 class=monster_parasite model=none start=0,0,24 end=1,0,24 normal=-1,0,0')
END=line('end','traces=3 printed=1 touches=1 waterlevel=0 pm_type=0 pm_time=0 pm_flags=4 origin=8,0,192 velocity=0,0,0')

class Tests(unittest.TestCase):
    def test_closed_call(self):
        calls=native_calls([BEGIN,TRACE,END])
        self.assertEqual(calls[0]['traces'][0]['class'],'monster_parasite')
        row=dict(spawncount=7,observation_frame=100,self=[0,0,24],sent_command=dict(Msec=100,Forward=10,Side=0,Up=0,Pitch=0,Yaw=0))
        self.assertIsNotNone(match_motion(row,calls))
        self.assertIsNone(match_motion(row,calls+calls))
        row['sent_command']['Forward']=11
        self.assertIsNone(match_motion(row,calls))

    def test_incomplete_duplicate_and_wrong_counts_rejected(self):
        for lines in [[BEGIN,TRACE],[BEGIN,BEGIN,TRACE,END],[BEGIN,TRACE,END.replace('printed=1','printed=2')]]:
            with self.assertRaises(AssertionError):native_calls(lines)

    def test_nonfinite_and_unwrapped_rejected(self):
        for event in [TRACE.replace('fraction=0.5','fraction=nan'),TRACE.split('g_test_motion',1)[1].join(['g_test_motion',''])]:
            with self.assertRaises(AssertionError):native_calls([BEGIN,event,END])

if __name__=='__main__':unittest.main()

"""Counterexamples for recontact admission and command coordinates; no NN."""
import copy
import math
from prepare_combat_recontact_demonstrations import recontacts,world_movement


def fixture():
    steps=[];rows=[]
    for frame in range(1,7):
        o=dict(identity=dict(map='ware2',connection=1,spawncount=2,actor=1,life=1,frame=frame),health=100,
            enemies=[dict(id=519,**{'class':'monster_gunner'})] if frame in (1,6) else [],
            remembered_threats=[dict(id=519,**{'class':'monster_gunner'},age_frames=frame-1)],
            remembered_static_routes=dict(version='combat_observed_memory_static_routes_v1',frame=frame,hints=[
                dict(observed_id=519,memory_age_frames=frame-1,static_geometry_known=True,static_route_found=True,
                     next_static_waypoint_relative=[32,0,0])]))
        steps.append(dict(observation=o,index=frame,owner='provider',native_step=True,
                          server_execution=dict(matched=True,window_exclusive=True)))
        rows.append(dict(frame=frame))
    outcomes=[dict(available=True,step=7,events=[dict(target=519,target_class='monster_gunner',health_before=10,health_after=0)])]
    return steps,rows,outcomes


def main():
    steps,rows,outcomes=fixture();windows=recontacts(steps,rows,outcomes)
    assert len(windows)==1 and windows[0]['route_frames']==[2,3,4,5]
    tests=[]
    for field,value in [('health',0),('enemies',[dict(id=7,**{'class':'monster_soldier'})])]:
        def modify(s,r,o,field=field,value=value):
            for step in s[1:-1]:step['observation'][field]=value
        tests.append(modify)
    tests.extend([
        lambda s,r,o:s[2].update(owner='rules'),
        lambda s,r,o:s[2].update(native_step=False),
        lambda s,r,o:s[2]['server_execution'].update(matched=False),
        lambda s,r,o:s[2]['server_execution'].update(window_exclusive=False),
        lambda s,r,o:s[2]['observation']['identity'].update(life=2),
        lambda s,r,o:s[2]['observation']['identity'].update(spawncount=9),
        lambda s,r,o:s.pop(2),lambda s,r,o:r.pop(2),
        lambda s,r,o:o.clear(),lambda s,r,o:o[0].update(available=False),
        lambda s,r,o:o[0].update(step=3),
        lambda s,r,o:o[0]['events'][0].update(target=520),
        lambda s,r,o:o[0]['events'][0].update(health_before=0),
        lambda s,r,o:s[-1]['observation']['enemies'][0].update({'class':'monster_infantry'}),
    ])
    for mode in ('age','id','found'):
        def change(s,r,o,mode=mode):
            for step in s[1:-1]:
                hint=step['observation']['remembered_static_routes']['hints'][0]
                hint[{'age':'memory_age_frames','id':'observed_id','found':'static_route_found'}[mode]]={'age':99,'id':520,'found':False}[mode]
        tests.append(change)
    for i,mutate in enumerate(tests):
        s,r,o=copy.deepcopy(fixture());mutate(s,r,o)
        assert not recontacts(s,r,o),i
    for yaw,expected in [(0,[1,0]),(16384,[0,1]),(-32768,[-1,0])]:
        assert max(abs(a-b) for a,b in zip(world_movement(1,0,yaw),expected))<1e-12
    assert world_movement(0,1,0)==[0.,-1.]
    checks=0
    for yaw in (-32768,-24576,-16384,-1,0,16384,32767):
        angle=yaw*2*math.pi/65536;c,s=math.cos(angle),math.sin(angle)
        for f,r in ((1.,0.),(0.,1.),(.3,-.7)):
            x,y=world_movement(f,r,yaw)
            assert abs(x*c+y*s-f)<1e-12 and abs(x*s-y*c-r)<1e-12
            checks+=1
    print(dict(negative_cases=len(tests),coordinate_roundtrips=checks,neural_execution=False),flush=True)


if __name__=='__main__':main()

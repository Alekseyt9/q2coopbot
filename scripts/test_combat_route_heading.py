"""Static heading supervision counterexamples; metadata/geometry only, no NN."""
import copy
import math
from prepare_combat_route_heading import heading


def main():
    o=dict(health=100,identity=dict(frame=101),view_angles=[0,0,0],remembered_threats=[dict(id=7,age_frames=1)],
        remembered_static_routes=dict(version='combat_observed_memory_static_routes_v1',frame=101,hints=[
            dict(observed_id=7,memory_age_frames=1,static_geometry_known=True,static_route_found=True,
                 next_static_waypoint_relative=[0,16,0],static_horizontal_path_length=16)]))
    assert abs(heading(o)['turn_radians']-math.pi/2)<1e-12
    for point,expected in [([16,0,0],0),([0,-16,0],-math.pi/2),([-16,0,0],math.pi)]:
        c=copy.deepcopy(o);c['remembered_static_routes']['hints'][0]['next_static_waypoint_relative']=point
        assert abs(heading(c)['turn_radians']-expected)<1e-12
    checks=0
    for field,value in [('health',0),('enemies',[{'id':8}]),('age_ms',301)]:
        c=copy.deepcopy(o);c[field]=value;assert heading(c) is None;checks+=1
    for field,value in [('memory_age_frames',201),('observed_id',8),('static_route_found',False),
                        ('static_geometry_known',False),('next_static_waypoint_relative',[1,0,0])]:
        c=copy.deepcopy(o);c['remembered_static_routes']['hints'][0][field]=value;assert heading(c) is None;checks+=1
    c=copy.deepcopy(o);c['remembered_threats'][0]['age_frames']=2;assert heading(c) is None;checks+=1
    c=copy.deepcopy(o);c['remembered_static_routes']['frame']=100;assert heading(c) is None;checks+=1
    print(dict(cardinal_checks=4,negative_cases=checks,neural_execution=False),flush=True)


if __name__=='__main__':main()

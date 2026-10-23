"""Physical coordinate and action-gradient checks; CUDA only, no CPU NN."""
import math
from ppo_combat import torch,training_devices
from combat_action_geometry import features,NAMES,features_for_version,SMOOTH_VERSION,CENTERED_VERSION


def main():
    training_devices();torch.set_default_device('cuda')
    x=torch.zeros((1,185),device='cuda');x[:,7]=1;x[:,9]=1
    x[:,99]=1;x[:,100]=.5;x[:,101]=.5;x[:,147]=1
    x[:,148]=16/64;x[:,149]=-24/64;x[:,150]=32/64;x[:,152]=1;x[:,161]=1;x[:,181]=1
    index=lambda name:NAMES.index(name)
    base=features(x)
    assert abs(float(base[0,index('eye_yaw_error_pi')]))<1e-6
    assert abs(float(base[0,index('eye_pitch_error_pi')]))<1e-6
    assert float(base[0,index('eye_ray_bbox_intersection')])==1
    expected=math.atan2(8,256)/math.pi
    assert abs(float(base[0,index('muzzle_yaw_error_pi')])-expected)<1e-6
    rotated=x.clone();rotated[:,6]=1;rotated[:,7]=0
    assert torch.allclose(features(rotated),base,atol=1e-6,rtol=1e-5), 'Yaw-local target coordinates interpreted as world coordinates'
    miss=x.clone();miss[:,164]=5/180
    assert float(features(miss)[0,index('eye_ray_bbox_intersection')])==0
    assert float(features(miss)[0,index('eye_yaw_error_pi')])<0
    recoil=x.clone();recoil[:,182]=-6/32
    assert abs(float(features(recoil)[0,index('eye_pitch_error_pi')])-6/180)<1e-6
    recoil[:,165]=6/180
    assert abs(float(features(recoil)[0,index('eye_pitch_error_pi')]))<1e-6
    unknown=x.clone();unknown[:,147]=0
    assert bool((features(unknown)==0).all()), 'Unknown bbox must remain explicitly unavailable'
    gradient=x.clone().requires_grad_(True)
    features(gradient)[:,[index('eye_yaw_error_pi'),index('eye_pitch_error_pi')]].sum().backward()
    assert bool(torch.isfinite(gradient.grad).all())
    assert torch.allclose(gradient.grad[:,164:166],torch.full((1,2),-1.,device='cuda'),atol=1e-5)
    grazing=x.clone();grazing[:,164]=math.atan2(16,240)/math.pi
    soft=features_for_version(grazing,SMOOTH_VERSION)
    assert abs(float(soft[0,index('eye_ray_bbox_intersection')])-.5)<.0002
    centered=features_for_version(x,CENTERED_VERSION)
    assert abs(float(centered[0,index('muzzle_yaw_error_pi')]))<1e-6
    roll=x.clone();roll[:,184]=.5
    assert torch.equal(features_for_version(roll,CENTERED_VERSION),centered)
    print('CUDA geometry: frame rotation, ray hit/miss, parallax, recoil cancellation, unknown mask and action gradients passed')


if __name__=='__main__':main()

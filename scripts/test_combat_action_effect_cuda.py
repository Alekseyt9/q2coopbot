"""Physical frame/forecast invariants, checked only on CUDA."""
from ppo_combat import torch,training_devices
from combat_action_effect import inputs,projected_geometry,FactorizedMotor
from combat_action_geometry import features_for_version,CENTERED_VERSION


def main():
    training_devices();torch.set_default_device('cuda')
    c=torch.zeros((1,185),device='cuda');c[:,7]=1;c[:,9]=1
    c[:,99]=1;c[:,101]=.5;c[:,147]=1;c[:,148]=.25;c[:,149]=-.375;c[:,150]=.5;c[:,181]=1
    m=torch.zeros((1,5),device='cuda');timing=torch.zeros((1,2),device='cuda')
    base=features_for_version(c,CENTERED_VERSION)
    assert torch.equal(projected_geometry(c,m,False),base)
    # Own muzzle and observed target moving16units together leave the ray's
    # relative geometry unchanged. Unknown velocity must never be extrapolated.
    moving=c.clone();moving[:,104]=1;moving[:,105]=.4;m[:,0]=.5
    assert torch.allclose(projected_geometry(moving,m),base,atol=1e-6,rtol=1e-6)
    moving[:,104]=0
    assert float(projected_geometry(moving,m)[0,1])>float(base[0,1])
    m.zero_();m[:,4]=1;direct=c.clone();direct[:,164]=5/180
    assert torch.allclose(projected_geometry(c,m,False),features_for_version(direct,CENTERED_VERSION),atol=1e-6,rtol=1e-6)
    # Forward after a90deg yaw delta becomes +localY; side at yaw0 is -localY.
    command=c.clone();command[:,162]=1;command[:,164]=.5
    assert torch.allclose(inputs(command,timing)[:,188:190],torch.tensor([[0.,1.]],device='cuda'),atol=1e-6)
    command[:,162]=0;command[:,163]=1;command[:,164]=0
    assert torch.equal(inputs(command,timing)[:,188:190],torch.tensor([[0.,-1.]],device='cuda'))
    net=FactorizedMotor()
    with torch.no_grad():
        net.position[-1].weight.fill_(.1);net.direction[-1].weight.fill_(.1)
        x=inputs(c,timing);changed=x.clone();changed[:,99:161]=1234
        assert torch.equal(net(x),net(changed)), 'Enemy/target state leaked into motor mechanisms'
    print('CUDA action-effect invariants: centered zero forecast, co-moving translation, unavailable velocity, yaw correction, Quake movement frame and separated motor mechanisms passed')


if __name__=='__main__':main()

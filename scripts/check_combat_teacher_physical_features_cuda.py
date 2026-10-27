"""Physical coordinate/availability invariants; no CPU neural tests."""
import math

from combat_teacher_physical_features import candidate_contexts,features
from combat_teacher_logistic_cuda import fit
from ppo_combat import torch,training_devices


def main():
    training_devices()
    x=torch.zeros((1,187),device='cuda',dtype=torch.float64)
    x[:,7]=x[:,9]=1
    x[:,99]=1;x[:,100]=x[:,101]=.5
    x[:,147]=1;x[:,148]=.25;x[:,149]=-.375;x[:,150]=.5
    x[:,152]=1;x[:,160]=1;x[:,162]=1;x[:,164]=.5;x[:,181]=1
    x[:,186]=-math.atan2(8,256)/math.pi
    before=x.clone();out=features(x)
    assert torch.equal(x,before),'Feature derivation mutated source context'
    # Forward after90deg turns into lateral movement; anchor0 faces target.
    assert abs(float(out[0,16]))<1e-9 and abs(float(out[0,17])-1)<1e-9
    assert abs(float(out[0,20])-math.sqrt(2))<1e-9
    assert float(out[0,8])==0 and float(out[0,15])>0, 'Anchor must improve the current ray/bbox proxy'
    # Quake positive side at yaw0 points toward local -Y.
    side=x.clone();side[:,162]=0;side[:,163]=1;side[:,164]=0
    s=features(side);assert abs(float(s[0,16]))<1e-9 and float(s[0,18])==1
    # World rotation must preserve movement in the target-relative local frame.
    x[:,10]=.3;x[:,104]=1;x[:,105]=.2
    y=x.clone();y[:,6]=1;y[:,7]=0;y[:,10]=0;y[:,11]=.3
    assert torch.allclose(features(x)[:,16:],features(y)[:,16:],atol=1e-9,rtol=0)
    # Unknown observed target velocity never becomes a made-up prediction.
    x[:,104]=0;x[:,105]=4
    assert float(features(x)[0,22])==0
    x[:,8]=math.sin(math.radians(80));x[:,9]=math.cos(math.radians(80));x[:,165]=.5
    source,anchor=candidate_contexts(x)
    for c in (source,anchor):
        absolute=80+180*c[:,165]
        assert bool(((absolute>=-89)&(absolute<=89)).all())
    assert abs(float(source[0,165])-9/180)<1e-12
    # Weighted intercept has an independent analytic optimum log(3).
    offset=torch.zeros(4,device='cuda',dtype=torch.float64)
    design=torch.ones((4,1),device='cuda',dtype=torch.float64)
    labels=torch.tensor([0.,0.,1.,1.],device='cuda',dtype=torch.float64)
    weights=torch.tensor([1.,1.,3.,3.],device='cuda',dtype=torch.float64)
    theta,audit=fit(offset,design,labels,weights,10.)
    assert abs(float(theta[0])-math.log(3))<1e-6 and audit['gradient_maximum']<1e-6
    # A regularized slope must separate a simple known causal ordering.
    design=torch.tensor([[-2.,1.],[-1.,1.],[1.,1.],[2.,1.]],device='cuda',dtype=torch.float64)
    theta,audit=fit(offset,design,labels,torch.ones_like(labels),.1)
    assert float(theta[0])>0 and torch.equal((design@theta>=0).double(),labels)
    print('CUDA physical features PASS: causal input isolation,pitch clamp,centered bbox geometry,Quake side sign,rotation and unavailable velocity')
    print('CUDA convex optimizer PASS: analytic weighted intercept and regularized ordered signal')


if __name__=='__main__':main()

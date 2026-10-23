"""CUDA-only exploratory relative movement geometry; no ordinary runtime integration."""
import math
from ppo_combat import torch,nn,layers


def geometry_features(joint,actor,x,raw,target,mode,vertical,weapon,timing,deltas):
    from combat_precision_head import selected_means
    from combat_action_effect import inputs as motor_inputs
    assert x.is_cuda and x.shape[1]==1147 and deltas.shape==(len(x),6,2)
    n=len(x);repeat=lambda v:v[:,None].expand(-1,6,*v.shape[1:]).reshape(n*6,*v.shape[1:])
    xx,rr=repeat(x),repeat(raw)
    tt,mm,vv,ww=(repeat(v) for v in (target,mode,vertical,weapon))
    movement=rr[:,:2]+deltas.reshape(-1,2)
    means=selected_means(rr,tt,mm,xx,movement,actor.sampled_motion_aim_branch,vv,actor.motor_branch)
    z=torch.cat((movement,means[:,2:]),-1)
    c=joint.action_context(xx,z,tt,mm,vv,ww).double()
    motor=joint.hints.motor((motor_inputs(c,repeat(timing).double())-joint.hints.motor_mean)/joint.hints.motor_std).reshape(n,6,5)
    displacement=motor[:,:,:2]*32
    enemies=x[:,73:169].reshape(n,8,12).double();typed=x[:,466:786].reshape(n,8,40).double()
    expand=lambda value:value[:,None].expand(-1,6,*value.shape[1:])
    e,t=expand(enemies),expand(typed)
    r=e[:,:,:,2:5]
    projected=r-torch.cat((displacement/512,torch.zeros((n,6,1),device='cuda',dtype=torch.float64)),-1)[:,:,None,:]
    current_distance=r.norm(dim=-1,keepdim=True);projected_distance=projected.norm(dim=-1,keepdim=True)
    target_slot=(target-1).clamp_min(0)
    selected=(torch.arange(8,device='cuda')[None,:]==target_slot[:,None])&(target>0)[:,None]
    own=displacement[:,:,None,:].expand(-1,-1,8,-1)/32
    tokens=torch.cat((t[:,:,:,:21],r,e[:,:,:,6:9],e[:,:,:,5:6],t[:,:,:,27:30],projected,
        current_distance,projected_distance,projected_distance-current_distance,
        expand(selected.double())[:,:,:,None],t[:,:,:,36:38],own),-1)
    assert tokens.shape==(n,6,8,42)
    probes=x[:,25:73].reshape(n,8,6).double()
    angle=torch.arange(8,device='cuda',dtype=torch.float64)*math.pi/4
    directions=torch.stack((angle.cos(),angle.sin()),-1)
    projection=torch.einsum('ncd,kd->nck',displacement,directions)
    ray=probes[:,:,1]*256;known=probes[:,:,0]
    normalized=(projection-projection[:,:1])/(ray[:,None,:]+32)*known[:,None,:]
    pressure=nn.functional.softplus((32-(ray[:,None,:]-projection))/16)/4
    pressure=(pressure-pressure[:,:1])*known[:,None,:]
    analog=movement.tanh().reshape(n,6,2).double();d=deltas.double()
    near=torch.zeros((n,6,1),device='cuda',dtype=torch.float64);near[:,5]=1
    wall=torch.cat((normalized,pressure,analog,analog-analog[:,:1],d/2,(displacement-displacement[:,:1])/32,
        d.square().sum(-1,keepdim=True)/4,near),-1)
    assert wall.shape==(n,6,26) and bool(torch.isfinite(tokens).all()) and bool(torch.isfinite(wall).all())
    return tokens.detach().float(),wall.detach().float(),enemies[:,:,0]>0,motor.detach()


class GeometryMovementSelector(nn.Module):
    def __init__(self,nonlinear):
        super().__init__()
        self.enemy=(nn.Sequential(nn.Linear(42,8,device='cuda'),nn.ReLU(),nn.Linear(8,1,device='cuda'))
            if nonlinear else nn.Sequential(nn.Linear(42,1,device='cuda')))
        self.wall=nn.Linear(26,1,bias=False,device='cuda')
        self.beta=nn.Parameter(torch.tensor(math.log(495.),device='cuda'))
        with torch.no_grad():self.enemy[-1].weight.zero_();self.enemy[-1].bias.zero_();self.wall.weight.zero_()
    def forward(self,tokens,wall,present):
        risk=self.enemy(tokens).squeeze(-1)
        score=((risk-risk[:,:1])*present[:,None,:]).sum(-1)+self.wall(wall).squeeze(-1)
        control=torch.zeros(6,device='cuda');control[0]=1
        return (score+self.beta*control).clamp(-20,20)
    def export(self):
        return dict(enemy_layers=layers(self.enemy),wall_weight=self.wall.weight[0].detach().tolist(),beta=float(self.beta.detach()))

"""Versioned learned choice after the original Gaussian+25-way aim event."""
import math
from train_combat_bc import torch,nn
from combat_body_features import legacy
from combat_conditional_fire import inputs as fire_inputs
from combat_aim_candidates import frame_uniforms

VERSION='combat_observed_mg_wide_choice_v1'
DISTRIBUTION='combat_joint_aim_wide_v2'
SAMPLER='frame_hash_wide_bernoulli_v1'
SALT=0x6a09e667f3bcc909
WIDTH=187
PROBE_VERSION='combat_one_frame_wide_gate_clamp_v1'


def validate(model,*,allow_probe=False):
    h=model.get('wide_aim')
    probe=model.get('diagnostic_wide_gate_probe')
    if probe is not None:
        assert allow_probe and h and h['enabled'],'Diagnostic gate cannot enter ordinary scoring/training'
        assert set(probe)=={'version','frame','choice'} and probe['version']==PROBE_VERSION
        assert type(probe['frame']) is int and 0<probe['frame']<=2**31-1 and type(probe['choice']) is int and probe['choice'] in (0,1)
    if model.get('diagnostic_continuation_probe') is not None and h is not None:
        assert allow_probe,'Diagnostic continuation cannot enter ordinary scoring/training'
        from combat_continuation_probe import validate as validate_continuation
        validate_continuation(model)
    if h is None:
        assert model.get('action_distribution')!=DISTRIBUTION,'Wide event requires a head'
        return
    assert model.get('action_distribution')==DISTRIBUTION
    assert set(h)=={'version','sampler','enabled','layers'} and h['version']==VERSION and h['sampler']==SAMPLER and type(h['enabled']) is bool
    assert not model.get('deterministic') and model.get('combat_execution_contract')=='stock_weapons_fire_v1'
    assert model.get('conditional_fire') and model.get('aim_candidate_head')
    assert not any(k.startswith('diagnostic_') and k not in ('diagnostic_wide_gate_probe','diagnostic_continuation_probe') for k in model)
    assert not any(model.get(k) for k in ('execution_mode','geometry_aim_ranker','geometry_aim_exploration'))
    width=WIDTH
    assert len(h['layers'])==3
    for l,n in zip(h['layers'],(32,16,1)):
        assert set(l)=={'weight','bias'} and len(l['weight'])==len(l['bias'])==n
        assert all(len(row)==width and all(math.isfinite(v) and abs(v)<=1e4 for v in [*row,b]) for row,b in zip(l['weight'],l['bias']))
        width=n


def geometry(features,target,mode):
    x=legacy(features);n=len(x);slot=(target-1).clamp_min(0)
    e=x[:,73:169].reshape(n,8,12).gather(1,slot[:,None,None].expand(-1,1,12))[:,0]
    t=x[:,466:786].reshape(n,8,40).gather(1,slot[:,None,None].expand(-1,1,40))[:,0]
    active=(x[:,0]>0)&(x[:,1125]>0)&(x[:,810]>0)&(target>0)&(mode==0)&(e[:,0]>0)&(e[:,11]>0)&(t[:,36]>0)
    yaw=torch.atan2(x[:,6],x[:,7]);pitch=torch.atan2(x[:,8],x[:,9]);local=e[:,2:5]*512
    p=torch.stack((yaw.cos()*local[:,0]-yaw.sin()*local[:,1],yaw.sin()*local[:,0]+yaw.cos()*local[:,1],local[:,2]),-1)
    lo,hi=t[:,38]*64,t[:,39]*64
    height=torch.where(hi-lo<16,(lo+hi)/2,torch.minimum(torch.maximum(torch.full_like(hi,22),lo+8),hi-8))
    eye=torch.where(x[:,3]>0,-2.,22.)
    p[:,2]+=height-eye+8
    active=active&(p.square().sum(-1)>1e-18)
    kick=x[:,811:814]*32;wrap=lambda a:torch.remainder(a+180,360)-180
    a=torch.stack((wrap(torch.atan2(p[:,1],p[:,0])*180/math.pi-yaw*180/math.pi-kick[:,1]),
        wrap(-torch.atan2(p[:,2],torch.linalg.vector_norm(p[:,:2],dim=-1))*180/math.pi-pitch*180/math.pi-kick[:,0])),-1)
    return torch.where(active[:,None],a,0.),active


def transform(source,anchor,choice):
    assert source.is_cuda and source.shape==(len(choice),4) and anchor.shape==(len(choice),2)
    assert bool(((choice==0)|(choice==1)).all())
    aim=(anchor/180).clamp(-1+1e-6,1-1e-6).atanh()
    return torch.cat((source[:,:2],torch.where((choice==1)[:,None],aim,source[:,2:])), -1)


class WideAim(nn.Module):
    def __init__(self,spec):
        super().__init__();self.enabled=spec['enabled'];self.probe=None
        self.layers=nn.Sequential(nn.Linear(WIDTH,32,device='cuda'),nn.ReLU(),nn.Linear(32,16,device='cuda'),nn.ReLU(),nn.Linear(16,1,device='cuda'))
        with torch.no_grad():
            for i,l in zip((0,2,4),spec['layers']):
                self.layers[i].weight.copy_(torch.tensor(l['weight'],device='cuda',dtype=torch.float32))
                self.layers[i].bias.copy_(torch.tensor(l['bias'],device='cuda',dtype=torch.float32))

    def forward(self,features,source,target,mode,vertical,weapon):
        anchor,active=geometry(features,target,mode)
        context=fire_inputs(features,target,mode,source,vertical,weapon)
        x=torch.cat((context,features[:,810:814],anchor/180),-1).detach()
        assert x.shape==(len(features),WIDTH) and bool(torch.isfinite(x).all())
        logits=self.layers(x.to(next(self.parameters()).dtype))[:,0].clamp(-20,20)
        return logits,active&self.enabled


def events(model,rows,features,target,mode,vertical,weapon,*,allow_probe=False):
    """Pure sampled-event verification on CUDA; first-life export contract."""
    validate(model,allow_probe=allow_probe);annotations=[r['sample'].get('wide_aim') for r in rows]
    h=model.get('wide_aim')
    if h is None or not h['enabled']:
        assert all(a is None for a in annotations),'Wide annotation on disabled/legacy policy'
        return dict(features=features,source=torch.tensor([r['sample']['latent'] for r in rows],device='cuda',dtype=features.dtype),
            choice=torch.zeros(len(rows),device='cuda',dtype=torch.long),active=torch.zeros(len(rows),device='cuda',dtype=torch.bool),
            uniforms=None,annotations=annotations)
    exact=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float64)
    anchor,active=geometry(exact,target,mode)
    if h is None or not h['enabled']:active=torch.zeros_like(active)
    present=torch.tensor([a is not None for a in annotations],device='cuda',dtype=torch.bool)
    assert torch.equal(active,present),'Missing or ineligible wide event'
    keys={'version','source_latent','anchor_degrees','choice','uniform','fire_uniform','logit','log_probability'}
    assert all(a is None or (set(a)==keys and a['version']==VERSION and a['choice'] in (0,1) and
        all(math.isfinite(a[k]) for k in ('uniform','fire_uniform','logit','log_probability')) and 0<=a['fire_uniform']<1 and abs(a['logit'])<=20) for a in annotations)
    source=torch.tensor([a['source_latent'] if a else r['sample']['latent'] for r,a in zip(rows,annotations)],device='cuda',dtype=torch.float64)
    choice=torch.tensor([a['choice'] if a else 0 for a in annotations],device='cuda',dtype=torch.long)
    final=torch.tensor([r['sample']['latent'] for r in rows],device='cuda',dtype=torch.float64)
    assert bool(torch.isfinite(source).all()) and float((transform(source,anchor,choice)-final).abs().max())<1e-8
    from combat_continuation_probe import effective_seed
    uniforms=torch.tensor([frame_uniforms(effective_seed(model,r)^SALT,r['frame'],1)[0] for r in rows],device='cuda',dtype=torch.float64)
    if bool(active.any()):
        actual=torch.tensor([a['anchor_degrees'] if a else [0,0] for a in annotations],device='cuda',dtype=torch.float64)
        assert float((actual[active]-anchor[active]).abs().max())<1e-8
        recorded=torch.tensor([a['uniform'] if a else 0. for a in annotations],device='cuda',dtype=torch.float64)
        assert torch.equal(recorded[active],uniforms[active])
        l=torch.tensor([a['logit'] if a else 0. for a in annotations],device='cuda',dtype=torch.float64)
        lp=torch.tensor([a['log_probability'] if a else 0. for a in annotations],device='cuda',dtype=torch.float64)
        d=torch.distributions.Bernoulli(logits=l)
        assert float((d.log_prob(choice.double())[active]-lp[active]).abs().max())<1e-9
        assert torch.equal((uniforms[active]<l[active].sigmoid()).long(),choice[active])
    return dict(features=exact,source=source,choice=choice,active=active,uniforms=uniforms,annotations=annotations,frames=torch.tensor([r['frame'] for r in rows],device='cuda',dtype=torch.long))


def terms(module,data,target,mode,vertical,weapon,*,verify=False):
    if module is None or not module.enabled:
        assert not bool(data['active'].any());zero=data['source'][:,0]*0
        return zero,zero
    logits,active=module(data['features'],data['source'],target,mode,vertical,weapon)
    assert torch.equal(active,data['active'])
    if module.probe is not None:
        root=data['frames']==module.probe['frame']
        assert int(root.sum())==1 and bool(active[root].all()),'Missing or ineligible diagnostic gate root'
        logits=torch.where(root,torch.full_like(logits,40*module.probe['choice']-20),logits)
        assert int(data['choice'][root][0])==module.probe['choice'],'Clamp branch differs'
    d=torch.distributions.Bernoulli(logits=logits)
    lp=torch.where(active,d.log_prob(data['choice'].to(logits.dtype)),torch.zeros_like(logits))
    ent=torch.where(active,d.entropy(),torch.zeros_like(logits))
    if verify and bool(active.any()):
        expected=torch.tensor([a['logit'] if a else 0. for a in data['annotations']],device='cuda',dtype=logits.dtype)
        assert float((expected[active]-logits[active]).abs().max())<1e-8,'Wide live logits differ'
        assert torch.equal((data['uniforms'][active]<logits[active].double().sigmoid()).long(),data['choice'][active]),'Wide CDF assignment differs'
    return lp,ent


def verify_fire(data,logits,attack):
    active=data['active']
    if bool(active.any()):
        uniforms=torch.tensor([a['fire_uniform'] if a else 0. for a in data['annotations']],device='cuda',dtype=logits.dtype)
        assert torch.equal((uniforms[active]<logits[active].sigmoid()),attack[active].bool()),'Wide final-fire CDF differs'

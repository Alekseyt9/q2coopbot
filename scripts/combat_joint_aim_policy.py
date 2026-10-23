"""Versioned CUDA joint likelihood; original Gaussian draw plus aim category.

Frozen motor/hit predictions are hints, never native outcomes or a firing mask.
Legacy precision scorers continue to reject the augmented policy header.
"""
import copy
import math
from train_combat_bc import torch,nn
import combat_aim_candidates as candidates
from combat_conditional_fire import inputs as fire_inputs,logits as fire_logits
from combat_action_effect import FactorizedMotor,inputs as motor_inputs,projected_geometry,POSITION_COLUMNS,DIRECTION_COLUMNS


PROBE_VERSION='combat_joint_aim_one_frame_probe_v1'


def base_model(model,*,allow_probe=False):
    assert model.get('action_distribution')==candidates.VERSION
    assert model.get('aim_candidate_head',{}).get('version')==candidates.VERSION
    assert not model.get('execution_mode') and not model.get('geometry_aim_ranker') and not model.get('geometry_aim_exploration')
    assert not model.get('deterministic') and model['feature_version']=='combat_features_v10'
    assert allow_probe or not model.get('diagnostic_joint_aim_probe'),'Diagnostic one-frame probes cannot enter ordinary training/scoring'
    result=copy.deepcopy(model)
    result.pop('action_distribution');result.pop('aim_candidate_head')
    result.pop('diagnostic_joint_aim_probe',None)
    from combat_precision_head import validate_precision_model
    validate_precision_model(result)
    return result


def _layers(spec,width,outputs):
    assert len(spec)==len(outputs)
    for layer,output in zip(spec,outputs):
        assert set(layer)=={'weight','bias'} and len(layer['weight'])==len(layer['bias'])==output
        assert all(len(row)==width and all(math.isfinite(v) and abs(v)<=1e6 for v in [*row,b]) for row,b in zip(layer['weight'],layer['bias']))
        width=output


def validate(model,*,allow_probe=False):
    base_model(model,allow_probe=allow_probe)
    h=model['aim_candidate_head']
    assert set(h)=={'version','offset_degrees','sampler','layers','predictor'}
    assert h['offset_degrees']==list(candidates.OFFSETS) and h['sampler']=='frame_hash_categorical_v1'
    _layers(h['layers'],1359,(64,25))
    probe=model.get('diagnostic_joint_aim_probe')
    if probe:
        assert allow_probe and set(probe)=={'version','frame','logits'}
        assert probe['version']==PROBE_VERSION and type(probe['frame']) is int and 0<probe['frame']<=2**31-1
        for values in (probe['logits'],h['layers'][-1]['bias']):
            assert len(values)==25 and all(math.isfinite(v) and abs(v)<=100 for v in values) and max(values)-min(values)<=60
        assert all(v==0 for row in h['layers'][-1]['weight'] for v in row),'Probe requires constant baseline head'
    p=h['predictor'];m=p['motor']
    assert p['kind']=='combat_geometry_shot_predictor_v1' and p['geometry_version']=='combat_predicted_action_geometry_v1'
    assert p['input_width']==214 and p['causal_width']==185 and p['trained_weapon']=='machinegun' and len(p['dataset_sha256'])==64
    from combat_action_geometry import names_for_version,CENTERED_VERSION
    assert p['geometry_names']==list(names_for_version(CENTERED_VERSION))
    assert m['version']=='combat_action_effect_motor_v1' and m['hand']==2 and m['input_width']==192
    assert m['position_columns']==list(POSITION_COLUMNS) and m['direction_columns']==list(DIRECTION_COLUMNS)
    assert (m['target_extrapolation_seconds'],m['displacement_scale'],m['angle_scale_degrees'])==(0,32,5)
    _layers(m['position_layers'],76,(32,3));_layers(m['direction_layers'],21,(16,2))
    for spec,width in ((p,214),(m,192)):
        for name in ('mean','std'):
            assert len(spec[name])==width and all(math.isfinite(v) and abs(v)<=1e6 for v in spec[name])
        assert all(v>0 for v in spec['std'])
    assert len(p['weight'])==214 and all(math.isfinite(v) and abs(v)<=1e6 for v in [*p['weight'],p['bias']])


class FrozenHints(nn.Module):
    def __init__(self,p):
        super().__init__()
        self.motor=FactorizedMotor().double()
        m=p['motor']
        with torch.no_grad():
            for branch,key in ((self.motor.position,'position_layers'),(self.motor.direction,'direction_layers')):
                for index,layer in zip((0,2),m[key]):
                    branch[index].weight.copy_(torch.tensor(layer['weight'],device='cuda',dtype=torch.float64))
                    branch[index].bias.copy_(torch.tensor(layer['bias'],device='cuda',dtype=torch.float64))
        for name,value in (('motor_mean',m['mean']),('motor_std',m['std']),('shot_mean',p['mean']),('shot_std',p['std']),('shot_weight',p['weight']),('shot_bias',p['bias'])):
            self.register_buffer(name,torch.tensor(value,device='cuda',dtype=torch.float64))
        self.requires_grad_(False)

    def forward(self,causal,timing):
        with torch.no_grad():
            c=causal.detach().double();t=timing.detach().double()
            motor=self.motor((motor_inputs(c,t)-self.motor_mean)/self.motor_std)
            g=projected_geometry(c,motor,False)
            return (((torch.cat((c,g),-1)-self.shot_mean)/self.shot_std)*self.shot_weight).sum(-1)+self.shot_bias


class JointAim(nn.Module):
    def __init__(self,model,*,allow_probe=False):
        super().__init__();validate(model,allow_probe=allow_probe)
        probe=model.get('diagnostic_joint_aim_probe')
        self.probe_frame=probe['frame'] if probe else None
        self.register_buffer('probe_logits',torch.tensor(probe['logits'],device='cuda',dtype=torch.float32) if probe else None)
        self.head=candidates.AimCandidateHead()
        with torch.no_grad():
            for index,layer in zip((0,2),model['aim_candidate_head']['layers']):
                self.head.layers[index].weight.copy_(torch.tensor(layer['weight'],device='cuda',dtype=torch.float32))
                self.head.layers[index].bias.copy_(torch.tensor(layer['bias'],device='cuda',dtype=torch.float32))
        self.hints=FrozenHints(model['aim_candidate_head']['predictor'])

    def action_context(self,features,source,target,mode,vertical,weapon):
        c=fire_inputs(features,target,mode,source,vertical,weapon)
        return torch.cat((c,features[:,810:814].detach()),-1)

    def forward(self,features,source,target,mode,vertical,weapon,timing,frames=None):
        c=self.action_context(features,source,target,mode,vertical,weapon)
        active=candidates.eligible(c)
        if self.probe_frame is not None:
            assert frames is not None and frames.is_cuda and frames.dtype==torch.long and frames.shape==(len(c),),'Probe scoring requires explicit recorded frames'
            constant=self.head.layers[-1].bias[None,:].expand(len(c),-1)
            return torch.where((frames==self.probe_frame)[:,None],self.probe_logits[None,:],constant),active
        # Ineligible rows contribute no categorical event. Avoid forecasting
        # their absent bbox while retaining a differentiable zero head branch.
        hints=torch.zeros((len(c),25),device='cuda',dtype=features.dtype)
        if bool(active.any()):
            cc=c[active];count=len(cc)
            expanded=cc[:,None,:].expand(-1,25,-1).clone()
            z=source[active,None,:].expand(-1,25,-1).reshape(-1,4).double()
            modes=mode[active,None].expand(-1,25).reshape(-1)
            choices=torch.arange(25,device='cuda').expand(count,-1).reshape(-1)
            final=candidates.apply_offsets(z,modes,choices)
            scale=torch.where(modes==1,15./180.,1.).double()
            expanded[:,:,164:166]=(final[:,2:].tanh()*scale[:,None]).reshape(count,25,2).to(expanded.dtype)
            t=timing[active,None,:].expand(-1,25,-1).reshape(-1,2)
            hints[active]=self.hints(expanded.reshape(-1,185),t).reshape(count,25).to(features.dtype)
        return self.head(candidates.inputs(features,c,hints,timing.to(features.dtype))),active


def events(rows,features,target,mode,vertical,weapon):
    """Validate explicit source/category records on CUDA, before scoring."""
    annotations=[r['sample'].get('aim_candidate') for r in rows]
    source=torch.tensor([a['source_latent'] if a else r['sample']['latent'] for r,a in zip(rows,annotations)],device='cuda',dtype=torch.float64)
    choice=torch.tensor([a['candidate'] if a else 0 for a in annotations],device='cuda',dtype=torch.long)
    timing=torch.tensor([a['timing'] if a else [0.,0.] for a in annotations],device='cuda',dtype=torch.float64)
    active=candidates.eligible(torch.cat((fire_inputs(features,target,mode,source,vertical,weapon),features[:,810:814]),-1))
    present=torch.tensor([a is not None for a in annotations],device='cuda',dtype=torch.bool)
    assert torch.equal(active,present),'Missing or ineligible joint aim event'
    assert all(a is None or (set(a)=={'source_latent','candidate','candidates','log_probability','timing'} and a['candidates']==25 and math.isfinite(a['log_probability'])) for a in annotations)
    final=torch.tensor([r['sample']['latent'] for r in rows],device='cuda',dtype=torch.float64)
    expected=candidates.apply_offsets(source,mode,choice)
    assert bool(torch.isfinite(timing).all()) and float((expected-final).abs().max())<1e-8,'Recorded correction transform differs'
    return source.to(features.dtype),choice,timing.to(features.dtype),active


def probabilities(joint,raw,std,source,final,attack,vertical,weapon,features,target,mode,choice,timing,motion_branch=None,reference_branch=None,fire_branch=None,*,frames=None):
    from combat_precision_head import probabilities as original
    lp,ent=original(raw,std,source,attack,vertical,weapon,features,target,mode,motion_branch,reference_branch,fire_branch)
    # Gaussian terms keep source latent. Only fire sees the final aim.
    before=torch.distributions.Bernoulli(logits=fire_logits(fire_branch,features,raw,target,mode,source,vertical,weapon))
    after=torch.distributions.Bernoulli(logits=fire_logits(fire_branch,features,raw,target,mode,final,vertical,weapon))
    logits,active=joint(features,source,target,mode,vertical,weapon,timing,frames=frames)
    lp=lp+after.log_prob(attack)-before.log_prob(attack)+candidates.log_probability(logits,choice,active)
    ent=ent+after.entropy()-before.entropy()+candidates.entropy(logits,active)
    assert bool(torch.isfinite(lp).all()) and bool(torch.isfinite(ent).all())
    return lp,ent,logits,active

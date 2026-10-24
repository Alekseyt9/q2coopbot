"""Versioned CUDA joint likelihood; original Gaussian draw plus aim category.

Frozen motor/hit predictions are hints, never native outcomes or a firing mask.
Legacy precision scorers continue to reject the augmented policy header.
"""
from combat_body_features import VERSIONS as BODY_FEATURE_VERSIONS, WIDTH as BODY_FEATURE_WIDTH, legacy as body_legacy
import copy
import math
from train_combat_bc import torch,nn
import combat_aim_candidates as candidates
from combat_conditional_fire import inputs as fire_inputs,logits as fire_logits
from combat_action_effect import FactorizedMotor,inputs as motor_inputs,projected_geometry,POSITION_COLUMNS,DIRECTION_COLUMNS


PROBE_VERSION='combat_joint_aim_one_frame_probe_v1'
POLICY_PROBE_VERSION='combat_joint_aim_one_frame_policy_probe_v2'
GEOMETRY_KIND='geometry_candidate_delta31_mlp_v1'
GEOMETRY_SAMPLER='frame_hash_geometry_delta31_v1'


def base_model(model,*,allow_probe=False,allow_shared=False,allow_target_sampling=False):
    from combat_target_sampling import validate as validate_target_sampling
    header=validate_target_sampling(model.get("target_sampling"))
    assert header is None or allow_target_sampling,"Target calibration requires explicit scoring; ordinary training unsupported"
    assert model.get('action_distribution')==candidates.VERSION
    assert model.get('aim_candidate_head',{}).get('version')==candidates.VERSION
    assert not model.get('execution_mode') and not model.get('geometry_aim_ranker') and not model.get('geometry_aim_exploration')
    assert not model.get('deterministic') and model['feature_version'] in BODY_FEATURE_VERSIONS
    assert allow_probe or not (model.get('diagnostic_joint_aim_probe') or model.get('diagnostic_movement_probe') is not None or model.get('diagnostic_continuation_probe') is not None or model.get('diagnostic_weapon_probe') is not None),'Diagnostic acquisition cannot enter ordinary training/scoring'
    if model.get('diagnostic_weapon_probe') is not None:
        from combat_weapon_probe import validate as validate_weapon_probe
        validate_weapon_probe(model)
    if model.get('diagnostic_movement_probe') is not None:
        from combat_movement_probe import validate as validate_movement
        validate_movement(model)
    if model.get('diagnostic_continuation_probe') is not None:
        from combat_continuation_probe import validate as validate_continuation
        validate_continuation(model)
    assert allow_shared or not model['aim_candidate_head'].get('shared_scorer'),'Shared scorer optimizer/resume is not integrated; use explicit evaluation scoring'
    result=copy.deepcopy(model)
    result.pop('action_distribution');result.pop('aim_candidate_head');result.pop('target_sampling',None)
    result.pop('diagnostic_joint_aim_probe',None)
    result.pop('diagnostic_movement_probe',None)
    result.pop('diagnostic_continuation_probe',None)
    result.pop('diagnostic_weapon_probe',None)
    from combat_precision_head import validate_precision_model
    validate_precision_model(result)
    return result


def _layers(spec,width,outputs):
    assert len(spec)==len(outputs)
    for layer,output in zip(spec,outputs):
        assert set(layer)=={'weight','bias'} and len(layer['weight'])==len(layer['bias'])==output
        assert all(len(row)==width and all(math.isfinite(v) and abs(v)<=1e6 for v in [*row,b]) for row,b in zip(layer['weight'],layer['bias']))
        width=output


def validate(model,*,allow_probe=False,allow_shared=False,allow_target_sampling=False):
    base_model(model,allow_probe=allow_probe,allow_shared=allow_shared,allow_target_sampling=allow_target_sampling)
    h=model['aim_candidate_head']
    assert set(h)=={'version','offset_degrees','sampler','layers','predictor'} | ({'shared_scorer'} if h.get('shared_scorer') else set())
    shared=h.get('shared_scorer')
    if shared:
        assert allow_shared and (not model.get('diagnostic_joint_aim_probe') or model['diagnostic_joint_aim_probe']['version']==POLICY_PROBE_VERSION)
        assert set(shared)=={'kind','layers','beta','score_clip','beta_bounds'}
        assert shared['kind'] in ('shared_candidate_causal191_v1',GEOMETRY_KIND) and shared['score_clip']==20 and shared['beta_bounds']==[-20,20]
        assert math.isfinite(shared['beta']) and abs(shared['beta'])<=20
        _layers(shared['layers'],31 if shared['kind']==GEOMETRY_KIND else 191,(8,1) if shared['kind']==GEOMETRY_KIND else (16,1))
    expected_sampler=GEOMETRY_SAMPLER if shared and shared['kind']==GEOMETRY_KIND else 'frame_hash_categorical_v1'
    assert h['offset_degrees']==list(candidates.OFFSETS) and h['sampler']==expected_sampler
    _layers(h['layers'],1359,(64,25))
    probe=model.get('diagnostic_joint_aim_probe')
    if probe:
        assert allow_probe and set(probe)=={'version','frame','logits'}
        assert probe['version'] in (PROBE_VERSION,POLICY_PROBE_VERSION) and type(probe['frame']) is int and 0<probe['frame']<=2**31-1
        for values in ([probe['logits'],h['layers'][-1]['bias']] if probe['version']==PROBE_VERSION else [probe['logits']]):
            assert len(values)==25 and all(math.isfinite(v) and abs(v)<=100 for v in values) and max(values)-min(values)<=60
        if probe['version']==PROBE_VERSION:assert all(v==0 for row in h['layers'][-1]['weight'] for v in row),'Probe requires constant baseline head'
        if shared:assert all(abs(v)<=40 for v in probe['logits'])
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
        return self.with_geometry(causal,timing)[0]

    def with_geometry(self,causal,timing):
        with torch.no_grad():
            c=causal.detach().double();t=timing.detach().double()
            motor=self.motor((motor_inputs(c,t)-self.motor_mean)/self.motor_std)
            g=projected_geometry(c,motor,False)
            return (((torch.cat((c,g),-1)-self.shot_mean)/self.shot_std)*self.shot_weight).sum(-1)+self.shot_bias,g


class JointAim(nn.Module):
    def __init__(self,model,*,allow_probe=False,allow_shared=False,allow_target_sampling=False):
        super().__init__();validate(model,allow_probe=allow_probe,allow_shared=allow_shared,allow_target_sampling=allow_target_sampling)
        from combat_target_sampling import validate as validate_target_sampling
        self.target_sampling=validate_target_sampling(model.get("target_sampling"))
        probe=model.get('diagnostic_joint_aim_probe')
        self.probe_frame=probe['frame'] if probe else None
        self.probe_version=probe['version'] if probe else None
        self.register_buffer('probe_logits',torch.tensor(probe['logits'],device='cuda',dtype=torch.float32) if probe else None)
        self.head=candidates.AimCandidateHead()
        with torch.no_grad():
            for index,layer in zip((0,2),model['aim_candidate_head']['layers']):
                self.head.layers[index].weight.copy_(torch.tensor(layer['weight'],device='cuda',dtype=torch.float32))
                self.head.layers[index].bias.copy_(torch.tensor(layer['bias'],device='cuda',dtype=torch.float32))
        self.hints=FrozenHints(model['aim_candidate_head']['predictor'])
        self.shared=None
        shared=model['aim_candidate_head'].get('shared_scorer')
        self.geometry_head=bool(shared and shared['kind']==GEOMETRY_KIND)
        if shared:
            width,hidden=(31,8) if self.geometry_head else (191,16)
            self.shared=nn.Sequential(nn.Linear(width,hidden,device='cuda'),nn.ReLU(),nn.Linear(hidden,1,device='cuda'))
            with torch.no_grad():
                for index,layer in zip((0,2),shared['layers']):
                    self.shared[index].weight.copy_(torch.tensor(layer['weight'],device='cuda',dtype=torch.float32))
                    self.shared[index].bias.copy_(torch.tensor(layer['bias'],device='cuda',dtype=torch.float32))
            self.register_buffer('shared_beta',torch.tensor(shared['beta'],device='cuda',dtype=torch.float32))

    def action_context(self,features,source,target,mode,vertical,weapon):
        c=fire_inputs(features,target,mode,source,vertical,weapon)
        return torch.cat((c,features[:,810:814].detach()),-1)

    def forward(self,features,source,target,mode,vertical,weapon,timing,frames=None):
        c=self.action_context(features,source,target,mode,vertical,weapon)
        active=candidates.eligible(c)
        if self.probe_frame is not None:
            assert frames is not None and frames.is_cuda and frames.dtype==torch.long and frames.shape==(len(c),),'Probe scoring requires explicit recorded frames'
        if self.probe_version==PROBE_VERSION:
            constant=self.head.layers[-1].bias[None,:].expand(len(c),-1)
            return torch.where((frames==self.probe_frame)[:,None],self.probe_logits[None,:],constant),active
        # Ineligible rows contribute no categorical event. Avoid forecasting
        # their absent bbox while retaining a differentiable zero head branch.
        hints=torch.zeros((len(c),25),device='cuda',dtype=features.dtype)
        candidate_context=c[:,None,:].expand(-1,25,-1).clone() if self.shared is not None else None
        geometry=torch.zeros((len(c),25,29),device='cuda',dtype=torch.float64) if self.geometry_head else None
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
            if self.geometry_head:
                predicted,g=self.hints.with_geometry(expanded.reshape(-1,185),t)
                geometry[active]=g.reshape(count,25,29)
                hints[active]=predicted.reshape(count,25).to(features.dtype)
            else:hints[active]=self.hints(expanded.reshape(-1,185),t).reshape(count,25).to(features.dtype)
            if candidate_context is not None:candidate_context[active]=expanded
        if self.shared is not None:
            p=hints.sigmoid();ids=torch.arange(25,device='cuda').expand(len(c),-1)
            offsets=torch.tensor(candidates.OFFSETS,device='cuda',dtype=features.dtype)
            deltas=torch.stack((offsets[ids//5],offsets[ids%5]),-1)/5
            if self.geometry_head:
                relative=((geometry-geometry[:,:1])/self.hints.shot_std[185:214]).clamp(-20,20).to(features.dtype)
                shared_inputs=torch.cat((relative,(p-p[:,:1])[:,:,None],deltas.square().sum(-1)[:,:,None]),-1)
            else:
                normalized=((candidate_context-self.hints.shot_mean[:185].to(features.dtype))/self.hints.shot_std[:185].to(features.dtype)).clamp(-20,20)
                shared_inputs=torch.cat((normalized,deltas,p[:,:,None],p[:,:1,None].expand(-1,25,-1),(p-p[:,:1])[:,:,None],(ids==0).to(features.dtype)[:,:,None]),-1)
            logits=self.shared(shared_inputs).squeeze(-1).clamp(-20,20)+self.shared_beta*(ids==0).to(features.dtype)
        else:logits=self.head(candidates.inputs(features,c,hints,timing.to(features.dtype)))
        if self.probe_version==POLICY_PROBE_VERSION:
            logits=torch.where((frames==self.probe_frame)[:,None],self.probe_logits[None,:],logits)
        return logits,active


def events(rows,features,target,mode,vertical,weapon):
    """Validate explicit source/category records on CUDA, before scoring."""
    annotations=[r['sample'].get('aim_candidate') for r in rows]
    source=torch.tensor([a['source_latent'] if a else r['sample']['latent'] for r,a in zip(rows,annotations)],device='cuda',dtype=torch.float64)
    choice=torch.tensor([a['candidate'] if a else 0 for a in annotations],device='cuda',dtype=torch.long)
    timing=torch.tensor([a['timing'] if a else [0.,0.] for a in annotations],device='cuda',dtype=torch.float64)
    active=candidates.eligible(torch.cat((fire_inputs(features,target,mode,source,vertical,weapon),features[:,810:814]),-1))
    present=torch.tensor([a is not None for a in annotations],device='cuda',dtype=torch.bool)
    assert torch.equal(active,present),'Missing or ineligible joint aim event'
    assert all(a is None or (set(a)=={'source_latent','candidate','candidates','log_probability','timing'} | ({'logits'} if 'logits' in a else set()) and a['candidates']==25 and math.isfinite(a['log_probability'])) for a in annotations)
    final=torch.tensor([r['sample']['latent'] for r in rows],device='cuda',dtype=torch.float64)
    expected=candidates.apply_offsets(source,mode,choice)
    assert bool(torch.isfinite(timing).all()) and float((expected-final).abs().max())<1e-8,'Recorded correction transform differs'
    return source.to(features.dtype),choice,timing.to(features.dtype),active


def probabilities(joint,raw,std,source,final,attack,vertical,weapon,features,target,mode,choice,timing,motion_branch=None,reference_branch=None,fire_branch=None,*,frames=None,score_inputs=None,movement_shift=None):
    from combat_precision_head import probabilities as original
    lp,ent=original(raw,std,source,attack,vertical,weapon,features,target,mode,motion_branch,reference_branch,fire_branch,movement_shift=movement_shift)
    from combat_target_sampling import correction as target_correction
    correction=target_correction(raw,features,target,joint.target_sampling)
    if correction is not None:
        lp=lp+correction[0];ent=ent+correction[1]
    # Gaussian terms keep source latent. Only fire sees the final aim.
    before=torch.distributions.Bernoulli(logits=fire_logits(fire_branch,features,raw,target,mode,source,vertical,weapon))
    after=torch.distributions.Bernoulli(logits=fire_logits(fire_branch,features,raw,target,mode,final,vertical,weapon))
    sf,sz,st=score_inputs if score_inputs is not None else (features,source,timing)
    logits,active=joint(sf,sz,target,mode,vertical,weapon,st,frames=frames)
    lp=lp+after.log_prob(attack)-before.log_prob(attack)+candidates.log_probability(logits,choice,active)
    ent=ent+after.entropy()-before.entropy()+candidates.entropy(logits,active)
    assert bool(torch.isfinite(lp).all()) and bool(torch.isfinite(ent).all())
    return lp,ent,logits,active

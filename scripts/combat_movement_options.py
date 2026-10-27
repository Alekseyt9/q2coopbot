"""Causal five-frame movement choices. All numerical network work is CUDA."""
import math
from ppo_combat import torch, nn, layers
from combat_aim_candidates import frame_uniforms

VERSION='combat_five_frame_movement_options_v1'
DISTRIBUTION='combat_joint_aim_movement_options_v3'
SAMPLER='frame_hash_movement_options_v1'
SALT=0x3c6ef372fe94f82b
SHIFTS=[[0.,0.],[-1.,0.],[1.,0.],[0.,-1.],[0.,1.]]
WIDTH=1635


def validate(model):
    h=model.get('movement_options')
    if h is None:
        assert model.get('action_distribution')!=DISTRIBUTION,'Movement options require head'
        return
    assert set(h)=={'version','sampler','enabled','duration_frames','mean_shifts','actor','value'}
    assert h['version']==VERSION and h['sampler']==SAMPLER and type(h['enabled']) is bool
    assert h['duration_frames']==5 and h['mean_shifts']==SHIFTS
    assert model['action_distribution']==DISTRIBUTION and model['feature_version']=='combat_features_v13_memory_static_routes'
    assert model.get('attention') and not model.get('memory') and not model.get('entity_attention') and not model.get('deterministic')
    assert model.get('aim_candidate_head') and model.get('conditional_fire')
    assert not any(k.startswith('diagnostic_') for k in model)
    assert not any(model.get(k) for k in ('wide_aim','target_sampling','movement_inertia','execution_mode','geometry_aim_ranker','geometry_aim_exploration'))
    for name,n in (('actor',5),('value',1)):
        width=WIDTH;assert len(h[name])==3
        for layer,count in zip(h[name],(32,16,n)):
            assert set(layer)=={'weight','bias'} and len(layer['weight'])==len(layer['bias'])==count
            assert all(len(row)==width and all(math.isfinite(v) and abs(v)<=1e4 for v in [*row,b]) for row,b in zip(layer['weight'],layer['bias']))
            width=count


class MovementOptions(nn.Module):
    def __init__(self,h):
        super().__init__()
        assert torch.cuda.is_available()
        for name,count in (('actor',5),('value',1)):
            net=nn.Sequential(nn.Linear(WIDTH,32,device='cuda'),nn.ReLU(),nn.Linear(32,16,device='cuda'),nn.ReLU(),nn.Linear(16,count,device='cuda'))
            with torch.no_grad():
                for module,layer in zip((net[0],net[2],net[4]),h[name]):
                    module.weight.copy_(torch.tensor(layer['weight'],device='cuda'));module.bias.copy_(torch.tensor(layer['bias'],device='cuda'))
            setattr(self,name,net)

    def forward(self,features,raw):
        assert features.is_cuda and raw.is_cuda and features.shape[-1]==1554 and raw.shape[-1]==81
        x=torch.cat((features.detach(),raw.detach()),-1)
        return self.actor(x).clamp(-20,20),self.value(x).squeeze(-1)

    def export(self,h):
        return {**h,'actor':layers(self.actor),'value':layers(self.value)}


def verify(model,rows,context,prepared,actor):
    """Verify all provider contexts, including option roots outside loss rows."""
    validate(model)
    h=model.get('movement_options')
    n=len(rows);device='cuda';zero=torch.zeros(n,device=device)
    if h is None or not h['enabled']:
        assert all(r['sample'].get('movement_option') is None for r in rows)
        assert all(c.get('movement_option') is None for c in context)
        return torch.zeros((n,2),device=device),zero,dict(enabled=False,decisions=0,rows=0)
    assert prepared is not None and context
    x,_,_,segments=prepared
    with torch.no_grad():
        full_raw,_=actor(x)
    flat_context=[c for segment in segments for c in segment]
    locations=[(b,t) for b,segment in enumerate(segments) for t in range(len(segment))]
    bi=torch.tensor([v[0] for v in locations],device=device);ti=torch.tensor([v[1] for v in locations],device=device)
    module=MovementOptions(h)
    logits,values=module(x[bi,ti],full_raw[bi,ti])
    decisions=[];choices=[];receipts=[];lookup={}
    for segment in segments:
        previous=None
        for c in segment:
            r=c.get('movement_option');assert r is not None
            assert set(r)=={'version','started_frame','decision','choice','uniform','log_probability','value'}|({'logits'} if 'logits' in r else set())
            assert r['version']==VERSION and type(r['choice']) is int and 0<=r['choice']<5 and type(r['decision']) is bool
            decision=previous is None or c['frame']==previous['started_frame']+5
            assert r['decision']==decision
            if decision:
                assert r['started_frame']==c['frame'] and len(r['logits'])==5
            else:
                assert r['started_frame']==previous['started_frame'] and r['choice']==previous['choice']
                assert not r.get('logits') and r['uniform']==r['log_probability']==r['value']==0
            key=(c['seed'],c['index']);assert key not in lookup;lookup[key]=len(receipts)
            receipts.append(r);decisions.append(decision);choices.append(r['choice']);previous=r
    ids=torch.tensor([i for i,d in enumerate(decisions) if d],device=device,dtype=torch.long)
    recorded=torch.tensor([receipts[i]['logits'] for i,d in enumerate(decisions) if d],device=device,dtype=logits.dtype)
    value_recorded=torch.tensor([r['value'] for r,d in zip(receipts,decisions) if d],device=device,dtype=values.dtype)
    error=float((logits[ids]-recorded).abs().max());value_error=float((values[ids]-value_recorded).abs().max())
    assert error<.001 and value_error<.001,(error,value_error)
    uniform=torch.tensor([frame_uniforms(c['seed']^SALT,c['frame'],1)[0]
        for c,d in zip(flat_context,decisions) if d],device=device,dtype=torch.float64)
    recorded_uniform=torch.tensor([r['uniform'] for r,d in zip(receipts,decisions) if d],device=device,dtype=torch.float64)
    assert torch.equal(uniform,recorded_uniform)
    choice=torch.tensor(choices,device=device,dtype=torch.long)
    # Go accumulates the float64 CDF. Use recorded live logits here, after CUDA
    # numerical agreement above, so a CDF boundary cannot hide a changed draw.
    p=recorded.double().softmax(-1);selected=(uniform[:,None]>p.cumsum(-1)).sum(-1).clamp_max(4)
    assert torch.equal(selected,choice[ids]),'Option hash-CDF differs'
    logp=logits.log_softmax(-1).gather(-1,choice[:,None]).squeeze(-1)
    recorded_lp=torch.tensor([r['log_probability'] for r in receipts],device=device,dtype=logp.dtype)
    assert float((logp[ids]-recorded_lp[ids]).abs().max())<.001
    selected_rows=torch.tensor([lookup[(r['seed'],r['index'])] for r in rows],device=device,dtype=torch.long)
    for r,i in zip(rows,selected_rows.cpu().tolist()):assert r['sample']['movement_option']==receipts[i]
    mask=torch.tensor(decisions,device=device,dtype=torch.bool)
    shifts=torch.tensor(SHIFTS,device=device,dtype=logits.dtype)[choice[selected_rows]]
    return shifts,torch.where(mask,logp,0.)[selected_rows],dict(enabled=True,rows=n,context_rows=len(receipts),
        decisions=len(ids),logit_max_error=error,value_max_error=value_error,held_choices_and_cdf_verified=True)

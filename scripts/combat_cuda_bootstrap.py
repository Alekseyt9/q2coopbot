"""Batched CUDA critic bootstrap from verified provider context, no Go replay."""
import math
from ppo_combat import torch
from combat_attention import CausalAttention,EntityAttention
from ppo_recurrent import Recurrent

def next_values(value,next_features,prepared=None,selection=None,reset=None):
    assert next_features.device.type=='cuda' and next_features.ndim==2
    if isinstance(value,EntityAttention):return value.single(next_features)[:,0]
    if not isinstance(value,(CausalAttention,Recurrent)):return value(next_features)[:,0]
    assert prepared is not None
    x,indices,inverse,_=prepared;locations=indices[inverse]
    if selection is not None:locations=locations[selection]
    assert len(locations)==len(next_features) and x.device.type=='cuda'
    batch,time=locations.unbind(-1)
    reset=torch.zeros(len(locations),device='cuda',dtype=torch.bool) if reset is None else reset
    assert reset.device.type=='cuda' and reset.shape==(len(locations),)
    if isinstance(value,Recurrent):
        _,_,states=value(x);after=states[batch,time].clone();after[reset]=0
        return value(next_features[:,None,:],after[None,:,:])[0][:,0,0]
    _,tokens=value(x);window=value.window;width=tokens.shape[-1]
    history_index=time[:,None]+1-torch.arange(window-1,0,-1,device='cuda')[None,:]
    valid=(history_index>=0)&~reset[:,None]
    history=tokens[batch[:,None],history_index.clamp_min(0)]
    position=torch.where(reset,torch.zeros_like(time),time+1).to(next_features.dtype)[:,None]
    frequency=torch.exp(-math.log(10000)*torch.arange(0,width,2,device='cuda',dtype=next_features.dtype)/width)
    positional=torch.zeros(len(locations),width,device='cuda',dtype=next_features.dtype)
    positional[:,0::2]=torch.sin(position*frequency);positional[:,1::2]=torch.cos(position*frequency)
    encoded=value.encoder(next_features);query=encoded+positional
    keys=torch.cat((history,query[:,None,:]),1)
    padding=torch.cat((~valid,torch.zeros(len(locations),1,device='cuda',dtype=torch.bool)),1)
    attended,_=value.mha(query[:,None,:],keys,keys,key_padding_mask=padding,need_weights=False)
    return (value.head(encoded)+value.residual(attended[:,0,:]))[:,0]

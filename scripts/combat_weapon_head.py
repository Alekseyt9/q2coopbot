"""Masked weapon likelihood for the Go-owned PPO action contract."""
from train_combat_bc import torch
from combat_weapon_features import MASK_OFFSET, WIDTH

HEAD_VERSION='combat_masked_weapon_v1'

def feature_mask(features):
    if features.ndim!=2 or features.shape[1]!=WIDTH:
        raise ValueError('Weapon head requires V6 observations')
    values=features[:,MASK_OFFSET:WIDTH]
    if not bool(((values==0)|(values==1)).all()) or not bool((values[:,0]==1).all()):
        raise ValueError('Invalid availability mask')
    return values.bool()

def distribution(logits,mask):
    if logits.ndim!=2 or logits.shape[1]!=12 or mask is None or mask.shape!=logits.shape or mask.dtype!=torch.bool:
        raise ValueError('Invalid weapon distribution shape/mask')
    if not bool(torch.isfinite(logits).all()) or not bool(mask[:,0].all()):
        raise ValueError('Invalid weapon logits or missing keep')
    return torch.distributions.Categorical(logits=logits.masked_fill(~mask,float('-inf')))

def probabilities(raw,std,z,attack,vertical,weapon=None,mask=None):
    if raw.ndim!=2 or raw.shape[1] not in (8,20):raise ValueError('Unknown action width')
    d=torch.distributions
    normal=d.Normal(raw[:,:4],std.exp())
    fire=d.Bernoulli(logits=raw[:,4])
    pose=d.Categorical(logits=raw[:,5:8])
    lp=normal.log_prob(z).sum(1)+fire.log_prob(attack)+pose.log_prob(vertical)
    entropy=normal.entropy().sum(1)+fire.entropy()+pose.entropy()
    if raw.shape[1]==20:
        if weapon is None or weapon.ndim!=1 or len(weapon)!=len(raw) or weapon.dtype!=torch.long:
            raise ValueError('Missing weapon sample')
        if not bool(((weapon>=0)&(weapon<12)).all()):raise ValueError('Invalid weapon category')
        choice=distribution(raw[:,8:20],mask)
        if not bool(mask.gather(1,weapon[:,None]).all()):raise ValueError('Masked weapon sample')
        lp=lp+choice.log_prob(weapon)
        entropy=entropy+choice.entropy()
    elif weapon is not None or mask is not None:
        raise ValueError('Weapon sample on legacy actor')
    return lp,entropy

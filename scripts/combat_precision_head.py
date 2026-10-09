"""CUDA prototype: learned coarse/fine aim conditioned on selected target.

Requires the versioned 81-output runtime contract; never publish as a
45-output target policy.
"""
import copy
from combat_target_head import validate_model,distribution as targets,selected_means as coarse_means
from combat_weapon_head import probabilities as common_probabilities,feature_mask
from train_combat_bc import torch

VERSION='combat_target_coarse_fine_aim_v1'
WIDTH=81
FINE_OFFSET=45
MODE_OFFSET=63
FINE_DEGREES=15.
COARSE_DEGREES=180.

def validate_precision_model(model):
    if model.get('aim_mode_head') != VERSION or len(model['actor'][-1]['bias']) != WIDTH:
        raise ValueError('Invalid precision contract')
    view=copy.deepcopy(model)
    view['actor'][-1]['bias']=view['actor'][-1]['bias'][:45]
    view['actor'][-1]['weight']=view['actor'][-1]['weight'][:45]
    validate_model(view)


def migrate(model):
    validate_model(model)
    if model.get('aim_mode_head'):raise ValueError('Precision head already exists')
    result=copy.deepcopy(model)
    def expand(layer,mode_bias=False):
        if len(layer['bias'])!=45:raise ValueError('Invalid precision migration source')
        width=len(layer['weight'][0])
        layer['weight'].extend([[0.]*width for _ in range(36)])
        layer['bias'].extend([0.]*18)
        layer['bias'].extend([v for _ in range(9) for v in ((5.,-5.) if mode_bias else (0.,0.))])
    expand(result['actor'][-1],True)
    if result.get('memory'):expand(result['memory']['actor']['output'])
    if result.get('attention'):expand(result['attention']['actor']['residual'])
    result['aim_mode_head']=VERSION
    return result

def mode_distribution(raw,target):
    if raw.ndim!=2 or raw.shape[1]!=WIDTH or not bool(torch.isfinite(raw).all()):raise ValueError('Invalid precision actor')
    if target.shape!=(len(raw),) or target.dtype!=torch.long or not bool(((target>=0)&(target<=8)).all()):raise ValueError('Invalid target category')
    logits=raw[:,MODE_OFFSET:].reshape(-1,9,2).gather(1,target[:,None,None].expand(-1,1,2))[:,0,:]
    return torch.distributions.Categorical(logits=logits)

def selected_means(raw,target,mode):
    coarse=coarse_means(raw[:,:45],target)
    if mode.shape!=(len(raw),) or mode.dtype!=torch.long or not bool(((mode==0)|(mode==1)).all()):raise ValueError('Invalid aim mode')
    fine=raw[:,FINE_OFFSET:MODE_OFFSET].reshape(-1,9,2).gather(1,target[:,None,None].expand(-1,1,2))[:,0,:]
    return torch.cat((coarse[:,:2],torch.where(mode[:,None]==1,fine,coarse[:,2:4])),1)

def probabilities(raw,std,z,attack,vertical,weapon,features,target,mode):
    choice=targets(raw[:,:45],features);stage=mode_distribution(raw,target)
    if not bool(torch.isfinite(choice.log_prob(target)).all()):raise ValueError('Unavailable selected target')
    means=selected_means(raw,target,mode);common=torch.cat((means,raw[:,4:20]),1)
    lp,entropy=common_probabilities(common,std,z,attack,vertical,weapon,feature_mask(features[:,:845]))
    return lp+choice.log_prob(target)+stage.log_prob(mode),entropy+choice.entropy()+stage.entropy()

def physical_actions(latent,mode):
    if latent.ndim!=2 or latent.shape[1]!=4:raise ValueError('Invalid action latent')
    if mode.shape!=(len(latent),) or mode.dtype!=torch.long or not bool(((mode==0)|(mode==1)).all()):raise ValueError('Invalid aim mode')
    scale=torch.where(mode==1,FINE_DEGREES,COARSE_DEGREES)
    return torch.cat((latent[:,:2].tanh(),latent[:,2:4].tanh()*scale[:,None]),1)

def query_modes(angles):
    """Unexecuted geometric bootstrap; angles are normalized by 180 degrees."""
    if angles.shape[-1]!=2 or not bool(torch.isfinite(angles).all()):raise ValueError('Invalid angle queries')
    return (angles.abs().amax(-1)<=FINE_DEGREES/COARSE_DEGREES).long()

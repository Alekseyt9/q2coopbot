"""Observed target selection and target-conditioned aim, shared by all actors."""
import copy
from combat_weapon_head import probabilities as legacy_probabilities, feature_mask
from train_combat_bc import torch

HEAD_VERSION = 'combat_target_conditioned_aim_v1'
FEATURE_VERSION = 'combat_features_v7'
WIDTH = 854
OUTPUTS = 45

def validate_model(model):
    if model.get('target_head') != HEAD_VERSION or model['feature_version'] != FEATURE_VERSION:
        raise ValueError('Invalid target feature/head contract')
    if model.get('weapon_head') != 'combat_masked_weapon_v1' or model.get('entity_attention'):
        raise ValueError('Target actor requires masked weapon head; entity attention unsupported')
    if len(model['actor'][-1]['bias']) != OUTPUTS or len(model['actor'][0]['weight'][0]) != WIDTH:
        raise ValueError('Invalid target actor dimensions')

def availability(features):
    if features.ndim != 2 or features.shape[1] != WIDTH:
        raise ValueError('Target head requires V7 observations')
    valid = features[:,426:466:5]
    if not bool(((valid == 0) | (valid == 1)).all()):
        raise ValueError('Invalid observed bbox mask')
    return torch.cat((torch.ones_like(valid[:,:1],dtype=torch.bool),valid.bool()),1)

def distribution(raw, features):
    if raw.ndim != 2 or raw.shape[1] != OUTPUTS or len(raw) != len(features):
        raise ValueError('Invalid target actor shape')
    if not bool(torch.isfinite(raw).all()):
        raise ValueError('Nonfinite target outputs')
    return torch.distributions.Categorical(logits=raw[:,20:29].masked_fill(~availability(features),float('-inf')))

def selected_means(raw, target):
    if target.dtype != torch.long or target.shape != (len(raw),) or not bool(((target >= 0) & (target <= 8)).all()):
        raise ValueError('Invalid target sample')
    pairs = torch.cat((raw[:,None,2:4],raw[:,29:45].reshape(-1,8,2)),1)
    aim = pairs.gather(1,target[:,None,None].expand(-1,1,2))[:,0,:]
    return torch.cat((raw[:,:2],aim),1)

def probabilities(raw,std,z,attack,vertical,weapon,features,target):
    choice = distribution(raw,features)
    means = selected_means(raw,target)
    if not bool(availability(features).gather(1,target[:,None]).all()):
        raise ValueError('Unavailable target sample')
    common = torch.cat((means,raw[:,4:20]),1)
    lp,entropy = legacy_probabilities(common,std,z,attack,vertical,weapon,feature_mask(features[:,:845]))
    return lp + choice.log_prob(target), entropy + choice.entropy()

def migrate(model):
    """Preserve physical outputs; no optimizer moments survive changed shapes."""
    if model['kind'] != 'combat_ppo_v1' or model['feature_version'] != 'combat_features_v6' or model.get('target_head'):
        raise ValueError('Migration requires an unmodified V6 PPO actor')
    if model.get('weapon_head') != 'combat_masked_weapon_v1' or model.get('entity_attention'):
        raise ValueError('Unsupported migration source')
    result = copy.deepcopy(model)
    for name in ('actor','value'):
        first = result[name][0]
        if any(len(row) != 845 for row in first['weight']):
            raise ValueError('Invalid source feature width')
        for row in first['weight']: row.extend([0.] * 9)
    def expand(layer, selector_bias=False):
        if len(layer['bias']) != 20: raise ValueError('Invalid source action width')
        layer['weight'].extend([[0.] * len(layer['weight'][0]) for _ in range(9)])
        layer['bias'].extend([-4. if selector_bias else 0.] + [0.] * 8)
        for _ in range(8):
            for i in (2,3):
                layer['weight'].append(list(layer['weight'][i]));layer['bias'].append(layer['bias'][i])
    expand(result['actor'][-1],True)
    if result.get('memory'): expand(result['memory']['actor']['output'])
    if result.get('attention'): expand(result['attention']['actor']['residual'])
    result.update(feature_version=FEATURE_VERSION,target_head=HEAD_VERSION)
    validate_model(result)
    return result

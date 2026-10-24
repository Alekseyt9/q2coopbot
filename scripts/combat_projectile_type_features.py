"""Experimental appended projectile type bits; metadata plus CUDA encoding only."""
PARENT_VERSION = 'combat_features_v11_body_ablated'
VERSION = 'combat_features_v12_projectile_types'
ABLATED_VERSION = 'combat_features_v12_projectile_types_ablated'
NEUTRAL_VERSION = 'combat_features_v12_projectile_types_neutral'
VERSIONS = (VERSION, NEUTRAL_VERSION, ABLATED_VERSION)
PARENT_WIDTH = 1469
TAIL_WIDTH = 20
WIDTH = PARENT_WIDTH + TAIL_WIDTH


def encode_tail(observations, version=VERSION):
    from ppo_combat import torch
    assert torch.cuda.is_available() and version in (VERSION, ABLATED_VERSION, NEUTRAL_VERSION)
    kinds = {'grenade': 0, 'hand_grenade': 1, 'rocket': 2, 'blaster_bolt': 3}
    codes = []
    for observation in observations:
        # Match the legacy stable distance sort exactly, including equal ranges.
        items = sorted(observation.get('visible_projectiles') or [], key=lambda p: p['distance'])[:4]
        row = [kinds.get(p['class'], 4) for p in items]
        codes.append(row + [5] * (4 - len(row)))
    indices = torch.tensor(codes, device='cuda', dtype=torch.int64)
    assert indices.shape == (len(observations), 4)
    if version == NEUTRAL_VERSION:
        indices = torch.where(indices < 5, 4, indices)
    result = torch.nn.functional.one_hot(indices, num_classes=6)[..., :5].float().flatten(1)
    if version == ABLATED_VERSION:
        result.zero_()
    assert result.is_cuda and result.shape == (len(observations), TAIL_WIDTH)
    return result

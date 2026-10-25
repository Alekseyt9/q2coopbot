"""Versioned appended client-body channels; legacy helpers use only the v10 prefix."""
from combat_projectile_type_features import VERSIONS as PROJECTILE_VERSIONS, WIDTH as PROJECTILE_WIDTH
from combat_memory_route_features import VERSION as ROUTE_VERSION, WIDTH as ROUTE_WIDTH
LEGACY_VERSION = 'combat_features_v10'
VERSION = 'combat_features_v11'
ABLATED_VERSION = 'combat_features_v11_body_ablated'
VERSIONS = (LEGACY_VERSION, VERSION, ABLATED_VERSION, *PROJECTILE_VERSIONS, ROUTE_VERSION)
LEGACY_WIDTH = 1147
TAIL_WIDTH = 322
WIDTH = LEGACY_WIDTH + TAIL_WIDTH
SCOPE = 'combat_body_input_ppo_v1'


def legacy(features):
    assert features.is_cuda and features.shape[-1] in (LEGACY_WIDTH, WIDTH, PROJECTILE_WIDTH, ROUTE_WIDTH), 'Invalid equipped/body feature schema'
    return features[..., :LEGACY_WIDTH]


def encode_tail(observations, version=VERSION):
    """Encode recorded client metadata with all vector math on CUDA."""
    from ppo_combat import torch
    assert torch.cuda.is_available() and version in (VERSION, ABLATED_VERSION)
    records, availability, truncated, view_yaws = [], [], [], []
    for observation in observations:
        group = observation.get('observed_dynamic_bodies')
        availability.append(group is not None)
        truncated.append(bool(group and group['truncated']))
        view_yaws.append(observation['view_angles'][1])
        slots = []
        if group is not None:
            assert group['version'] == 'observed_dynamic_bodies_v1' and group['frame'] == observation['identity']['frame']
            assert group['range'] == 512 and group['capacity'] == 16 and len(group['bodies']) <= 16
            assert group['bodies'] == sorted(group['bodies'], key=lambda body: (body['distance'], body['id']))
            for body in group['bodies']:
                known = body['protocol_bounds_known']
                assert known == ('protocol_mins' in body and 'protocol_maxs' in body)
                path = body.get('observed_model', '').lower()
                slots.append([1, *body['relative'], body['distance'], known,
                    *(body['protocol_mins'] if known else (0, 0, 0)),
                    *(body['protocol_maxs'] if known else (0, 0, 0)),
                    body['observed_death_animation'], '/monsters/' in path, 'players/' in path,
                    body.get('observed_class') == 'misc_explobox', bool(path), body['observed_angles'][1], 0,
                    body['observed_animation_frame']])
        slots.extend([[0]*20 for _ in range(16-len(slots))])
        records.append(slots)
    raw = torch.tensor(records, device='cuda', dtype=torch.float64)
    assert raw.shape == (len(observations), 16, 20) and bool(torch.isfinite(raw).all())
    result = raw.clone()
    yaw = torch.tensor(view_yaws, device='cuda', dtype=torch.float64) * (360/65536) * (torch.pi/180)
    cosine, sine = yaw.cos()[:, None], yaw.sin()[:, None]
    result[:, :, 1] = (raw[:, :, 1]*cosine+raw[:, :, 2]*sine)/512
    result[:, :, 2] = (-raw[:, :, 1]*sine+raw[:, :, 2]*cosine)/512
    result[:, :, 3:5] /= 512
    result[:, :, 6:12] /= 64
    relative_yaw = raw[:, :, 17]*(torch.pi/180)-yaw[:, None]
    result[:, :, 17] = relative_yaw.sin()*raw[:, :, 0]
    result[:, :, 18] = relative_yaw.cos()*raw[:, :, 0]
    result[:, :, 19] /= 512
    header = torch.tensor(list(zip(availability, truncated)), device='cuda', dtype=torch.float64)
    result = torch.cat((header, result.flatten(1)), -1).float()
    assert result.shape == (len(observations), TAIL_WIDTH) and bool(torch.isfinite(result).all())
    if version == ABLATED_VERSION:
        result.zero_()
    return result

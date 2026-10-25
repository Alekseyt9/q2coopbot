"""Static route observations to remembered endpoints; CUDA vector encoding."""
import math

VERSION = 'combat_features_v13_memory_static_routes'
OBSERVATION_VERSION = 'combat_observed_memory_static_routes_v1'
PARENT_WIDTH = 1489
TAIL_WIDTH = 65
WIDTH = PARENT_WIDTH + TAIL_WIDTH


def encode_tail(observations):
    from ppo_combat import torch
    assert torch.cuda.is_available()
    rows, yaws = [], []
    for o in observations:
        row = [0.] * TAIL_WIDTH
        yaws.append(o['view_angles'][1])
        group = o.get('remembered_static_routes')
        if group is not None:
            remembered = o.get('remembered_threats') or []
            assert group['version'] == OBSERVATION_VERSION and group['frame'] == o['identity']['frame']
            assert len(group['hints']) <= 8 and len(remembered) <= 8
            row[0] = 1
            seen = set()
            for h in group['hints']:
                identity, age = h['observed_id'], h['memory_age_frames']
                assert identity > 0 and identity not in seen and 0 <= age <= 200
                seen.add(identity)
                slots = [i for i, t in enumerate(remembered) if t['id'] == identity and t['age_frames'] == age]
                assert len(slots) == 1
                known, found, direct = h['static_geometry_known'], h['static_route_found'], h['static_direct_hull_clear']
                distance, kind, point = h['static_horizontal_path_length'], h.get('static_travel_kind', 0), h.get('next_static_waypoint_relative')
                assert math.isfinite(distance) and 0 <= distance <= 65536 and 0 <= kind <= 16
                assert known or not (found or direct or distance or point is not None or kind)
                assert found == (point is not None) and (found or not (distance or kind))
                if point is not None:
                    assert len(point) == 3 and all(math.isfinite(v) and abs(v) <= 32768 for v in point)
                at = 1 + slots[0]*8
                row[at:at+8] = [known, found, direct, min(16, distance/4096), *(point or [0., 0., 0.]), kind/16]
        rows.append(row)
    x = torch.tensor(rows, device='cuda', dtype=torch.float64)
    yaw = torch.tensor(yaws, device='cuda', dtype=torch.float64)*(360/65536)*(torch.pi/180)
    slots = x[:, 1:].reshape(-1, 8, 8)
    px, py = slots[..., 4].clone(), slots[..., 5].clone()
    slots[..., 4] = (px*yaw.cos()[:, None]+py*yaw.sin()[:, None])/512
    slots[..., 5] = (-px*yaw.sin()[:, None]+py*yaw.cos()[:, None])/512
    slots[..., 6] /= 512
    assert x.shape == (len(observations), TAIL_WIDTH) and bool(torch.isfinite(x).all())
    return x.float()

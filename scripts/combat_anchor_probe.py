"""Diagnostic actuation only; source-event likelihood is NOT actuation density."""
import math

VERSION = 'combat_one_frame_observed_mg_anchor_v1'


def validate(model):
    h = model.get('diagnostic_anchor_probe')
    if h is None:
        return
    assert set(h) == {'version', 'frame', 'mode'} and h['version'] == VERSION
    assert type(h['frame']) is int and 0 < h['frame'] <= 2**31-1
    assert h['mode'] in ('identity', 'observed_mg') and not model.get('deterministic')
    assert model.get('action_distribution') == 'combat_joint_aim_candidates_v1'
    assert model.get('aim_candidate_head') and model.get('target_head') and model.get('weapon_head') and model.get('aim_mode_head')
    assert not any(model.get(k) for k in ('diagnostic_joint_aim_probe', 'diagnostic_movement_probe',
        'diagnostic_weapon_probe', 'diagnostic_continuation_probe', 'execution_mode', 'geometry_aim_ranker', 'geometry_aim_exploration'))


def verify(model, rows):
    from train_combat_bc import torch
    from combat_body_features import legacy
    validate(model)
    h = model['diagnostic_anchor_probe']
    selected = []
    for row in rows:
        r = row['sample'].get('anchor_probe')
        if row['frame'] == h['frame']:
            assert r is not None, 'Missing diagnostic anchor receipt'
            assert set(r) == {'version', 'frame', 'mode', 'target_entity', 'source_degrees', 'returned_degrees'}
            assert (r['version'], r['frame'], r['mode']) == (VERSION, h['frame'], h['mode'])
            assert type(r['target_entity']) is int and r['target_entity'] > 0
            selected.append(row)
        else:
            assert r is None, 'Unexpected anchor outside branch'
    assert len(selected) == 1, 'Diagnostic root missing or repeated'
    row = selected[0]; s = row['sample']; r = s['anchor_probe']
    x = legacy(torch.tensor([row['features']], device='cuda', dtype=torch.float64))
    target = s.get('target', 0)
    assert 1 <= target <= 8 and s.get('aim_mode', 0) == 0
    e = x[:, 73:169].reshape(1, 8, 12)[:, target-1]
    t = x[:, 466:786].reshape(1, 8, 40)[:, target-1]
    assert bool((e[:, 0] > 0).all()) and bool((e[:, 11] == 1).all()) and bool((t[:, 36] > 0).all())
    assert bool((x[:, 1121+4] == 1).all()) and bool((x[:, 810] == 1).all()), 'MG/current recoil required'
    yaw = torch.atan2(x[:, 6], x[:, 7]); pitch = torch.atan2(x[:, 8], x[:, 9])
    local = e[:, 2:5]*512
    position = torch.stack((yaw.cos()*local[:, 0]-yaw.sin()*local[:, 1],
        yaw.sin()*local[:, 0]+yaw.cos()*local[:, 1], local[:, 2]), -1)
    bottom, top = t[:, 38]*64, t[:, 39]*64
    height = torch.where(top-bottom < 16, (top+bottom)/2,
        torch.minimum(torch.maximum(torch.full_like(top, 22), bottom+8), top-8))
    eye = torch.where(x[:, 3] > 0, -2., 22.)
    position[:, 2] += height-eye+8
    kick = x[:, 811:814]*32
    wrap = lambda a: torch.remainder(a+180, 360)-180
    anchor = torch.stack((wrap(torch.atan2(position[:, 1], position[:, 0])*180/math.pi-yaw*180/math.pi-kick[:, 1]),
        wrap(-torch.atan2(position[:, 2], torch.linalg.vector_norm(position[:, :2], dim=-1))*180/math.pi-pitch*180/math.pi-kick[:, 0])), -1)
    source = torch.tensor([s['latent'][2:]], device='cuda', dtype=torch.float64).tanh()*180
    recorded_source = torch.tensor([r['source_degrees']], device='cuda', dtype=torch.float64)
    recorded_returned = torch.tensor([r['returned_degrees']], device='cuda', dtype=torch.float64)
    source_error = float((source-recorded_source).abs().max())
    returned_error = float(((source if h['mode'] == 'identity' else anchor)-recorded_returned).abs().max())
    assert math.isfinite(source_error) and source_error < 1e-10
    assert math.isfinite(returned_error) and returned_error < 1e-8
    if h['mode'] == 'identity':
        assert r['source_degrees'] == r['returned_degrees'], 'Identity must be exact'
    return dict(version=VERSION, frame=h['frame'], mode=h['mode'], active_rows=1,
        source_degrees_max_error=source_error, returned_degrees_max_error=returned_error,
        source_event_likelihood_only=True, ordinary_training_eligible=False)

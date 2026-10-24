"""Acquisition-only one-frame weapon CDF intervention and CUDA receipt checks."""
import math
from train_combat_bc import torch
from combat_body_features import VERSIONS

VERSION = 'combat_one_frame_weapon_choice_v1'


def validate(model):
    p = model.get('diagnostic_weapon_probe')
    if p is None:
        return
    assert set(p) == {'version', 'frame', 'mode', 'expected_equipped_weapon', 'forced_weapon'}
    assert p['version'] == VERSION and type(p['frame']) is int and 0 < p['frame'] <= 2**31-1
    assert p['mode'] in ('identity', 'force')
    assert type(p['expected_equipped_weapon']) is int and p['expected_equipped_weapon'] == 1
    assert type(p['forced_weapon']) is int and p['forced_weapon'] == 2
    assert model['feature_version'] in VERSIONS and not model.get('deterministic')
    assert model['weapon_head'] == 'combat_masked_weapon_v1'
    assert model['target_head'] == 'combat_target_conditioned_aim_v1'
    assert model['action_distribution'] == 'combat_joint_aim_candidates_v1'
    assert model.get('aim_candidate_head') and model.get('conditional_fire')
    assert not any(model.get(k) for k in ('diagnostic_joint_aim_probe', 'diagnostic_movement_probe',
        'diagnostic_continuation_probe', 'execution_mode', 'geometry_aim_ranker', 'geometry_aim_exploration'))


def adjust_log_probability(model, frames, raw, features, weapon, lp):
    validate(model)
    assert all(t.is_cuda for t in (frames, raw, features, weapon, lp))
    p = model['diagnostic_weapon_probe']
    active = frames == p['frame']
    if p['mode'] == 'identity':
        return lp
    assert bool((weapon[active] == p['forced_weapon']).all())
    mask = features[:, 833:845].bool()
    old = raw[:, 8:20].masked_fill(~mask, float('-inf')).log_softmax(-1)
    source_lp = old.gather(1, weapon[:, None]).squeeze(1)
    # Only the weapon term changes. Conditional fire/aim already see the actual
    # selected weapon through the ordinary scorer's contextual inputs.
    return torch.where(active, lp - source_lp, lp)


def verify_draw_receipts(model, rows, raw, std, source, final, features, target, mode, vertical, weapon, actor):
    validate(model)
    assert all(t.is_cuda for t in (raw, std, source, final, features, target, mode, vertical, weapon))
    p = model['diagnostic_weapon_probe']
    indices, receipts = [], []
    keys = {'version', 'frame', 'mode', 'source_probabilities', 'applied_probabilities',
            'weapon_uniform', 'fire_uniform', 'gaussian_noise'}
    for i, row in enumerate(rows):
        receipt = row['sample'].get('weapon_probe')
        if row['frame'] != p['frame']:
            assert receipt is None, 'Weapon receipt outside declared frame'
            continue
        assert receipt is not None and set(receipt) == keys
        assert receipt['version'] == VERSION and receipt['mode'] == p['mode'] and receipt['frame'] == p['frame']
        for key, length in (('source_probabilities', 12), ('applied_probabilities', 12), ('gaussian_noise', 4)):
            assert len(receipt[key]) == length and all(math.isfinite(v) for v in receipt[key])
        assert all(math.isfinite(receipt[k]) and 0 <= receipt[k] < 1 for k in ('weapon_uniform', 'fire_uniform'))
        indices.append(i); receipts.append(receipt)
    assert len(indices) == 1, 'The single root must occur exactly once'
    idx = torch.tensor(indices, device='cuda', dtype=torch.long)
    x = features[idx]
    assert bool(((x[:, 814] == 1) & (x[:, 815] == 1) & (x[:, 1122] == 1) &
                 (x[:, 835] == 1) & (target[idx] > 0)).all()), 'Root inventory/weapon/target guards differ'
    mask = features[:, 833:845].bool()
    predicted = raw[:, 8:20].double().masked_fill(~mask, float('-inf')).softmax(-1)[idx]
    recorded = torch.tensor([r['source_probabilities'] for r in receipts], device='cuda', dtype=torch.float64)
    assert bool(((recorded >= 0) & (recorded <= 1)).all())
    probability_error = float((predicted - recorded).abs().max())
    assert probability_error < .0002, probability_error
    assert float((recorded.sum(-1) - 1).abs().max()) < 1e-12
    expected = recorded.clone()
    if p['mode'] == 'force':
        expected.zero_(); expected[:, p['forced_weapon']] = 1
    applied = torch.tensor([r['applied_probabilities'] for r in receipts], device='cuda', dtype=torch.float64)
    assert torch.equal(expected, applied)
    uniform = torch.tensor([r['weapon_uniform'] for r in receipts], device='cuda', dtype=torch.float64)
    chosen = ((uniform[:, None] <= applied.cumsum(-1)) & (applied > 0)).to(torch.long).argmax(-1)
    assert torch.equal(chosen, weapon[idx]), 'Weapon draw/category differs'
    from combat_precision_head import selected_means
    from combat_conditional_fire import logits as fire_logits
    means = selected_means(raw, target, mode, features, source[:, :2],
                          actor.sampled_motion_aim_branch, vertical, actor.motor_branch)
    noise = torch.tensor([r['gaussian_noise'] for r in receipts], device='cuda', dtype=means.dtype)
    noise_error = float((means[idx] + noise - source[idx]).abs().max())
    assert noise_error < .001, noise_error
    fire = fire_logits(actor.conditional_fire_branch, features, raw, target, mode, final, vertical, weapon)
    fu = torch.tensor([r['fire_uniform'] for r in receipts], device='cuda', dtype=fire.dtype)
    actual = torch.tensor([rows[i]['sample']['attack'] for i in indices], device='cuda', dtype=torch.bool)
    assert torch.equal(fu < fire[idx].sigmoid(), actual), 'Weapon-conditioned fire draw differs'
    return {'root_rows': len(indices), 'source_probability_error': probability_error, 'noise_error': noise_error}

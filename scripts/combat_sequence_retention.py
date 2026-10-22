"""Full target-conditioned policy KL on pinned own-policy training histories."""
import math
import pathlib
import collections
from process_combat_architecture_pool import read, sha
from combat_target_head import availability
from combat_weapon_head import feature_mask
from train_combat_bc import torch

VERSION = 'combat_sequence_retention_v1'
CONDITIONAL_FIRE_EXPECTATION = 'fixed_behavior_complete_action_conditional_fire_v1'


def categorical_kl(teacher, student, valid=None):
    if valid is None:
        valid = torch.ones_like(teacher, dtype=torch.bool)
    assert valid.shape == teacher.shape == student.shape and bool(valid.any(-1).all())
    before = torch.log_softmax(teacher.masked_fill(~valid, float('-inf')), -1)
    after = torch.log_softmax(student.masked_fill(~valid, float('-inf')), -1)
    # Remove undefined -inf-minus--inf before multiplication/backpropagation.
    delta = before.masked_fill(~valid, 0) - after.masked_fill(~valid, 0)
    return (before.exp() * delta).sum(-1)


def normal_kl(teacher, student, teacher_std, std):
    return (std - teacher_std +
            (teacher_std.mul(2).exp() + (teacher - student).square()) /
            (2 * std.mul(2).exp()) - .5).sum(-1)


def policy_kl(raw, std, teacher, teacher_std, features, motion_branch=None, teacher_motion_branch=None, movement=None, reference_branch=None, teacher_reference_branch=None, fire_branch=None, teacher_fire_branch=None, fire_context=None):
    """KL(teacher || current), including all available targets and aim modes.

    Inputs are outputs from the same actual sequence histories. Target/mode
    branches are weighted by teacher probabilities, never just sampled choices.
    Common tanh/aim-scale transforms are identical conditional on each mode.
    """
    conditional_fire=fire_branch is not None or teacher_fire_branch is not None
    assert not getattr(motion_branch,'requires_conditional_fire',False) or fire_branch is not None, 'Conditional fire requires its explicit conditional retention contract'
    assert not getattr(teacher_motion_branch,'requires_conditional_fire',False) or teacher_fire_branch is not None, 'Conditional fire requires its explicit conditional retention contract'
    assert raw.is_cuda and teacher.is_cuda and std.is_cuda and teacher_std.is_cuda
    assert features.is_cuda and raw.shape == teacher.shape and raw.shape[1] in (45, 81)
    assert std.shape == teacher_std.shape == (4,)
    assert bool(torch.isfinite(raw).all()) and bool(torch.isfinite(teacher).all())
    assert bool(torch.isfinite(std).all()) and bool(torch.isfinite(teacher_std).all())
    current, previous = raw.double(), teacher.detach().double()
    std, teacher_std = std.double(), teacher_std.detach().double()
    valid = availability(features)
    target_probs = torch.softmax(previous[:,20:29].masked_fill(~valid, float('-inf')), -1)
    terms = normal_kl(previous[:,:2], current[:,:2], teacher_std[:2], std[:2])
    distributions = torch.distributions
    before_fire,after_fire=previous[:,4],current[:,4]
    if conditional_fire:
        from combat_conditional_fire import logits as fire_logits
        assert raw.shape[1]==81 and fire_context is not None and len(fire_context)==5
        assert all(value.is_cuda and not value.requires_grad for value in fire_context)
        # One fixed behavior action per history estimates the teacher expectation
        # over target/mode/movement/aim/pose/weapon. Never redraw on backtracking.
        target,mode,latent,vertical,weapon=fire_context
        with torch.no_grad():
            before_fire=fire_logits(teacher_fire_branch,features,previous,target,mode,latent,vertical,weapon)
        after_fire=fire_logits(fire_branch,features,current,target,mode,latent,vertical,weapon)
    else:
        assert fire_context is None, 'Fire context supplied without conditional branches'
    terms = terms + distributions.kl_divergence(
        distributions.Bernoulli(logits=before_fire), distributions.Bernoulli(logits=after_fire))
    terms = terms + categorical_kl(previous[:,5:8], current[:,5:8])
    terms = terms + categorical_kl(previous[:,8:20], current[:,8:20], feature_mask(features[:,:845]))
    terms = terms + categorical_kl(previous[:,20:29], current[:,20:29], valid)
    coarse_before = torch.cat((previous[:,None,2:4], previous[:,29:45].reshape(-1,8,2)), 1)
    coarse_after = torch.cat((current[:,None,2:4], current[:,29:45].reshape(-1,8,2)), 1)
    coarse = normal_kl(coarse_before, coarse_after, teacher_std[2:], std[2:])
    if raw.shape[1] == 81:
        mode_before, mode_after = previous[:,63:81].reshape(-1,9,2), current[:,63:81].reshape(-1,9,2)
        modes = mode_before.softmax(-1)
        fine_before, fine_after = previous[:,45:63].reshape(-1,9,2), current[:,45:63].reshape(-1,9,2)
        if motion_branch is not None or teacher_motion_branch is not None:
            from combat_sampled_motion_aim import correction, uses_vertical
            assert movement is not None and movement.shape == (len(raw),2) and movement.is_cuda
            assert bool(torch.isfinite(movement).all()) and not movement.requires_grad
            # Fixed native behavior movement supplies one MC draw/history for
            # E_teacher(move)[KL(teacher aim|move || current aim|move)]. Sum all
            # targets/modes analytically. Never redraw during guarded retries.
            targets=torch.arange(9,device=raw.device).repeat_interleave(2).repeat(len(raw))
            stages=torch.tensor([0,1],device=raw.device).repeat(len(raw)*9)
            expanded_x=features.repeat_interleave(18,dim=0)
            expanded_move=movement.repeat_interleave(18,dim=0)
            vertical_conditioned=uses_vertical(motion_branch) or uses_vertical(teacher_motion_branch)
            conditional=[]
            for v in range(3 if vertical_conditioned else 1):
                vertical=torch.full_like(targets,v) if vertical_conditioned else None
                before_aim=torch.stack((coarse_before,fine_before),2)
                after_aim=torch.stack((coarse_after,fine_after),2)
                if teacher_motion_branch is not None:
                    with torch.no_grad():
                        delta=correction(teacher_motion_branch,expanded_x,teacher.detach().repeat_interleave(18,dim=0),expanded_move,targets,stages,vertical,teacher_reference_branch)
                    before_aim=before_aim+delta.reshape(-1,9,2,2).double()
                if motion_branch is not None:
                    delta=correction(motion_branch,expanded_x,raw.repeat_interleave(18,dim=0),expanded_move,targets,stages,vertical,reference_branch)
                    after_aim=after_aim+delta.reshape(-1,9,2,2).double()
                conditional.append(normal_kl(before_aim,after_aim,teacher_std[2:],std[2:]))
            if vertical_conditioned:
                # Exact categorical expectation; no sampled-vertical KL proxy.
                vertical_probs=previous[:,5:8].softmax(-1)
                conditional=(torch.stack(conditional,-1)*vertical_probs[:,None,None,:]).sum(-1)
            else:
                conditional=conditional[0]
            per_target=categorical_kl(mode_before,mode_after)+(modes*conditional).sum(-1)
        else:
            fine = normal_kl(fine_before,fine_after,teacher_std[2:],std[2:])
            per_target = categorical_kl(mode_before, mode_after) + modes[:,:,0] * coarse + modes[:,:,1] * fine
    else:
        per_target = coarse
    terms = terms + (target_probs * per_target).sum(-1)
    assert bool(torch.isfinite(terms).all())
    return terms


def load_spec(path, meta, rows, model_sha):
    spec = read(path)
    assert spec['version'] in (VERSION, 'combat_sequence_retention_v2') and spec['feature_version'] == meta['feature_version']
    assert spec['model_sha256'] == model_sha == meta['model_sha256']
    assert spec['rollout_sha256'] == meta['rollout_sha256']
    assert spec['sequence_sha256'] == meta['sequence_sha256']
    plan = read(spec['training_plan'])
    assert sha(spec['training_plan']) == spec['training_plan_sha256']
    assert plan['model_sha256'] == model_sha and all(t['split'] == 'train' for t in plan['tasks'])
    registered = spec['version'] == 'combat_sequence_retention_v2'
    if registered:
        from combat_registered_retention import validate
        validate(spec, meta)
    pool_path = pathlib.Path(spec['registered_pool']) if registered else pathlib.Path(spec['training_plan']).parent.parent / 'pool/report.json'
    assert sha(pool_path) == spec['pool_sha256']
    pool = read(pool_path)
    assert pool['state'] == 'complete' and pool['source_unchanged'] and not any(j['error'] for j in pool['jobs'])
    plan_index = spec['plan_index'] if registered else 0
    jobs = [j for j in pool['jobs'] if j['plan_index'] == plan_index] if registered else pool['jobs']
    expected_jobs = {(plan_index,i,s,'learned') for i,t in enumerate(plan['tasks']) for s in t['seeds']}
    actual_jobs = {(j['plan_index'],j['task_index'],j['seed'],j['mode']) for j in jobs}
    assert actual_jobs == expected_jobs and len(actual_jobs) == len(jobs)
    retained = set(spec['retained_families'])
    assert retained and retained <= {t['episode']['id'] for t in plan['tasks']}
    seed_family = {seed:t['episode']['id'] for t in plan['tasks'] for seed in t['seeds']}
    lookup = {(row['seed'],row['index']):index for index,row in enumerate(rows)}
    assert len(lookup) == len(rows) and all(row['seed'] in seed_family for row in rows)
    expected = {key for key in lookup if seed_family[key[0]] in retained}
    actual = {(r['seed'],r['index']) for r in spec['members']}
    assert actual == expected and len(actual) == len(spec['members']) > 0
    for member in spec['members']:
        assert member['family'] == seed_family[member['seed']]
        assert math.isfinite(member['weight']) and member['weight'] > 0
    if registered:
        counts = collections.Counter(m['seed'] for m in spec['members'])
        for family in retained:
            seeds = {seed for seed, name in seed_family.items() if name == family}
            assert seeds <= set(counts), 'Every retained training seed must have histories'
            for member in spec['members']:
                if member['family'] == family:
                    expected_weight = 1 / len(retained) / len(seeds) / counts[member['seed']]
                    assert math.isclose(member['weight'], expected_weight, rel_tol=1e-12, abs_tol=1e-15)
    assert math.isclose(sum(m['weight'] for m in spec['members']), 1., abs_tol=1e-10)
    return spec, [lookup[(m['seed'],m['index'])] for m in spec['members']]

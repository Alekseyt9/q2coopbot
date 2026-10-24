"""CUDA-only fixed-budget full-policy continuation across all diverse registry recipes.

Ordinary trainers still reject shared heads. This experimental trainer admits
only sealed TRAIN captures with complete native/CUDA shared-logit proofs.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

from evaluate_registered_combat_models import read, sha
from ppo_combat import torch, nn, training_devices, advantages, validate_objective, guarded_actor_step
from ppo_recurrent import prepare_context, sequence, durable_json, durable_write
from combat_geometry_ppo import GeometryJoint, TRAINING_VERSION
from combat_joint_aim_policy import base_model, events, probabilities, validate
from combat_aim_candidates import VERSION, log_probability, entropy
from combat_precision_head import probabilities as base_probabilities
from combat_conditional_fire import logits as fire_logits
from combat_spatial_aim import export as export_actor_branches
from combat_weapon_state_residual import optimizer as actor_optimizer
from migrate_combat_navigation_cuda import branch
from fork_conditional_weapon_exploration_cuda import equal_tree
from train_combat_joint_aim_cuda import cuda_tree

_output = None
CURRICULUM_VERSION='combat_diverse_registry_ppo_v1'


def split_geometry_optimizer(opt, base_groups):
    state = copy.deepcopy(opt.state_dict())
    assert len(state['param_groups']) == base_groups + 1
    group = state['param_groups'][-1]; ids = group['params']
    assert len(ids) == 5, 'Four network tensors plus baseline beta'
    head_group = copy.deepcopy(group); head_group.pop('name', None)
    head_group['params'] = list(range(len(ids)))
    head = dict(state={i: state['state'][j] for i, j in enumerate(ids) if j in state['state']}, param_groups=[head_group])
    base = dict(state={i: v for i, v in state['state'].items() if i not in ids}, param_groups=state['param_groups'][:base_groups])
    return base, head


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'corpus', 'credit_audit', 'advantage_audit', 'protocol', 'out'):
        ap.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    ap.add_argument('--arm', choices=('B', 'G'), required=True)
    ap.add_argument('--round', type=int, choices=tuple(range(1,9)), required=True)
    ap.add_argument('--boundary-acceptance', type=Path)
    a = ap.parse_args(); assert not a.out.exists()
    training_devices(); torch.set_default_device('cuda'); torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve(); digest = sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest; return path

    source, corpus = a.source.resolve(), a.corpus.resolve()
    seal = read(pin(source / 'complete.json'))
    for name, key in (('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'), ('report.json', 'report_sha256')):
        pin(source / name, seal[key])
    pin(a.protocol)
    experiment = read(a.protocol)
    assert experiment['version'] == CURRICULUM_VERSION and experiment['rounds'] == 8
    assert experiment['arms']==['G'] and a.arm=='G' and experiment['native_training_cases']==384 and experiment['cases_per_update']==48
    assert experiment['actor_steps'] == 30 and experiment['critic_steps'] == 40
    assert experiment['actor_group_lrs'] == [3e-5, .0003, .0003, .0003]
    assert experiment['head_lr'] == .0003 and experiment['value_lr'] == .001
    assert experiment['target_full_kl'] == .005 and experiment['target_factor_kl'] == .0025
    boundary_sha = None
    if experiment.get('handoff_bootstrap')=='nonterminal_v1':
        assert a.boundary_acceptance is not None, 'New objective requires four live boundary proofs before fitting'
        bs = read(pin(a.boundary_acceptance/'complete.json'))
        boundary = read(pin(a.boundary_acceptance/'report.json',bs['report_sha256']))
        assert boundary['version']=='combat_nonterminal_boundary_acceptance_v1' and boundary['state']=='complete'
        assert boundary['native_cases']==boundary['cuda_cases']==4 and boundary['physical_starts_exact']
        assert boundary['handoff_bootstrap']=='nonterminal_v1' and boundary['reward_sha256']==experiment['reward_sha256']
        boundary_sha = bs['report_sha256']
    model = read(source / 'weights.json'); validate(model, allow_shared=True)
    core = base_model(model, allow_shared=True)
    closed = read(pin(corpus / 'complete.json'))
    collection = read(pin(corpus / 'protocol.json', closed['protocol_sha256']))
    assert closed['state'] == 'complete' and closed['native_cases'] == closed['cuda_cases'] == 48 and not closed['training_performed']
    assert collection['train_only'] and collection['ppo_training_objective'] and len(collection['assignments']) == 1
    assert not collection.get('diagnostic_intervention')
    assignment = collection['assignments'][0]
    assert assignment['split'] == 'train' and len(set(assignment['seeds'])) == 48
    plan = read(pin(assignment['plan'], assignment['plan_sha256']))
    assert plan['model_sha256'] == seal['weights_sha256']
    assert all(t['split'] == 'train' and t['modes'] == ['learned'] for t in plan['tasks'])
    data = corpus / 'processing-train/merged'; meta = read(pin(data / 'report.json'))
    assert meta['version'] == 'combat_ppo_rollout_v1' and meta['numerical_verification'] == 'cuda_verified_v1'
    assert not meta.get('diagnostic_only') and meta['action_distribution'] == VERSION and meta['model_sha256'] == seal['weights_sha256']
    for name, key in (('rollout.jsonl', 'rollout_sha256'), ('sequence.jsonl', 'sequence_sha256')): pin(data / name, meta[key])
    proof_count = 0; verified_frames = 0
    for path, digest in meta['source_sha256'].items():
        pin(path, digest)
        if Path(path).name == 'cuda-verification.json':
            proof = read(path)
            assert proof['frame_hash_candidate_assignments_verified'] and proof['original_latent_transform_verified']
            assert proof['shared_candidate_precision'] == 'float64'
            assert proof['shared_candidate_all_logits_max_error'] < .001
            assert proof['shared_candidate_all_probabilities_max_error'] < .0001
            proof_count += 1; verified_frames += proof['shared_candidate_frames']
    assert proof_count == 48 and verified_frames > 0, 'Require all per-case geometry proofs'
    native_seal = read(pin(a.credit_audit / 'complete.json'))
    native = read(pin(a.credit_audit / 'report.json', native_seal['report_sha256']))
    gae_seal = read(pin(a.advantage_audit / 'complete.json'))
    gae = read(pin(a.advantage_audit / 'report.json', gae_seal['report_sha256']))
    assert native['state'] == gae['state'] == 'complete'
    assert native['inputs'][str(corpus / 'complete.json')] == sha(corpus / 'complete.json')
    assert gae['inputs'][str((a.credit_audit / 'report.json').resolve())] == sha(a.credit_audit / 'report.json')
    assert native['totals']['rows'] == gae['rows'] == meta['rows']
    if experiment.get('handoff_bootstrap')=='nonterminal_v1':
        assert meta.get('handoff_bootstrap')=='nonterminal_v1'
        assert all(h['bootstrap_contract_verified'] and h['handoff_bootstrap']=='nonterminal_v1' for h in native['handoffs'])
    else:
        assert not meta.get('handoff_bootstrap') and all(h['confirmed_goal'] for h in native['handoffs'])
    assert not native['unassigned_native_effects']
    assert native['spacing_telescoping_max_error'] < 1e-8 and gae['gae_closed_form_max_error'] < 1e-8
    cp = torch.load(source / 'checkpoint.pt', map_location='cuda', weights_only=True)
    snapshot = copy.deepcopy(cp)
    # CPU deserialization is only for byte-preserving RNG restoration/serialization.
    raw_cp = torch.load(source / 'checkpoint.pt', map_location='cpu', weights_only=True)
    assert cp['weights_sha256'] == seal['weights_sha256'] and cp['action_distribution'] == VERSION
    assert meta['rollout_sha256'] not in cp['consumed_rollouts'], 'Rollout already consumed'
    assert cp['retention_weights'] == [0., 0.] and cp['sequence_retention'] is None
    assert cp['policy_update']['method'] == 'ppo' and cp['critic_credit']['mode'] == 'legacy'
    geometry = cp['geometry_aim_training']
    assert geometry['version'] == (TRAINING_VERSION if a.round==1 else CURRICULUM_VERSION) and geometry['updates_completed'] == a.round - 1
    assert geometry['head_lr'] == experiment['head_lr'] and geometry['actor_group_lrs'] == experiment['actor_group_lrs']
    if a.round == 1:
        assert seal['weights_sha256'] == experiment['initializer_weights_sha256']
        initial = read(source / 'report.json')
        assert experiment['movement_variance_checkpoint_version']=='combat_movement_variance_checkpoint_fork_v1'
        assert initial['version']==experiment['movement_variance_checkpoint_version'] and initial['variant']=='noise-only'
        assert initial['source_weights_sha256']==experiment['parent_initializer_weights_sha256']==geometry['initializer_weights_sha256']
        assert all(initial[k] for k in ('evaluation_weights_bytes_exact','neural_tensors_except_std_exact','raw_actor_value_exact',
            'only_changed_std_moments_reset','source_optimizer_clocks_preserved','critic_geometry_optimizer_rng_history_config_retention_exact','checkpoint_roundtrip_verified'))
        assert not initial['training_performed'] and initial['persisted_optimizer_steps']==0
        migration=cp['model_migrations'][-1]
        assert migration['version']==initial['version'] and migration['variant']=='noise-only' and not migration['training_performed']
        assert migration['source_weights_sha256']==initial['source_weights_sha256'] and migration['std_coordinate_moments_reset']==[True,True,False,False]
        assert migration['fresh_own_policy_data_required'] and migration['source_history_and_retention_preserved']
    else:
        assert geometry['arm'] == a.arm and geometry['protocol_sha256'] == sha(a.protocol)
        assert geometry.get('boundary_acceptance_sha256') == boundary_sha
        assert cp['actor_training']['scope'] == CURRICULUM_VERSION + '_' + a.arm
    validate_objective(cp['config'], meta)
    rows = [json.loads(line) for line in (data / 'rollout.jsonl').read_text().splitlines()]
    context = [json.loads(line) for line in (data / 'sequence.jsonl').read_text().splitlines()]
    assert len(rows) == meta['rows'] and len(context) == meta['sequence_rows']
    assert {r['seed'] for r in rows} == set(assignment['seeds'])
    prepared = prepare_context(context, rows, 'cuda')
    x = torch.tensor([r['features'] for r in rows], device='cuda', dtype=torch.float32)
    cat = lambda key: torch.tensor([r['sample'].get(key, 0) for r in rows], device='cuda', dtype=torch.long)
    target, mode, vertical, weapon = (cat(k) for k in ('target', 'aim_mode', 'vertical', 'weapon'))
    final = torch.tensor([r['sample']['latent'] for r in rows], device='cuda', dtype=torch.float32)
    attack = torch.tensor([float(r['sample']['attack']) for r in rows], device='cuda')
    source_z, choice, timing, active = events(rows, x, target, mode, vertical, weapon)
    native_lp = torch.tensor([r['sample']['log_probability'] for r in rows], device='cuda')
    actor, value, joint = branch(core, 'actor'), branch(core, 'value'), GeometryJoint(model)
    std = nn.Parameter(torch.tensor(model['log_std'], device='cuda'))
    assert equal_tree(cp['actor'], actor.state_dict()) and equal_tree(cp['value'], value.state_dict())
    assert equal_tree(cp['geometry_aim_head'], joint.geometry_state()) and torch.equal(std, cp['log_std'])
    hints_before = copy.deepcopy(joint.hints.state_dict()); legacy_before = copy.deepcopy(joint.head.state_dict())

    def full_score(raw):
        return probabilities(joint, raw, std, source_z, final, attack, vertical, weapon, x, target, mode, choice, timing,
            actor.sampled_motion_aim_branch, actor.motor_branch, actor.conditional_fire_branch)

    cache = []
    hook = joint.shared.register_forward_pre_hook(lambda module, args: cache.append(args[0].detach().clone()))
    with torch.no_grad():
        old_raw, actor_memory_error = sequence(actor, prepared, 32, True, 'actor')
        old_value, value_memory_error = sequence(value, prepared, 32, True, 'value')
        old_lp, _, old_logits, old_active = full_score(old_raw)
        assert torch.equal(active, old_active) and len(cache) == 1
        head_inputs = cache[0]; assert head_inputs.shape == (len(rows), 25, 31) and not head_inputs.requires_grad
        old_cat = log_probability(old_logits, choice, active)
        old_base = (old_lp - old_cat).detach(); old_logp25 = old_logits.log_softmax(-1).detach()
        replay_error = float((old_lp - native_lp).abs().max()); assert replay_error < .003
        value_error = float((old_value[:, 0] - torch.tensor([r['sample']['value'] for r in rows], device='cuda')).abs().max())
        if value_error >= 1e-4:
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                precise_value = branch(core, 'value').double()
                precise_context = prepare_context(context, rows, 'cuda', dtype=torch.float64)
                checked, precise_memory_error = sequence(precise_value, precise_context, 32, True, 'value')
                value_error = float((checked[:, 0] - torch.tensor([r['sample']['value'] for r in rows], device='cuda', dtype=torch.float64)).abs().max())
                assert precise_memory_error < 2e-5
        assert value_error < 1e-4 and max(actor_memory_error, value_memory_error) < 2e-5
    hook.remove()
    assert int(active.sum()) == verified_frames
    adv, returns = advantages(rows, cp['config']['gamma'], cp['config']['lambda'])
    ad = torch.tensor(adv, device='cuda'); ad = (ad - ad.mean()) / (ad.std(unbiased=False) + 1e-8)
    ret = torch.tensor(returns, device='cuda')
    rates = experiment['actor_group_lrs']
    assert [g['lr'] for g in cp['actor_optimizer']['param_groups']] == rates
    config = dict(version=CURRICULUM_VERSION, arm=a.arm, round=a.round, **{k: experiment[k] for k in
        ('actor_steps', 'critic_steps', 'head_lr', 'value_lr', 'target_full_kl', 'target_factor_kl', 'actor_group_lrs')},
        gamma=cp['config']['gamma'], lambda_=cp['config']['lambda'], clip=cp['config']['clip'], entropy=cp['config']['entropy'],
        max_grad_norm=cp['config']['max_grad_norm'], protocol_sha256=sha(a.protocol), behavior_sha256=seal['weights_sha256'])
    assert config['gamma'] == .99 and config['lambda_'] == .95 and config['clip'] == .2
    assert config['entropy'] == .001 and config['max_grad_norm'] == .5 and cp['config']['value_lr'] == .001
    opt = actor_optimizer(actor, std, *rates); opt.load_state_dict(copy.deepcopy(cp['actor_optimizer']))
    head_opt = joint.geometry_optimizer(config['head_lr']); head_opt.load_state_dict(copy.deepcopy(cp['geometry_aim_optimizer']))
    value_opt = torch.optim.Adam(value.parameters(), lr=config['value_lr']); value_opt.load_state_dict(copy.deepcopy(cp['value_optimizer']))
    for optimizer, key in ((opt, 'actor_optimizer'), (head_opt, 'geometry_aim_optimizer'), (value_opt, 'value_optimizer')):
        assert equal_tree(cuda_tree(optimizer.state_dict()), cp[key])
    base_groups = len(opt.param_groups)
    if a.arm == 'G':
        group = copy.copy(head_opt.param_groups[0]); group['name'] = 'geometry_aim_head'; opt.add_param_group(group)
        for p in joint.geometry_parameters():
            if p in head_opt.state: opt.state[p] = copy.deepcopy(head_opt.state[p])
    else:
        joint.shared.requires_grad_(False); joint.shared_beta.requires_grad_(False)
    a.out.mkdir(parents=True)
    global _output
    _output = a.out; durable_json(a.out / 'protocol.json', config)
    torch.set_rng_state(raw_cp['rng']); torch.cuda.set_rng_state_all(raw_cp['cuda_rng'])

    def current():
        raw, _ = sequence(actor, prepared, 32, False, 'actor')
        lp, ent = base_probabilities(raw, std, source_z, attack, vertical, weapon, x, target, mode,
            actor.sampled_motion_aim_branch, actor.motor_branch, actor.conditional_fire_branch)
        before = torch.distributions.Bernoulli(logits=fire_logits(actor.conditional_fire_branch, x, raw, target, mode, source_z, vertical, weapon))
        after = torch.distributions.Bernoulli(logits=fire_logits(actor.conditional_fire_branch, x, raw, target, mode, final, vertical, weapon))
        base_lp = lp + after.log_prob(attack) - before.log_prob(attack)
        base_ent = ent + after.entropy() - before.entropy()
        baseline = (torch.arange(25, device='cuda') == 0).to(x.dtype)
        logits = joint.shared(head_inputs).squeeze(-1).clamp(-20, 20) + joint.shared_beta * baseline
        return base_lp + log_probability(logits, choice, active), base_ent + entropy(logits, active), logits, base_lp

    def metrics():
        lp, ent, logits, base_lp = current()
        delta = lp - native_lp; ratio = delta.exp()
        loss = -torch.minimum(ratio * ad, ratio.clamp(1-config['clip'], 1+config['clip']) * ad).mean() - config['entropy'] * ent.mean()
        full_kl = ((ratio - 1) - delta).mean(); base_delta = base_lp - old_base
        base_kl = ((base_delta.exp() - 1) - base_delta).mean()
        category = (old_logp25.exp() * (old_logp25 - logits.log_softmax(-1))).sum(-1)
        category_kl = category[active].mean()
        assert all(bool(torch.isfinite(v)) for v in (loss, full_kl, base_kl, category_kl))
        return loss, full_kl, base_kl, category_kl

    def objective():
        loss, full, base, category = metrics()
        return loss, torch.stack((full, 2*base, 2*category)).max()

    def project():
        std.clamp_(-8, 1); joint.project()

    started = time.perf_counter(); trials = []; norms = []
    parameters = list(actor.parameters()) + [std] + (joint.geometry_parameters() if a.arm == 'G' else [])
    lrs = rates + ([config['head_lr']] if a.arm == 'G' else [])
    with torch.no_grad(): initial_loss = float(metrics()[0])
    durable_json(a.out / 'progress.json', dict(stage='gpu-training', arm=a.arm, round=a.round))
    for step in range(config['actor_steps']):
        loss, kl = objective()
        if float(kl.detach()) > config['target_full_kl']: break
        opt.zero_grad(set_to_none=True); loss.backward()
        norm = nn.utils.clip_grad_norm_(parameters, config['max_grad_norm']); assert bool(torch.isfinite(norm)); norms.append(float(norm))
        trial = guarded_actor_step(parameters, opt, objective, loss.detach(), lrs[0], config['target_full_kl'], project, group_lrs=lrs)
        trials.append(trial)
        if not trial['accepted']: break
    for step in range(config['critic_steps']):
        predicted, _ = sequence(value, prepared, 32, False, 'value')
        loss = (predicted[:, 0] - ret).square().mean(); assert bool(torch.isfinite(loss))
        value_opt.zero_grad(set_to_none=True); loss.backward()
        norm = nn.utils.clip_grad_norm_(value.parameters(), config['max_grad_norm']); assert bool(torch.isfinite(norm)); value_opt.step()
    torch.cuda.synchronize()
    assert equal_tree(cp, snapshot), 'Source checkpoint mutated by optimizer aliasing'
    for module, state in ((joint.hints, hints_before), (joint.head, legacy_before)):
        assert equal_tree(module.state_dict(), state) and all(p.grad is None for p in module.parameters())
    with torch.no_grad():
        final_metrics = [float(v) for v in metrics()]
        lp, _, logits, _ = current(); raw, _ = sequence(actor, prepared, 32, False, 'actor')
        complete_lp, _, complete_logits, complete_active = full_score(raw)
        cache_error = float((lp - complete_lp).abs().max()); assert cache_error < 1e-4
        assert torch.equal(logits, complete_logits) and torch.equal(active, complete_active)
        assert final_metrics[1] <= .005 + 1e-6 and max(final_metrics[2:]) <= .0025 + 1e-6
        value_after, _ = sequence(value, prepared, 32, False, 'value')
        value_mse_before = float((old_value[:, 0] - ret).square().mean())
        value_mse_after = float((value_after[:, 0] - ret).square().mean())
    base_state, head_state = split_geometry_optimizer(opt, base_groups) if a.arm == 'G' else (copy.deepcopy(opt.state_dict()), copy.deepcopy(head_opt.state_dict()))
    if a.arm == 'B':
        assert equal_tree(joint.geometry_state(), cp['geometry_aim_head']) and equal_tree(cuda_tree(head_state), cp['geometry_aim_optimizer'])
        assert all(p.grad is None for p in joint.geometry_parameters())
    base_a, cell_a = actor.export(); base_v, cell_v = value.export(); key = 'attention' if core.get('attention') else 'memory'
    updated_core = copy.deepcopy(core); updated_core.update(actor=base_a, value=base_v, log_std=std.detach().tolist())
    updated_core[key]['actor'] = cell_a; updated_core[key]['value'] = cell_v; export_actor_branches(updated_core, actor)
    updated = joint.export({**updated_core, 'action_distribution': VERSION, 'aim_candidate_head': copy.deepcopy(model['aim_candidate_head'])})
    validate(updated, allow_shared=True)
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        decoded_actor, decoded_head = branch(base_model(updated, allow_shared=True), 'actor'), GeometryJoint(updated)
        assert equal_tree(decoded_actor.state_dict(), actor.state_dict()) and equal_tree(decoded_head.geometry_state(), joint.geometry_state())
        with torch.no_grad():
            decoded_raw, _ = sequence(decoded_actor, prepared, 32, False, 'actor')
            decoded_lp = probabilities(decoded_head, decoded_raw, torch.tensor(updated['log_std'], device='cuda'), source_z, final,
                attack, vertical, weapon, x, target, mode, choice, timing, decoded_actor.sampled_motion_aim_branch,
                decoded_actor.motor_branch, decoded_actor.conditional_fire_branch)[0]
            export_error = float((decoded_lp - lp).abs().max()); assert export_error < 1e-4
    durable_json(a.out / 'weights.json', updated)
    base_steps = sum(t['accepted'] for t in trials); head_steps = base_steps if a.arm == 'G' else 0
    record = dict(version=CURRICULUM_VERSION, arm=a.arm, round=a.round, source_weights_sha256=seal['weights_sha256'],
        rollout_sha256=meta['rollout_sha256'], base_steps=base_steps, head_steps=head_steps, critic_steps=config['critic_steps'], config=config)
    fork = copy.deepcopy(raw_cp)
    fork.update(weights_sha256=sha(a.out / 'weights.json'), actor=actor.state_dict(), value=value.state_dict(), log_std=std.detach(),
        actor_optimizer=base_state, value_optimizer=value_opt.state_dict(), geometry_aim_head=joint.geometry_state(), geometry_aim_optimizer=head_state,
        consumed_rollouts=cp['consumed_rollouts'] + [meta['rollout_sha256']], updates_completed=cp['updates_completed']+1,
        total_actor_steps=cp['total_actor_steps']+base_steps,
        geometry_aim_training={**geometry, 'version':CURRICULUM_VERSION, 'updates_completed': a.round, 'arm': a.arm, 'protocol_sha256': sha(a.protocol), 'boundary_acceptance_sha256': boundary_sha},
        geometry_aim_history=cp.get('geometry_aim_history', []) + [record],
        actor_training={**cp['actor_training'], 'scope': CURRICULUM_VERSION + '_' + a.arm},
        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    durable_write(a.out / 'checkpoint.pt', lambda f: torch.save(fork, f))
    loaded = torch.load(a.out / 'checkpoint.pt', map_location='cuda', weights_only=True)
    for k in ('actor', 'value', 'log_std', 'actor_optimizer', 'value_optimizer', 'geometry_aim_head', 'geometry_aim_optimizer', 'aim_candidate_head', 'aim_candidate_optimizer'):
        assert equal_tree(cuda_tree(fork[k]), loaded[k]), 'Checkpoint roundtrip: ' + k
    assert equal_tree(cp['rng'], loaded['rng']) and equal_tree(cp['cuda_rng'], loaded['cuda_rng'])
    assert loaded['consumed_rollouts'][:-1] == cp['consumed_rollouts']
    frozen = a.out / 'trainer-sources'; frozen.mkdir(); helpers = {}
    for path in sorted({Path(__file__).resolve()} | {Path(m.__file__).resolve() for m in list(sys.modules.values())
            if getattr(m, '__file__', None) and Path(m.__file__).resolve().parent == Path(__file__).resolve().parent and Path(m.__file__).suffix == '.py'}):
        shutil.copy2(path, frozen / path.name); helpers[path.name] = sha(frozen / path.name)
    for path, digest in pins.items(): assert sha(Path(path)) == digest, 'Input changed: ' + path
    report = dict(state='complete', version=CURRICULUM_VERSION, device='cuda', arm=a.arm, round=a.round, rows=len(rows),
        eligible_rows=int(active.sum()), verified_native_cases=proof_count, training_performed=True, promotion=False, quality_verified=False,
        base_steps=base_steps, head_steps=head_steps, critic_steps=config['critic_steps'], trials=trials, gradient_norm_max=max(norms, default=0.),
        initial_loss=initial_loss, final_loss=final_metrics[0], full_approx_kl=final_metrics[1], base_approx_kl=final_metrics[2], category_exact_kl_active=final_metrics[3],
        baseline_probability_after=float(logits.softmax(-1)[active, 0].mean()),
        value_mse_before=value_mse_before, value_mse_after=value_mse_after, old_policy_lp_max_error=replay_error,
        current_cached_full_lp_max_error=cache_error, json_current_policy_replay_max_error=export_error,
        predictor_exact=True, legacy_head_exact=True, checkpoint_optimizer_roundtrip_exact=True, rng_preserved=True, head_frozen_exact=a.arm == 'B',
        source=str(source), behavior_sha256=seal['weights_sha256'], weights_sha256=sha(a.out / 'weights.json'),
        rollout_sha256=meta['rollout_sha256'], experiment_protocol_sha256=sha(a.protocol), inputs=pins, trainer_sources=helpers, seconds=time.perf_counter()-started)
    durable_json(a.out / 'report.json', report)
    durable_json(a.out / 'complete.json', dict(state='complete', version='combat_update_complete_v1', weights_sha256=report['weights_sha256'],
        checkpoint_sha256=sha(a.out / 'checkpoint.pt'), report_sha256=sha(a.out / 'report.json')))
    durable_json(a.out / 'progress.json', dict(stage='complete', arm=a.arm, round=a.round, base_steps=base_steps, head_steps=head_steps))
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs', 'trainer_sources', 'trials')}), flush=True)


if __name__ == '__main__':
    try: main()
    except Exception as error:
        if _output is not None: durable_json(_output / 'progress.json', dict(stage='failed', error=str(error), exception=type(error).__name__))
        raise

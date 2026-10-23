"""One resumable B/G/S update from sealed own-policy TRAIN, exclusively CUDA."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

from evaluate_registered_combat_models import read, sha
from ppo_combat import torch, nn, layers, training_devices, advantages, validate_objective, guarded_actor_step
from ppo_recurrent import prepare_context, sequence, durable_json, durable_write
from combat_joint_aim_policy import JointAim, base_model, events, probabilities, validate
from combat_aim_candidates import VERSION, log_probability, entropy
from combat_precision_head import probabilities as base_probabilities
from combat_conditional_fire import logits as fire_logits
from combat_spatial_aim import export as export_actor_branches
from combat_weapon_state_residual import optimizer as actor_optimizer
from migrate_combat_navigation_cuda import branch
from fork_conditional_weapon_exploration_cuda import equal_tree
from train_combat_joint_aim_cuda import cuda_tree
from train_combat_joint_aim_coupled_cuda import split_optimizer

TRAINER_VERSION = 'combat_joint_aim_near_budget_v1'
_output = None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'corpus', 'credit_audit', 'advantage_audit', 'protocol', 'out'):
        ap.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    ap.add_argument('--arm', choices=('B', 'G', 'S'), required=True)
    ap.add_argument('--round', type=int, choices=range(1, 5), required=True)
    a = ap.parse_args()
    assert not a.out.exists()
    training_devices()
    torch.set_default_device('cuda'); torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve(); digest = sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    source, corpus = a.source.resolve(), a.corpus.resolve()
    seal = read(pin(source / 'complete.json'))
    for name, key in (('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'), ('report.json', 'report_sha256')):
        pin(source / name, seal[key])
    pin(a.protocol)
    model = read(source / 'weights.json'); validate(model); core = base_model(model)
    closed = read(pin(corpus / 'complete.json'))
    collection = read(pin(corpus / 'protocol.json', closed['protocol_sha256']))
    assert closed['state'] == 'complete' and closed['native_cases'] == closed['cuda_cases'] == 32 and not closed['training_performed']
    assert collection['train_only'] and collection['ppo_training_objective'] and len(collection['assignments']) == 1
    assignment = collection['assignments'][0]
    assert assignment['split'] == 'train' and len(set(assignment['seeds'])) == 32
    plan = read(pin(assignment['plan'], assignment['plan_sha256']))
    assert plan['model_sha256'] == seal['weights_sha256'] and all(t['split'] == 'train' and t['modes'] == ['learned'] for t in plan['tasks'])
    data = corpus / 'processing-train/merged'
    meta = read(pin(data / 'report.json'))
    assert meta['version'] == 'combat_ppo_rollout_v1' and meta['numerical_verification'] == 'cuda_verified_v1'
    assert meta['action_distribution'] == VERSION and meta['model_sha256'] == seal['weights_sha256']
    for name, key in (('rollout.jsonl', 'rollout_sha256'), ('sequence.jsonl', 'sequence_sha256')):
        pin(data / name, meta[key])
    for path, digest in meta['source_sha256'].items():
        pin(path, digest)
    native_seal = read(pin(a.credit_audit / 'complete.json'))
    native = read(pin(a.credit_audit / 'report.json', native_seal['report_sha256']))
    gae_seal = read(pin(a.advantage_audit / 'complete.json'))
    gae = read(pin(a.advantage_audit / 'report.json', gae_seal['report_sha256']))
    assert native['state'] == gae['state'] == 'complete'
    assert native['inputs'][str(corpus / 'complete.json')] == sha(corpus / 'complete.json')
    assert gae['inputs'][str((a.credit_audit / 'report.json').resolve())] == sha(a.credit_audit / 'report.json')
    assert native['totals']['rows'] == gae['rows'] == meta['rows']
    assert not native['unassigned_native_effects'] and all(h['confirmed_goal'] for h in native['handoffs'])
    assert native['spacing_telescoping_max_error'] < 1e-8 and gae['gae_closed_form_max_error'] < 1e-8
    cp = torch.load(source / 'checkpoint.pt', map_location='cuda', weights_only=True)
    snapshot = copy.deepcopy(cp)
    raw_cp = torch.load(source / 'checkpoint.pt', map_location='cpu', weights_only=True)
    assert cp['weights_sha256'] == seal['weights_sha256'] and cp['action_distribution'] == VERSION
    assert meta['rollout_sha256'] not in cp['consumed_rollouts'], 'Rollout already consumed'
    assert cp['retention_weights'] == [0., 0.] and cp['sequence_retention'] is None and cp['policy_update']['method'] == 'ppo'
    assert cp['critic_credit']['mode'] == 'legacy'
    scope = TRAINER_VERSION + '_' + a.arm
    if a.round == 1:
        assert not cp.get('joint_aim_updates_completed', 0) and cp['actor_training']['scope'] == 'all'
        initial_report = read(source / 'report.json')
        assert initial_report['baseline_probability'] == .99, 'Only the accepted near-identity initializer may start this experiment'
    else:
        assert cp['actor_training']['scope'] == scope and cp['joint_aim_budget']['arm'] == a.arm
        assert cp['joint_aim_budget']['round'] == a.round - 1 and cp['joint_aim_budget']['protocol_sha256'] == sha(a.protocol)
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
    actor, value, joint = branch(core, 'actor'), branch(core, 'value'), JointAim(model)
    std = nn.Parameter(torch.tensor(model['log_std'], device='cuda'))
    for key, module in (('actor', actor), ('value', value), ('aim_candidate_head', joint.head)):
        assert equal_tree(cp[key], module.state_dict()), key
    assert torch.equal(std, cp['log_std'])
    joint.hints.requires_grad_(False); hints_before = copy.deepcopy(joint.hints.state_dict())

    def full_score(raw):
        return probabilities(joint, raw, std, source_z, final, attack, vertical, weapon, x, target, mode, choice, timing,
            actor.sampled_motion_aim_branch, actor.motor_branch, actor.conditional_fire_branch)

    cache = []
    hook = joint.head.register_forward_pre_hook(lambda module, args: cache.append(args[0].detach().clone()))
    with torch.no_grad():
        old_raw, actor_memory_error = sequence(actor, prepared, 32, True, 'actor')
        old_value, value_memory_error = sequence(value, prepared, 32, True, 'value')
        old_lp, _, old_logits, old_active = full_score(old_raw)
        assert torch.equal(active, old_active) and len(cache) == 1
        head_inputs = cache[0]
        old_cat = log_probability(old_logits, choice, active)
        old_base = (old_lp - old_cat).detach(); old_logp25 = old_logits.log_softmax(-1).detach()
        replay_error = float((old_lp - native_lp).abs().max()); assert replay_error < .003
        value_error = float((old_value[:, 0] - torch.tensor([r['sample']['value'] for r in rows], device='cuda')).abs().max())
        assert value_error < 1e-4 and max(actor_memory_error, value_memory_error) < 2e-5
    hook.remove()
    adv, returns = advantages(rows, cp['config']['gamma'], cp['config']['lambda'])
    ad = torch.tensor(adv, device='cuda'); ad = (ad - ad.mean()) / (ad.std(unbiased=False) + 1e-8)
    ret = torch.tensor(returns, device='cuda')
    rates = [g['lr'] for g in cp['actor_optimizer']['param_groups']]
    assert rates == [3e-5, .0003, .0003, .0003]
    config = dict(version=TRAINER_VERSION, arm=a.arm, round=a.round, max_rounds=4, value_steps=40,
        phases=([('base', 15)] if a.arm == 'B' else [('global', 30)] if a.arm == 'G' else [('head', 15), ('base', 15)]),
        target_full_kl=.005, target_factor_kl=.0025, factor_category='exact categorical KL averaged over eligible TRAIN events',
        factor_base='sampled original actor factors, with final-aim conditional fire',
        base_group_lrs=rates, candidate_lr=.0003, value_lr=cp['config']['value_lr'],
        gamma=cp['config']['gamma'], lambda_=cp['config']['lambda'], clip=cp['config']['clip'], entropy=cp['config']['entropy'],
        max_grad_norm=cp['config']['max_grad_norm'], protocol_sha256=sha(a.protocol), behavior_sha256=seal['weights_sha256'],
        rollout_sha256=meta['rollout_sha256'], source=str(source), corpus=str(corpus), predictor='frozen', critic_credit='legacy')
    assert config['gamma'] == .99 and config['lambda_'] == .95 and config['clip'] == .2
    assert config['entropy'] == .001 and config['max_grad_norm'] == .5 and cp['config']['target_kl'] == .005
    opt = actor_optimizer(actor, std, *rates)
    opt.load_state_dict(copy.deepcopy(cp['actor_optimizer']))
    head_opt = torch.optim.Adam(joint.head.parameters(), lr=config['candidate_lr'])
    head_opt.load_state_dict(copy.deepcopy(cp['aim_candidate_optimizer']))
    value_opt = torch.optim.Adam(value.parameters(), lr=config['value_lr'])
    value_opt.load_state_dict(copy.deepcopy(cp['value_optimizer']))
    for optimizer, key in ((opt, 'actor_optimizer'), (head_opt, 'aim_candidate_optimizer'), (value_opt, 'value_optimizer')):
        assert equal_tree(cuda_tree(optimizer.state_dict()), cp[key])
    base_groups = len(opt.param_groups)
    if a.arm == 'G':
        group = copy.copy(head_opt.param_groups[0]); group['name'] = 'aim_candidate_head'; opt.add_param_group(group)
        for p in joint.head.parameters():
            if p in head_opt.state: opt.state[p] = copy.deepcopy(head_opt.state[p])
    a.out.mkdir(parents=True)
    global _output
    _output = a.out
    durable_json(a.out / 'protocol.json', config)
    torch.set_rng_state(raw_cp['rng']); torch.cuda.set_rng_state_all(raw_cp['cuda_rng'])

    def current():
        raw, _ = sequence(actor, prepared, 32, False, 'actor')
        lp, ent = base_probabilities(raw, std, source_z, attack, vertical, weapon, x, target, mode,
            actor.sampled_motion_aim_branch, actor.motor_branch, actor.conditional_fire_branch)
        before = torch.distributions.Bernoulli(logits=fire_logits(actor.conditional_fire_branch, x, raw, target, mode, source_z, vertical, weapon))
        after = torch.distributions.Bernoulli(logits=fire_logits(actor.conditional_fire_branch, x, raw, target, mode, final, vertical, weapon))
        base_lp = lp + after.log_prob(attack) - before.log_prob(attack)
        base_ent = ent + after.entropy() - before.entropy()
        logits = joint.head(head_inputs)
        return base_lp + log_probability(logits, choice, active), base_ent + entropy(logits, active), logits, base_lp

    def metrics():
        lp, ent, logits, base_lp = current()
        delta = lp - native_lp; ratio = delta.exp()
        loss = -torch.minimum(ratio * ad, ratio.clamp(1-config['clip'], 1+config['clip']) * ad).mean() - config['entropy'] * ent.mean()
        full_kl = ((ratio - 1) - delta).mean()
        base_delta = base_lp - old_base
        base_kl = ((base_delta.exp() - 1) - base_delta).mean()
        category = (old_logp25.exp() * (old_logp25 - logits.log_softmax(-1))).sum(-1)
        category_kl = category[active].mean() if bool(active.any()) else category.sum() * 0
        assert all(bool(torch.isfinite(v)) for v in (loss, full_kl, base_kl, category_kl))
        return loss, full_kl, base_kl, category_kl

    def objective():
        loss, full, base, category = metrics()
        guard = full if a.arm == 'G' else torch.stack((full, 2*base, 2*category)).max()
        return loss, guard

    def project():
        std.clamp_(-8, 1)

    history = []; started = time.perf_counter()
    with torch.no_grad(): initial_loss = float(metrics()[0])
    for phase, steps in config['phases']:
        durable_json(a.out / 'progress.json', dict(stage='gpu-training', arm=a.arm, round=a.round, phase=phase))
        actor.requires_grad_(phase != 'head'); std.requires_grad_(phase != 'head'); joint.head.requires_grad_(phase != 'base')
        actor.zero_grad(set_to_none=True); joint.head.zero_grad(set_to_none=True); std.grad = None
        inactive = joint.head if phase == 'base' else actor if phase == 'head' else None
        inactive_state = copy.deepcopy(inactive.state_dict()) if inactive else None
        inactive_opt = head_opt if phase == 'base' else opt if phase == 'head' else None
        inactive_adam = copy.deepcopy(inactive_opt.state_dict()) if inactive_opt else None
        std_before = std.detach().clone()
        parameters = (list(joint.head.parameters()) if phase == 'head' else list(actor.parameters()) + [std] + (list(joint.head.parameters()) if phase == 'global' else []))
        optimizer = head_opt if phase == 'head' else opt
        lrs = [config['candidate_lr']] if phase == 'head' else rates + ([config['candidate_lr']] if phase == 'global' else [])
        trials = []; norms = []
        for step in range(steps):
            loss, kl = objective()
            if float(kl.detach()) > config['target_full_kl']: break
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = nn.utils.clip_grad_norm_(parameters, config['max_grad_norm']); assert bool(torch.isfinite(norm)); norms.append(float(norm))
            trial = guarded_actor_step(parameters, optimizer, objective, loss.detach(), lrs[0], config['target_full_kl'], project, group_lrs=lrs)
            trials.append(trial)
            if not trial['accepted']: break
        if inactive:
            assert equal_tree(inactive.state_dict(), inactive_state) and all(p.grad is None for p in inactive.parameters())
            assert equal_tree(cuda_tree(inactive_opt.state_dict()), cuda_tree(inactive_adam)), 'Inactive Adam changed during another factor update'
        if phase == 'head': assert torch.equal(std, std_before)
        with torch.no_grad(): m = [float(v) for v in metrics()]
        history.append(dict(phase=phase, accepted_steps=sum(t['accepted'] for t in trials), trials=trials,
            gradient_norm_max=max(norms, default=0.), final_loss=m[0], full_kl=m[1], base_kl=m[2], category_kl_active=m[3], inactive_exact=True))
    for step in range(40):
        predicted, _ = sequence(value, prepared, 32, False, 'value')
        loss = (predicted[:, 0] - ret).square().mean(); assert bool(torch.isfinite(loss))
        value_opt.zero_grad(set_to_none=True); loss.backward()
        norm = nn.utils.clip_grad_norm_(value.parameters(), config['max_grad_norm']); assert bool(torch.isfinite(norm)); value_opt.step()
    torch.cuda.synchronize()
    assert equal_tree(cp, snapshot), 'Source checkpoint mutated by optimizer aliasing'
    assert equal_tree(hints_before, joint.hints.state_dict()) and all(p.grad is None for p in joint.hints.parameters())
    with torch.no_grad():
        final_metrics = [float(v) for v in metrics()]
        lp, ent, logits, _ = current(); raw, _ = sequence(actor, prepared, 32, False, 'actor')
        complete_lp, _, complete_logits, complete_active = full_score(raw)
        cache_error = float((lp - complete_lp).abs().max()); assert cache_error < 1e-4
        assert torch.equal(logits, complete_logits) and torch.equal(active, complete_active)
        assert final_metrics[1] <= .005 + 1e-6
        if a.arm != 'G': assert max(final_metrics[2:]) <= .0025 + 1e-6
        value_after, _ = sequence(value, prepared, 32, False, 'value')
        value_mse_before = float((old_value[:, 0] - ret).square().mean())
        value_mse_after = float((value_after[:, 0] - ret).square().mean())
    base_state, head_state = split_optimizer(opt, base_groups) if a.arm == 'G' else (copy.deepcopy(opt.state_dict()), copy.deepcopy(head_opt.state_dict()))
    if a.arm == 'B':
        assert equal_tree(joint.head.state_dict(), cp['aim_candidate_head']) and equal_tree(cuda_tree(head_state), cp['aim_candidate_optimizer'])
    base_a, cell_a = actor.export(); base_v, cell_v = value.export(); key = 'attention' if core.get('attention') else 'memory'
    updated_core = copy.deepcopy(core); updated_core.update(actor=base_a, value=base_v, log_std=std.detach().tolist())
    updated_core[key]['actor'] = cell_a; updated_core[key]['value'] = cell_v; export_actor_branches(updated_core, actor)
    updated = {**updated_core, 'action_distribution': VERSION, 'aim_candidate_head': copy.deepcopy(model['aim_candidate_head'])}
    updated['aim_candidate_head']['layers'] = layers(joint.head.layers); validate(updated)
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        decoded_actor, decoded_head = branch(base_model(updated), 'actor'), JointAim(updated)
        assert equal_tree(decoded_actor.state_dict(), actor.state_dict()) and equal_tree(decoded_head.head.state_dict(), joint.head.state_dict())
        with torch.no_grad():
            decoded_raw, _ = sequence(decoded_actor, prepared, 32, False, 'actor')
            decoded_lp = probabilities(decoded_head, decoded_raw, torch.tensor(updated['log_std'], device='cuda'), source_z, final,
                attack, vertical, weapon, x, target, mode, choice, timing, decoded_actor.sampled_motion_aim_branch,
                decoded_actor.motor_branch, decoded_actor.conditional_fire_branch)[0]
            export_error = float((decoded_lp - lp).abs().max()); assert export_error < 1e-4
    durable_json(a.out / 'weights.json', updated)
    base_steps = sum(h['accepted_steps'] for h in history if h['phase'] != 'head')
    head_steps = sum(h['accepted_steps'] for h in history if h['phase'] != 'base')
    record = dict(version=TRAINER_VERSION, arm=a.arm, round=a.round, source_weights_sha256=seal['weights_sha256'],
        rollout_sha256=meta['rollout_sha256'], base_steps=base_steps, head_steps=head_steps, critic_steps=40, config=config)
    fork = copy.deepcopy(raw_cp)
    fork.update(weights_sha256=sha(a.out / 'weights.json'), actor=actor.state_dict(), value=value.state_dict(), log_std=std.detach(),
        actor_optimizer=base_state, value_optimizer=value_opt.state_dict(), aim_candidate_head=joint.head.state_dict(), aim_candidate_optimizer=head_state,
        consumed_rollouts=cp['consumed_rollouts'] + [meta['rollout_sha256']], updates_completed=cp['updates_completed']+1,
        total_actor_steps=cp['total_actor_steps']+base_steps, joint_aim_updates_completed=cp.get('joint_aim_updates_completed', 0)+1,
        joint_aim_total_steps=cp.get('joint_aim_total_steps', 0)+head_steps, joint_aim_history=cp.get('joint_aim_history', [])+[record],
        actor_training={**cp['actor_training'], 'scope': scope}, joint_aim_budget=dict(version=TRAINER_VERSION, arm=a.arm, round=a.round, protocol_sha256=sha(a.protocol)),
        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    durable_write(a.out / 'checkpoint.pt', lambda f: torch.save(fork, f))
    loaded = torch.load(a.out / 'checkpoint.pt', map_location='cuda', weights_only=True)
    for k in ('actor', 'value', 'log_std', 'actor_optimizer', 'value_optimizer', 'aim_candidate_head', 'aim_candidate_optimizer'):
        assert equal_tree(cuda_tree(fork[k]), loaded[k]), 'Checkpoint roundtrip: ' + k
    assert equal_tree(cp['rng'], loaded['rng']) and equal_tree(cp['cuda_rng'], loaded['cuda_rng'])
    assert loaded['consumed_rollouts'][:-1] == cp['consumed_rollouts'] and loaded['actor_training']['scope'] == scope
    frozen = a.out / 'trainer-sources'; frozen.mkdir(); helpers = {}
    for path in sorted({Path(__file__).resolve()} | {Path(m.__file__).resolve() for m in list(sys.modules.values())
            if getattr(m, '__file__', None) and Path(m.__file__).resolve().parent == Path(__file__).resolve().parent and Path(m.__file__).suffix == '.py'}):
        shutil.copy2(path, frozen / path.name); helpers[path.name] = sha(frozen / path.name)
    for path, digest in pins.items(): assert sha(Path(path)) == digest, 'Input changed: ' + path
    report = dict(state='complete', version=TRAINER_VERSION, device='cuda', arm=a.arm, round=a.round, rows=len(rows),
        eligible_rows=int(active.sum()), nonzero_correction_rows=int(((choice != 0) & active).sum()),
        training_performed=True, promotion=False, quality_verified=False, native_runtime_verified=False,
        phases=history, base_steps=base_steps, head_steps=head_steps, critic_steps=40, initial_loss=initial_loss, final_loss=final_metrics[0],
        full_approx_kl=final_metrics[1], base_approx_kl=final_metrics[2], category_exact_kl_active=final_metrics[3],
        baseline_probability_after=float(logits.softmax(-1)[active, 0].mean()) if bool(active.any()) else None,
        value_mse_before=value_mse_before, value_mse_after=value_mse_after, old_policy_lp_max_error=replay_error,
        current_cached_full_lp_max_error=cache_error, json_current_policy_replay_max_error=export_error,
        predictor_exact=True, initial_adam_exact=True, source_checkpoint_memory_exact=True, inactive_phase_parameters_and_adam_exact=True,
        checkpoint_optimizer_roundtrip_exact=True, rng_preserved=True, head_frozen_exact=a.arm == 'B',
        source=str(source), behavior_sha256=seal['weights_sha256'], weights_sha256=sha(a.out / 'weights.json'),
        rollout_sha256=meta['rollout_sha256'], protocol_sha256=sha(a.out / 'protocol.json'), experiment_protocol_sha256=sha(a.protocol),
        inputs=pins, trainer_sources=helpers, seconds=time.perf_counter()-started)
    durable_json(a.out / 'report.json', report)
    durable_json(a.out / 'complete.json', dict(state='complete', version='combat_update_complete_v1', weights_sha256=report['weights_sha256'],
        checkpoint_sha256=sha(a.out / 'checkpoint.pt'), report_sha256=sha(a.out / 'report.json')))
    durable_json(a.out / 'progress.json', dict(stage='complete', arm=a.arm, round=a.round, base_steps=base_steps, head_steps=head_steps))
    print(json.dumps({k:v for k,v in report.items() if k not in ('phases', 'inputs', 'trainer_sources')}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if _output is not None:
            durable_json(_output / 'progress.json', dict(stage='failed', error=str(error), exception=type(error).__name__))
        raise

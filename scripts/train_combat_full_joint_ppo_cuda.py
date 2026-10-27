"""Full CUDA PPO on pinned, fresh registered TRAIN captures, with exact resume."""
import argparse
import json
import copy
from pathlib import Path
from ppo_combat import torch, nn, training_devices, advantages, validate_objective, guarded_actor_step
from ppo_recurrent import prepare_context, sequence
from combat_full_joint_ppo import FullJointScope, SCOPE, WIDE_SCOPE
from combat_joint_aim_policy import events, probabilities
from evaluate_registered_combat_models import read, save, sha


def admit(request, model_hash, consumed, consumed_seeds, pin):
    from combat_train_subset import REQUEST_VERSION,validate as validate_subset
    assert request['version'] in ('combat_full_joint_train_request_v1',REQUEST_VERSION) and request['split'] == 'train'
    if request['version']=='combat_full_joint_train_request_v1':
        assert 'selection' not in request and 'selection_sha256' not in request,'Subset requires explicit v2 request'
    plan = read(pin(request['plan'], request['plan_sha256']))
    pool = read(pin(request['pool_report'], request['pool_report_sha256']))
    assert plan['model_sha256'] == model_hash
    assert pool['state'] == 'complete' and pool['source_unchanged'] is True
    plan_indices={i for i,p in enumerate(pool['plans'])
        if p['sha256']==request['plan_sha256'] and p['model_sha256']==model_hash}
    assert len(plan_indices)==1, 'TRAIN request must bind exactly one pool plan'
    expected = {}
    for task in plan['tasks']:
        assert task['split'] == 'train' and task['modes'] == ['learned']
        assert task['episode']['recipe']['require_native_objective'] is True
        for instance in task['instances']:
            seed = instance['engine_seed']
            assert instance['split'] == 'train' and seed not in expected
            expected[seed] = instance
    if request['version']==REQUEST_VERSION:
        selection=read(pin(request['selection'],request['selection_sha256']))
        selected=validate_subset(plan,selection,request['plan_sha256'])
        expected={seed:instance for seed,instance in expected.items() if seed in selected}
        assert set(expected)==selected
    rows, history, tokens, seeds, metadata = [], [], [], set(), []
    assert request['cases']
    for case in request['cases']:
        seed = case['seed']; directory = Path(case['data']).resolve(); capture = Path(case['capture_root']).resolve()
        assert seed in expected and seed not in seeds and seed not in consumed_seeds
        jobs = [j for j in pool['jobs'] if j['seed'] == seed and j['mode'] == 'learned' and j['plan_index'] in plan_indices]
        assert len(jobs) == 1 and not jobs[0]['error'] and capture.is_relative_to(Path(jobs[0]['root']).resolve())
        assert read(pin(capture/'fixture-0.json')) == expected[seed], 'Capture is not the frozen TRAIN instance'
        assert read(pin(capture/'manifest.json'))['native_objective_required'] is True
        capture_report = read(pin(capture/'report.json'))
        assert capture_report['capture_complete'] and capture_report['provenance_valid']
        assert len(capture_report['results']) == 1
        result = capture_report['results'][0]
        assert result['seed'] == seed and result['native_objective_required'] is True and result['capture_valid']
        assert result['generated_fixture'] == expected[seed]
        meta = read(pin(directory/'report.json'))
        proof = read(pin(directory/'cuda-verification.json', meta['cuda_verification_sha256']))
        assert meta['version'] == 'combat_ppo_rollout_v1' and meta['numerical_verification'] == 'cuda_verified_v1'
        assert meta['model_sha256'] == proof['model_sha256'] == model_hash
        assert not meta.get('diagnostic_only')
        assert not any(k.startswith('diagnostic_') for k in meta), 'Diagnostic rollout cannot train'
        assert proof['state'] == 'passed' and proof['device'] == 'cuda' and proof['execution_contract_verified']
        assert meta['observation_timing_contract'] == 'frozen_world_v1'
        # This explicit optimizer scope admits supported branches. It does not
        # rewrite or bypass the ordinary trainer's readiness flags.
        for path, digest in meta['source_sha256'].items():
            pin(path, digest)
        source_paths = {str(Path(p).resolve()).lower() for p in meta['source_sha256']}
        assert all(str((capture/name).resolve()).lower() in source_paths for name in ('manifest.json','report.json'))
        token = meta['native_rollout_sha256']
        assert token not in consumed and token not in tokens, 'Native rollout already consumed'
        load = lambda p: [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines()]
        rr = load(pin(directory/'rollout.jsonl', meta['rollout_sha256']))
        hh = load(pin(directory/'sequence.jsonl', meta['sequence_sha256']))
        assert len(rr) == meta['rows'] > 0 and len(hh) == meta['sequence_rows']
        assert {r['seed'] for r in rr} == {seed} and {r['seed'] for r in hh} == {seed}
        rows.extend(rr); history.extend(hh); tokens.append(token); seeds.add(seed); metadata.append(meta)
    assert len({m['reward_config_sha256'] for m in metadata}) == 1
    assert len({m['combat_execution_contract'] for m in metadata}) == 1
    assert seeds == set(expected), 'Every planned TRAIN case must be present, without outcome filtering'
    return rows, history, tokens, seeds, metadata


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('model', 'request', 'config', 'out'):
        ap.add_argument('--'+key, type=Path, required=True)
    ap.add_argument('--resume', type=Path)
    ap.add_argument('--fork-conditional-fire',action='store_true',help='Explicit scope/LR fork preserving all restored Adam states')
    ap.add_argument('--fork-episode-balance',action='store_true',help='Explicit equal battle mass fork preserving full policy Adam/config')
    a = ap.parse_args(); assert not a.out.exists()
    training_devices(); torch.set_default_device('cuda'); torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    pins = {}
    def pin(path, expected=None):
        path = Path(path).resolve(); digest = sha(path)
        assert expected is None or digest == expected, 'Input changed: '+str(path)
        pins[str(path)] = digest
        return path
    model_hash = sha(pin(a.model)); model = read(a.model); config = read(pin(a.config))
    assert config['version'] == (WIDE_SCOPE if model.get('wide_aim') is not None else SCOPE) and 1 <= config['actor_steps'] <= 100 and 1 <= config['value_steps'] <= 100
    assert 0 < config['gamma'] <= 1 and 0 <= config['lambda'] <= 1 and 0 < config['clip'] <= .3
    assert 0 < config['target_kl'] <= .05 and config['entropy'] >= 0 and config['max_grad_norm'] > 0
    torch.manual_seed(config['seed'])
    checkpoint=torch.load(pin(a.resume),map_location='cuda',weights_only=True) if a.resume else None
    assert checkpoint is None or not checkpoint.get('technical_only'),'Synthetic checkpoint cannot train on games'
    assert not a.fork_conditional_fire or checkpoint is not None
    assert not a.fork_episode_balance or checkpoint is not None
    assert not (a.fork_episode_balance and a.fork_conditional_fire)
    scope = FullJointScope(model, checkpoint['config'] if a.fork_conditional_fire or a.fork_episode_balance else config)
    actor_scope=config.get('actor_scope','all')
    assert actor_scope in ('all','wide_only','conditional_fire_only')
    assert config.get('fire_phase_balance') is None or actor_scope=='conditional_fire_only'
    assert not a.fork_conditional_fire or actor_scope=='conditional_fire_only'
    if config.get('episode_balance') is not None:
        from combat_episode_balance import VERSION as EPISODE_BALANCE
        assert config['episode_balance']==EPISODE_BALANCE and actor_scope=='all' and scope.scope==SCOPE
        assert config.get('fire_phase_balance') is None
    if actor_scope=='wide_only':
        assert scope.joint.wide is not None and scope.joint.wide.enabled,'Wide-only fitting requires enabled option'
        frozen_actor={k:v.detach().clone() for k,v in scope.actor.state_dict().items()}
        frozen_joint={k:v.detach().clone() for k,v in scope.joint.state_dict().items() if not k.startswith('wide.')}
        frozen_std=scope.std.detach().clone()
    consumed, old_seeds, updates = [], [], 0
    if a.resume:
        cp = checkpoint
        consumed = scope.restore(cp, model_hash)
        old_seeds = cp['consumed_seeds']; updates = cp['updates_completed']
    fire_freeze=None
    if a.fork_episode_balance:
        from combat_episode_balance import fork_config as fork_episode_config
        fork_episode_config(scope,config)
    if actor_scope=='conditional_fire_only':
        from combat_conditional_fire_scope import FireFreeze,fork_config
        assert checkpoint is not None,'Fire-only training requires an explicit restored checkpoint'
        if a.fork_conditional_fire:fork_config(scope,config)
        fire_freeze=FireFreeze(scope)
    if actor_scope=='wide_only':
        gate_parameters=list(scope.joint.wide.parameters());gate_ids={id(p) for p in gate_parameters}
        frozen_moments={p:copy.deepcopy(v) for p,v in scope.actor_optimizer.state.items() if id(p) not in gate_ids}
    request = read(pin(a.request))
    rows, history, tokens, seeds, metadata = admit(request, model_hash, consumed, old_seeds, pin)
    for meta in metadata: validate_objective(config, meta)
    prepared = prepare_context(history, rows, 'cuda')
    x = torch.tensor([r['features'] for r in rows], device='cuda', dtype=torch.float32)
    phase_report=None
    episode_report=None
    loss_weights=torch.ones(len(rows),device='cuda',dtype=x.dtype)
    if config.get('episode_balance') is not None:
        from combat_episode_balance import weights as episode_weights
        loss_weights,episode_report=episode_weights(rows)
    if config.get('fire_phase_balance') is not None:
        from combat_fire_phase_balance import weights as fire_phase_weights
        loss_weights,phase_report=fire_phase_weights(x,config['fire_phase_balance'])
    cat = lambda key: torch.tensor([r['sample'].get(key, 0) for r in rows], device='cuda', dtype=torch.long)
    target, mode, vertical, weapon = (cat(k) for k in ('target', 'aim_mode', 'vertical', 'weapon'))
    final = torch.tensor([r['sample']['latent'] for r in rows], device='cuda', dtype=torch.float32)
    attack = torch.tensor([float(r['sample']['attack']) for r in rows], device='cuda')
    source, choice, timing, _ = events(rows, x, target, mode, vertical, weapon)
    from combat_wide_aim import events as wide_events
    wide_data=wide_events(model,rows,x,target,mode,vertical,weapon)
    score_inputs = None
    if scope.joint.geometry_head:
        score_inputs = (
            torch.tensor([r['features'] for r in rows], device='cuda', dtype=torch.float64),
            torch.tensor([(r['sample'].get('aim_candidate') or {}).get('source_latent',r['sample']['latent']) for r in rows], device='cuda', dtype=torch.float64),
            torch.tensor([(r['sample'].get('aim_candidate') or {}).get('timing',[0.,0.]) for r in rows], device='cuda', dtype=torch.float64))
    old = torch.tensor([r['sample']['log_probability'] for r in rows], device='cuda')
    def likelihood(raw):
        lp,ent=probabilities(scope.joint, raw, scope.std.detach() if fire_freeze else scope.std, source, final, attack, vertical,
            weapon, x, target, mode, choice, timing, scope.actor.sampled_motion_aim_branch,
            scope.actor.motor_branch, scope.actor.conditional_fire_branch, score_inputs=score_inputs,wide_data=wide_data)[:2]
        if actor_scope=='wide_only':
            from combat_wide_aim import terms
            gate_lp,gate_ent=terms(scope.joint.wide,wide_data,target,mode,vertical,weapon)
            # Same full forward likelihood; only the option receives gradients.
            # None gradients preserve old Adam moments AND parameters (zero
            # gradients would still move parameters with old momentum).
            lp=lp.detach()+(gate_lp-gate_lp.detach())
            ent=ent.detach()+(gate_ent-gate_ent.detach())
        return lp,ent
    with torch.no_grad():
        raw, actor_memory = sequence(scope.actor, prepared, 32, True, 'actor')
        value, value_memory = sequence(scope.value, prepared, 32, True, 'value')
        lp, _ = likelihood(raw)
        lp_error = float((lp-old).abs().max())
        value_error = float((value[:, 0]-torch.tensor([r['sample']['value'] for r in rows], device='cuda')).abs().max())
        assert lp_error < .003 and value_error < 1e-4 and max(actor_memory, value_memory) < 2e-5
    adv, returns = advantages(rows, config['gamma'], config['lambda'])
    ad = torch.tensor(adv, device='cuda')
    if episode_report is not None:
        from combat_episode_balance import normalize_advantages
        ad=normalize_advantages(ad,loss_weights)
    else:ad = (ad-ad.mean())/(ad.std(unbiased=False)+1e-8)
    ret = torch.tensor(returns, device='cuda')
    def objective():
        raw, _ = sequence(scope.actor, prepared)
        lp, entropy = likelihood(raw); delta = lp-old; ratio = delta.exp()
        surrogate=-torch.minimum(ratio*ad, ratio.clamp(1-config['clip'],1+config['clip'])*ad)
        divergence=(ratio-1)-delta
        # Bound both original-distribution and balanced-state KL so the rare
        # phase cannot move freely behind a small unweighted average.
        return ((surrogate-config['entropy']*entropy)*loss_weights).mean(), torch.maximum(
            divergence.mean(),(divergence*loss_weights).mean())
    trials = []
    for _ in range(config['actor_steps']):
        loss, kl = objective()
        if float(kl.detach()) > config['target_kl']: break
        assert bool(torch.isfinite(loss))
        scope.actor_optimizer.zero_grad(set_to_none=True); loss.backward()
        selected_parameters=fire_freeze.parameters if fire_freeze else scope.actor_parameters
        norm = nn.utils.clip_grad_norm_(selected_parameters, config['max_grad_norm'])
        assert bool(torch.isfinite(norm))
        trial = guarded_actor_step(selected_parameters, scope.actor_optimizer, objective,
            loss.detach(), config['actor_lr'], config['target_kl'], None if fire_freeze else scope.project,
            group_lrs=[g['lr'] for g in scope.actor_optimizer.param_groups],
            momentum_parameters=fire_freeze.parameters if fire_freeze else gate_parameters if actor_scope=='wide_only' else None)
        trials.append(trial)
        if not trial['accepted']: break
    for _ in range(config['value_steps']):
        value, _ = sequence(scope.value, prepared)
        squared=(value[:, 0]-ret).square()
        loss = (squared*loss_weights).mean() if episode_report is not None else squared.mean()
        assert bool(torch.isfinite(loss))
        scope.value_optimizer.zero_grad(); loss.backward()
        norm = nn.utils.clip_grad_norm_(scope.value.parameters(), config['max_grad_norm'])
        assert bool(torch.isfinite(norm)); scope.value_optimizer.step()
    with torch.no_grad():
        _, final_kl = objective(); assert float(final_kl) <= config['target_kl']+1e-6
    if fire_freeze:fire_freeze.verify()
    for path, digest in list(pins.items()): pin(path, digest)
    a.out.mkdir(parents=True)
    save(a.out/'weights.json', scope.export())
    cp = scope.snapshot(sha(a.out/'weights.json'), consumed+tokens)
    cp.update(consumed_seeds=sorted(set(old_seeds)|seeds), updates_completed=updates+1,
              parent_checkpoint_sha256=sha(a.resume) if a.resume else None,
              initialization='episode_balance_fork_preserved_full_adam' if a.fork_episode_balance else 'conditional_fire_scope_fork_preserved_full_adam' if a.fork_conditional_fire else 'resumed_full_joint_adam' if a.resume else 'explicit_fresh_full_policy_optimizer_boundary')
    if fire_freeze:cp['conditional_fire_updates_completed']=(0 if a.fork_conditional_fire else checkpoint.get('conditional_fire_updates_completed',0))+1
    torch.save(cp, a.out/'checkpoint.pt')
    if actor_scope=='wide_only':
        from combat_full_joint_ppo import equal_tree
        assert equal_tree(scope.actor.state_dict(),frozen_actor)
        assert equal_tree({k:v for k,v in scope.joint.state_dict().items() if not k.startswith('wide.')},frozen_joint)
        assert torch.equal(scope.std.detach(),frozen_std)
        assert all(equal_tree(scope.actor_optimizer.state[p],v) for p,v in frozen_moments.items())
    report = dict(version=scope.scope, state='passed', device='cuda', training_performed=True,actor_scope=actor_scope,
        legacy_policy_parameters_unchanged=actor_scope=='wide_only',legacy_policy_moments_unchanged=actor_scope=='wide_only',
        non_fire_actor_parameters_unchanged=bool(fire_freeze),non_fire_adam_states_unchanged=bool(fire_freeze),
        conditional_fire_updates_completed=cp.get('conditional_fire_updates_completed'),
        fire_phase_balance=phase_report,
        episode_balance=episode_report,
        wide_event_rows=int(wide_data['active'].sum()),wide_selected_rows=int(wide_data['choice'].sum()),
        rows=len(rows), train_seeds=sorted(seeds), updates_completed=updates+1,
        train_request_version=request['version'],condition_subset_sha256=request.get('selection_sha256'),
        actor_steps=sum(t['accepted'] for t in trials), actor_trials=trials, final_approx_kl=float(final_kl),
        old_log_probability_max_error=lp_error, old_value_max_error=value_error,
        behavior_sha256=model_hash, weights_sha256=sha(a.out/'weights.json'), checkpoint_sha256=sha(a.out/'checkpoint.pt'),
        optimizer_initialization=cp['initialization'], inputs=pins,
        scope='Fresh native TRAIN. Explicit all, wide_only or conditional_fire_only actor scope; critic fitted separately, geometry predictor frozen. No promotion or quality claim.')
    save(a.out/'report.json', report)
    save(a.out/'complete.json', dict(state='complete', weights_sha256=report['weights_sha256'],
        checkpoint_sha256=report['checkpoint_sha256'], report_sha256=sha(a.out/'report.json')))
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs','actor_trials')}))


if __name__ == '__main__':
    main()

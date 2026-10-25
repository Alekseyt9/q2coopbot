"""Full CUDA PPO on pinned, fresh registered TRAIN captures, with exact resume."""
import argparse
import json
from pathlib import Path
from ppo_combat import torch, nn, training_devices, advantages, validate_objective, guarded_actor_step
from ppo_recurrent import prepare_context, sequence
from combat_full_joint_ppo import FullJointScope, SCOPE
from combat_joint_aim_policy import events, probabilities
from evaluate_registered_combat_models import read, save, sha


def admit(request, model_hash, consumed, consumed_seeds, pin):
    assert request['version'] == 'combat_full_joint_train_request_v1' and request['split'] == 'train'
    plan = read(pin(request['plan'], request['plan_sha256']))
    pool = read(pin(request['pool_report'], request['pool_report_sha256']))
    assert plan['model_sha256'] == model_hash
    assert pool['state'] == 'complete' and pool['source_unchanged'] is True
    assert any(p['sha256'] == request['plan_sha256'] and p['model_sha256'] == model_hash for p in pool['plans'])
    expected = {}
    for task in plan['tasks']:
        assert task['split'] == 'train' and task['modes'] == ['learned']
        assert task['episode']['recipe']['require_native_objective'] is True
        for instance in task['instances']:
            seed = instance['engine_seed']
            assert instance['split'] == 'train' and seed not in expected
            expected[seed] = instance
    rows, history, tokens, seeds, metadata = [], [], [], set(), []
    assert request['cases']
    for case in request['cases']:
        seed = case['seed']; directory = Path(case['data']).resolve(); capture = Path(case['capture_root']).resolve()
        assert seed in expected and seed not in seeds and seed not in consumed_seeds
        jobs = [j for j in pool['jobs'] if j['seed'] == seed and j['mode'] == 'learned']
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
    assert config['version'] == SCOPE and 1 <= config['actor_steps'] <= 100 and 1 <= config['value_steps'] <= 100
    assert 0 < config['gamma'] <= 1 and 0 <= config['lambda'] <= 1 and 0 < config['clip'] <= .3
    assert 0 < config['target_kl'] <= .05 and config['entropy'] >= 0 and config['max_grad_norm'] > 0
    torch.manual_seed(config['seed'])
    scope = FullJointScope(model, config)
    consumed, old_seeds, updates = [], [], 0
    if a.resume:
        cp = torch.load(pin(a.resume), map_location='cuda', weights_only=True)
        consumed = scope.restore(cp, model_hash)
        old_seeds = cp['consumed_seeds']; updates = cp['updates_completed']
    request = read(pin(a.request))
    rows, history, tokens, seeds, metadata = admit(request, model_hash, consumed, old_seeds, pin)
    for meta in metadata: validate_objective(config, meta)
    prepared = prepare_context(history, rows, 'cuda')
    x = torch.tensor([r['features'] for r in rows], device='cuda', dtype=torch.float32)
    cat = lambda key: torch.tensor([r['sample'].get(key, 0) for r in rows], device='cuda', dtype=torch.long)
    target, mode, vertical, weapon = (cat(k) for k in ('target', 'aim_mode', 'vertical', 'weapon'))
    final = torch.tensor([r['sample']['latent'] for r in rows], device='cuda', dtype=torch.float32)
    attack = torch.tensor([float(r['sample']['attack']) for r in rows], device='cuda')
    source, choice, timing, _ = events(rows, x, target, mode, vertical, weapon)
    score_inputs = None
    if scope.joint.geometry_head:
        score_inputs = (
            torch.tensor([r['features'] for r in rows], device='cuda', dtype=torch.float64),
            torch.tensor([(r['sample'].get('aim_candidate') or {}).get('source_latent',r['sample']['latent']) for r in rows], device='cuda', dtype=torch.float64),
            torch.tensor([(r['sample'].get('aim_candidate') or {}).get('timing',[0.,0.]) for r in rows], device='cuda', dtype=torch.float64))
    old = torch.tensor([r['sample']['log_probability'] for r in rows], device='cuda')
    def likelihood(raw):
        return probabilities(scope.joint, raw, scope.std, source, final, attack, vertical,
            weapon, x, target, mode, choice, timing, scope.actor.sampled_motion_aim_branch,
            scope.actor.motor_branch, scope.actor.conditional_fire_branch, score_inputs=score_inputs)[:2]
    with torch.no_grad():
        raw, actor_memory = sequence(scope.actor, prepared, 32, True, 'actor')
        value, value_memory = sequence(scope.value, prepared, 32, True, 'value')
        lp, _ = likelihood(raw)
        lp_error = float((lp-old).abs().max())
        value_error = float((value[:, 0]-torch.tensor([r['sample']['value'] for r in rows], device='cuda')).abs().max())
        assert lp_error < .003 and value_error < 1e-4 and max(actor_memory, value_memory) < 2e-5
    adv, returns = advantages(rows, config['gamma'], config['lambda'])
    ad = torch.tensor(adv, device='cuda'); ad = (ad-ad.mean())/(ad.std(unbiased=False)+1e-8)
    ret = torch.tensor(returns, device='cuda')
    def objective():
        raw, _ = sequence(scope.actor, prepared)
        lp, entropy = likelihood(raw); delta = lp-old; ratio = delta.exp()
        return (-torch.minimum(ratio*ad, ratio.clamp(1-config['clip'],1+config['clip'])*ad).mean()
                -config['entropy']*entropy.mean()), ((ratio-1)-delta).mean()
    trials = []
    for _ in range(config['actor_steps']):
        loss, kl = objective()
        if float(kl.detach()) > config['target_kl']: break
        assert bool(torch.isfinite(loss))
        scope.actor_optimizer.zero_grad(); loss.backward()
        norm = nn.utils.clip_grad_norm_(scope.actor_parameters, config['max_grad_norm'])
        assert bool(torch.isfinite(norm))
        trial = guarded_actor_step(scope.actor_parameters, scope.actor_optimizer, objective,
            loss.detach(), config['actor_lr'], config['target_kl'], scope.project,
            group_lrs=[g['lr'] for g in scope.actor_optimizer.param_groups])
        trials.append(trial)
        if not trial['accepted']: break
    for _ in range(config['value_steps']):
        value, _ = sequence(scope.value, prepared)
        loss = (value[:, 0]-ret).square().mean(); assert bool(torch.isfinite(loss))
        scope.value_optimizer.zero_grad(); loss.backward()
        norm = nn.utils.clip_grad_norm_(scope.value.parameters(), config['max_grad_norm'])
        assert bool(torch.isfinite(norm)); scope.value_optimizer.step()
    with torch.no_grad():
        _, final_kl = objective(); assert float(final_kl) <= config['target_kl']+1e-6
    for path, digest in list(pins.items()): pin(path, digest)
    a.out.mkdir(parents=True)
    save(a.out/'weights.json', scope.export())
    cp = scope.snapshot(sha(a.out/'weights.json'), consumed+tokens)
    cp.update(consumed_seeds=sorted(set(old_seeds)|seeds), updates_completed=updates+1,
              parent_checkpoint_sha256=sha(a.resume) if a.resume else None,
              initialization='resumed_full_joint_adam' if a.resume else 'explicit_fresh_full_policy_optimizer_boundary')
    torch.save(cp, a.out/'checkpoint.pt')
    report = dict(version=SCOPE, state='passed', device='cuda', training_performed=True,
        rows=len(rows), train_seeds=sorted(seeds), updates_completed=updates+1,
        actor_steps=sum(t['accepted'] for t in trials), actor_trials=trials, final_approx_kl=float(final_kl),
        old_log_probability_max_error=lp_error, old_value_max_error=value_error,
        behavior_sha256=model_hash, weights_sha256=sha(a.out/'weights.json'), checkpoint_sha256=sha(a.out/'checkpoint.pt'),
        optimizer_initialization=cp['initialization'], inputs=pins,
        scope='Fresh native TRAIN; all policy modules and critic. Frozen geometry predictor. No promotion or quality claim.')
    save(a.out/'report.json', report)
    save(a.out/'complete.json', dict(state='complete', weights_sha256=report['weights_sha256'],
        checkpoint_sha256=report['checkpoint_sha256'], report_sha256=sha(a.out/'report.json')))
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs','actor_trials')}))


if __name__ == '__main__':
    main()

"""Fixed-budget conditional aiming SAC warm update on sealed native replay.

This first offline warm update must pass fresh native evaluation before more
collection. SAC critics/optimizers are saved separately from the inference
checkpoint; no PPO update or general full-policy SAC support is implied.
"""
import argparse
import copy
import pathlib
import time

from process_combat_architecture_pool import read, sha
from ppo_recurrent import torch, durable_json, durable_write
from prepare_combat_target_heads import build
from combat_spatial_aim import export as export_spatial
from combat_conditional_aim_scope import freeze_conditional_aim
from combat_conditional_sac import TwinQ, exogenous, condition_state, sample_aim, aim_distribution, soft_backup, polyak_update


def main():
    ap = argparse.ArgumentParser()
    for name in ('source', 'replay', 'out'): ap.add_argument('--' + name, type=pathlib.Path, required=True)
    ap.add_argument('--warmup', type=int, default=256)
    ap.add_argument('--steps', type=int, default=256)
    a = ap.parse_args()
    assert torch.cuda.is_available() and not a.out.exists()
    assert 1 <= a.steps <= 512 and 1 <= a.warmup <= 512
    torch.set_default_device('cuda'); torch.set_num_threads(2); torch.manual_seed(20261022)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    source, replay = a.source.resolve(), a.replay.resolve()
    seal = read(source / 'complete.json')
    pins = {str(source / 'complete.json'): sha(source / 'complete.json')}
    for file, key in (('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'), ('report.json', 'report_sha256')):
        assert sha(source / file) == seal[key]; pins[str(source / file)] = seal[key]
    cache = read(replay / 'complete.json'); assert cache['state'] == 'complete' and cache['device'] == 'cuda'
    assert not cache['training_performed'] and not cache['excluded_missing_successor']
    for file, key in (('replay.pt', 'replay_sha256'), ('keys.json', 'keys_sha256')):
        assert sha(replay / file) == cache[key]; pins[str(replay / file)] = cache[key]
    pins[str(replay / 'complete.json')] = sha(replay / 'complete.json')
    assert cache['source_weights_sha256'] == seal['weights_sha256'], 'First warm prototype requires the exact behavior actor'
    cp = torch.load(source / 'checkpoint.pt', map_location='cuda', weights_only=True)
    assert cp['config']['objective_reward_sha256'] == cache['reward_sha256']
    assert not cp['actor_optimizer']['state'], 'Explicit equal optimizer-reset source required'
    data = torch.load(replay / 'replay.pt', map_location='cuda', weights_only=True)
    model = read(source / 'weights.json'); actor = build(model, 'actor')
    initial = {k:v.detach().clone() for k,v in actor.state_dict().items()}
    std = torch.nn.Parameter(cp['log_std'].detach().clone())
    initial_std = std.detach().clone()
    actor_optimizer = torch.optim.Adam([*actor.sampled_motion_aim_branch.parameters(), std], lr=cp['config']['actor_lr'])
    project, trainable = freeze_conditional_aim(actor, std, actor_optimizer)
    x, raw, history = (data[k] for k in ('features', 'raw', 'history'))
    current, successor = data['current'], data['successor']
    state_width = x.shape[1] + history.shape[1] + 29
    critic = TwinQ(state_width); target_critic = copy.deepcopy(critic)
    for parameter in target_critic.parameters(): parameter.requires_grad_(False)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=.0003)
    protocol = dict(version='combat_conditional_aim_sac_v1', source=str(source), replay=str(replay),
        rows=cache['rows'], training_cases=cache['training_cases'], warmup_steps=a.warmup, actor_steps=a.steps,
        batch_size=256, actor_lr=cp['config']['actor_lr'], critic_lr=.0003, gamma=cp['config']['gamma'],
        entropy_alpha=.001, target_polyak_retention=.995, seed=20261022, selection_on_evaluation=False,
        actor_scope='conditional sampled-motion aim branch and aim log_std only',
        setting='offline first warm update; no fresh post-update collection or general SAC superiority yet',
        inputs=[dict(path=p, sha256=h) for p,h in pins.items()], script_sha256=sha(__file__))
    a.out.mkdir(parents=True); durable_json(a.out / 'protocol.json', protocol)
    rows = len(current); trajectory = []; started = time.perf_counter()

    def random_batch(): return torch.randint(rows, (256,), device='cuda')

    def fit_critic(indices):
        ci, ni = current[indices], successor[indices]
        state = condition_state(x[ci], history[ci], data['condition'][indices], data['latent'][indices, :2])
        with torch.no_grad():
            choice, movement = exogenous(raw[ni], x[ni], std)
            next_state = condition_state(x[ni], history[ni], choice, movement)
            action, lp = sample_aim(raw[ni], x[ni], choice, movement, actor.sampled_motion_aim_branch, std)
            qa, qb = target_critic(next_state, action)
            target = soft_backup(data['reward'][indices], data['bootstrap'][indices], qa, qb, lp, protocol['gamma'], protocol['entropy_alpha'])
        first, second = critic(state, data['latent'][indices, 2:].tanh())
        loss = ((first-target).square() + (second-target).square()).mean()
        assert bool(torch.isfinite(loss))
        critic_optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.)
        critic_optimizer.step(); polyak_update(critic, target_critic, .995)
        return float(loss.detach())

    for step in range(a.warmup + a.steps):
        indices = random_batch(); critic_loss = fit_critic(indices)
        actor_loss = None
        if step >= a.warmup:
            ci = current[random_batch()]
            with torch.no_grad(): choice, movement = exogenous(raw[ci], x[ci], std)
            state = condition_state(x[ci], history[ci], choice, movement)
            for parameter in critic.parameters(): parameter.requires_grad_(False)
            action, lp = sample_aim(raw[ci], x[ci], choice, movement, actor.sampled_motion_aim_branch, std)
            first, second = critic(state, action)
            loss = (protocol['entropy_alpha'] * lp - torch.minimum(first, second)).mean()
            assert bool(torch.isfinite(loss))
            actor_optimizer.zero_grad(); loss.backward()
            assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in actor.parameters())
            torch.nn.utils.clip_grad_norm_([*actor.sampled_motion_aim_branch.parameters(), std], .5)
            actor_optimizer.step()
            with torch.no_grad(): project()
            for parameter in critic.parameters(): parameter.requires_grad_(True)
            actor_loss = float(loss.detach())
        if (step + 1) % 64 == 0:
            trajectory.append(dict(step=step+1, critic_loss=critic_loss, actor_loss=actor_loss))
            durable_json(a.out / ('step-' + str(step+1) + '.json'), trajectory[-1])
    for key, value in actor.state_dict().items():
        assert bool(torch.isfinite(value).all())
        if not key.startswith('sampled_motion_aim_branch.'): assert torch.equal(value, initial[key]), key
    assert torch.equal(std[:2], initial_std[:2])
    for group, expected in ((actor_optimizer, a.steps), (critic_optimizer, a.steps+a.warmup)):
        assert group.state
        for state in group.state.values():
            assert float(state['step']) == expected
            for value in state.values():
                if torch.is_tensor(value): assert value.is_cuda and bool(torch.isfinite(value).all())
    updated = copy.deepcopy(model); export_spatial(updated, actor)
    updated['log_std'] = std.detach().cpu().tolist()
    for key, value in build(updated, 'actor').state_dict().items(): assert torch.equal(value, actor.state_dict()[key])
    durable_json(a.out / 'weights.json', updated)
    inference = copy.deepcopy(cp)
    inference.update(actor=actor.state_dict(), log_std=std.detach(), weights_sha256=sha(a.out / 'weights.json'))
    inference['actor_training'] = dict(scope='conditional-aim', learning_rate=cp['config']['actor_lr'],
        weapon_state_residual=cp['actor_training']['weapon_state_residual'])
    inference['sac_training'] = dict(version=protocol['version'], protocol_sha256=sha(a.out / 'protocol.json'),
        actor_steps=a.steps, critic_steps=a.steps+a.warmup, ppo_optimizer_reset=True, explicit_algorithm_fork_required=True)
    inference['consumed_rollouts'] = list(cp['consumed_rollouts']) + [read(pathlib.Path(cache['corpus']) / 'processing-train/merged/report.json')['rollout_sha256']]
    durable_write(a.out / 'checkpoint.pt', lambda f: torch.save(inference, f))
    sac_checkpoint = dict(version=protocol['version'], actor=actor.state_dict(), log_std=std.detach(),
        critic=critic.state_dict(), target_critic=target_critic.state_dict(), actor_optimizer=actor_optimizer.state_dict(),
        critic_optimizer=critic_optimizer.state_dict(), actor_steps=a.steps, critic_steps=a.steps+a.warmup,
        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), weights_sha256=sha(a.out / 'weights.json'))
    durable_write(a.out / 'sac-checkpoint.pt', lambda f: torch.save(sac_checkpoint, f))
    restored = torch.load(a.out / 'sac-checkpoint.pt', map_location='cuda', weights_only=True)
    for name in ('actor', 'critic', 'target_critic'):
        for key, value in sac_checkpoint[name].items(): assert torch.equal(value, restored[name][key])
    for path, digest in pins.items(): assert sha(path) == digest
    durable_json(a.out / 'report.json', dict(state='complete', device='cuda', algorithm=protocol['version'],
        training_performed=True, promotion=False, rows=rows, training_cases=cache['training_cases'], actor_steps=a.steps,
        critic_steps=a.steps+a.warmup, trainable_actor_parameters=trainable, seconds=time.perf_counter()-started,
        source_weights_sha256=seal['weights_sha256'], protocol_sha256=sha(a.out / 'protocol.json'),
        sac_checkpoint_sha256=sha(a.out / 'sac-checkpoint.pt'), history=trajectory,
        frozen_other_actor_tensors_and_movement_std=True, optimizer_steps_and_finiteness_verified=True,
        scope='Offline warm conditional SAC aim update on TRAIN replay, clipped twin Q/soft target/normalized tanh Gaussian entropy. Other policy factors and encoder frozen. Q-only cached history uses current/past observations. Inference checkpoint retains legacy container; SAC state separate, PPO counters not incremented. Requires independent fresh native quality evaluation and fresh feedback before any claim about efficient online SAC.'))
    durable_json(a.out / 'complete.json', dict(version='combat_update_complete_v1',
        weights_sha256=sha(a.out / 'weights.json'), checkpoint_sha256=sha(a.out / 'checkpoint.pt'), report_sha256=sha(a.out / 'report.json')))
    print(dict(state='complete', actor_steps=a.steps, critic_steps=a.steps+a.warmup, rows=rows, seconds=time.perf_counter()-started))


if __name__ == '__main__': main()

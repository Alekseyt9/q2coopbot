"""Fixed-budget conditional aiming SAC warm update on sealed native replay.

This first offline warm update must pass fresh native evaluation before more
collection. SAC critics/optimizers are saved separately from the inference
checkpoint; no PPO update or general full-policy SAC support is implied.
"""
import argparse
import copy
import pathlib
import time
import math

from process_combat_architecture_pool import read, sha
from ppo_recurrent import torch, durable_json, durable_write
from prepare_combat_target_heads import build
from combat_spatial_aim import export as export_spatial
from combat_conditional_aim_scope import freeze_conditional_aim
from combat_conditional_sac import TwinQ, exogenous, condition_state, sample_aim, aim_distribution, behavior_aim_kl, soft_backup, polyak_update


def main():
    ap = argparse.ArgumentParser()
    for name in ('source', 'out'): ap.add_argument('--' + name, type=pathlib.Path, required=True)
    ap.add_argument('--replay',type=pathlib.Path,nargs='+',required=True)
    ap.add_argument('--resume',action='store_true',help='Continue the sealed SAC state beside the source inference checkpoint')
    ap.add_argument('--warmup', type=int)
    ap.add_argument('--steps', type=int, default=256)
    ap.add_argument('--behavior-kl',type=float,default=0.)
    ap.add_argument('--record-warmup-boundary',action='store_true')
    a = ap.parse_args()
    assert torch.cuda.is_available() and not a.out.exists()
    if a.warmup is None:a.warmup=0 if a.resume else 256
    assert 1 <= a.steps <= 512 and 0 <= a.warmup <= 512
    assert math.isfinite(a.behavior_kl) and 0<=a.behavior_kl<=1
    torch.set_default_device('cuda'); torch.set_num_threads(2); torch.manual_seed(20261022)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    source = a.source.resolve()
    seal = read(source / 'complete.json')
    pins = {str(source / 'complete.json'): sha(source / 'complete.json')}
    for file, key in (('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'), ('report.json', 'report_sha256')):
        assert sha(source / file) == seal[key]; pins[str(source / file)] = seal[key]
    cp = torch.load(source / 'checkpoint.pt', map_location='cuda', weights_only=True)
    assert not cp['actor_optimizer']['state'], 'Explicit equal optimizer-reset source required'
    model = read(source / 'weights.json')
    assert not model.get('conditional_fire'), 'SAC sampler does not support conditional fire'
    actor = build(model, 'actor')
    def pin(path,expected=None):
        path=pathlib.Path(path);digest=sha(path);assert expected is None or digest==expected,str(path)
        pins[str(path.resolve())]=digest;return path
    from combat_replay_catalog import load_catalog
    data,catalog,family=load_catalog(a.replay,model,cp['config']['objective_reward_sha256'],pin)
    if not a.resume:
        assert all(m['behavior_weights_sha256']==seal['weights_sha256'] for m in catalog), 'Initial warm update requires exact behavior actor'
    initial = {k:v.detach().clone() for k,v in actor.state_dict().items()}
    std = torch.nn.Parameter(cp['log_std'].detach().clone())
    initial_std = std.detach().clone()
    teacher_branch=copy.deepcopy(actor.sampled_motion_aim_branch) if a.behavior_kl else None
    teacher_std=initial_std.clone();anchor_weights_sha256=seal['weights_sha256']
    if teacher_branch is not None:
        for parameter in teacher_branch.parameters():parameter.requires_grad_(False)
    actor_optimizer = torch.optim.Adam([*actor.sampled_motion_aim_branch.parameters(), std], lr=cp['config']['actor_lr'])
    x, raw, history = (data[k] for k in ('features', 'raw', 'history'))
    current, successor = data['current'], data['successor']
    state_width = x.shape[1] + history.shape[1] + 29
    critic = TwinQ(state_width); target_critic = copy.deepcopy(critic)
    for parameter in target_critic.parameters(): parameter.requires_grad_(False)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=.0003)
    previous_actor_steps=previous_critic_steps=0
    if a.resume:
        report=read(source/'report.json');path=pin(source/'sac-checkpoint.pt',report['sac_checkpoint_sha256'])
        previous_protocol=read(pin(source/'protocol.json',report['protocol_sha256']))
        assert previous_protocol.get('behavior_kl_weight',0.)==a.behavior_kl,'Explicit behavior KL objective fork required'
        for key,expected in dict(batch_size=256,actor_lr=cp['config']['actor_lr'],critic_lr=.0003,
            gamma=cp['config']['gamma'],entropy_alpha=.001,target_polyak_retention=.995).items():
            assert previous_protocol[key]==expected,'Explicit SAC hyperparameter fork required: '+key
        previous=torch.load(path,map_location='cuda',weights_only=True)
        assert previous['version']=='combat_conditional_aim_sac_v1' and previous['weights_sha256']==seal['weights_sha256']
        for key,value in initial.items():assert torch.equal(value,previous['actor'][key])
        assert torch.equal(std,previous['log_std'])
        if a.behavior_kl:
            anchor=previous['behavior_anchor'];teacher_branch.load_state_dict(anchor['branch'])
            teacher_std=anchor['log_std'];anchor_weights_sha256=anchor['weights_sha256']
        critic.load_state_dict(previous['critic']);target_critic.load_state_dict(previous['target_critic'])
        actor_optimizer.load_state_dict(previous['actor_optimizer']);critic_optimizer.load_state_dict(previous['critic_optimizer'])
        previous_actor_steps=previous['actor_steps'];previous_critic_steps=previous['critic_steps']
        assert previous_actor_steps==report.get('total_actor_steps',report['actor_steps'])
        assert previous_critic_steps==report.get('total_critic_steps',report['critic_steps'])
        torch.set_rng_state(previous['rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in previous['cuda_rng']])
    project, trainable = freeze_conditional_aim(actor, std, actor_optimizer)
    protocol = dict(version='combat_conditional_aim_sac_v1', source=str(source), replay_catalog=catalog,policy_family_sha256=family,
        rows=len(current), training_cases=sum(m['cases'] for m in catalog), warmup_steps=a.warmup, actor_steps=a.steps,
        resume_sealed_update=a.resume,previous_actor_steps=previous_actor_steps,previous_critic_steps=previous_critic_steps,
        batch_size=256, actor_lr=cp['config']['actor_lr'], critic_lr=.0003, gamma=cp['config']['gamma'],
        entropy_alpha=.001, target_polyak_retention=.995, seed=20261022, selection_on_evaluation=False,
        behavior_kl_weight=a.behavior_kl,behavior_anchor_weights_sha256=anchor_weights_sha256 if a.behavior_kl else None,
        actor_scope='conditional sampled-motion aim branch and aim log_std only',
        setting='sequential sealed SAC continuation on validated TRAIN replay' if a.resume else 'offline first warm update',
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
        if a.record_warmup_boundary and step==a.warmup:
            durable_write(a.out/'warmup-boundary.pt',lambda f:torch.save(dict(actor=actor.state_dict(),log_std=std.detach(),
                critic=critic.state_dict(),target_critic=target_critic.state_dict(),actor_optimizer=actor_optimizer.state_dict(),
                critic_optimizer=critic_optimizer.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),f))
        indices = random_batch(); critic_loss = fit_critic(indices)
        actor_loss = None
        kl_value=None
        if step >= a.warmup:
            ci = current[random_batch()]
            with torch.no_grad(): choice, movement = exogenous(raw[ci], x[ci], std)
            state = condition_state(x[ci], history[ci], choice, movement)
            for parameter in critic.parameters(): parameter.requires_grad_(False)
            action, lp = sample_aim(raw[ci], x[ci], choice, movement, actor.sampled_motion_aim_branch, std)
            first, second = critic(state, action)
            loss = (protocol['entropy_alpha'] * lp - torch.minimum(first, second)).mean()
            if a.behavior_kl:
                kl=behavior_aim_kl(raw[ci],x[ci],choice,movement,actor.sampled_motion_aim_branch,std,teacher_branch,teacher_std).mean()
                loss=loss+a.behavior_kl*kl;kl_value=float(kl.detach())
            assert bool(torch.isfinite(loss))
            actor_optimizer.zero_grad(); loss.backward()
            assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in actor.parameters())
            torch.nn.utils.clip_grad_norm_([*actor.sampled_motion_aim_branch.parameters(), std], .5)
            actor_optimizer.step()
            with torch.no_grad(): project()
            for parameter in critic.parameters(): parameter.requires_grad_(True)
            actor_loss = float(loss.detach())
        if (step + 1) % 64 == 0 or step+1==a.warmup+a.steps:
            trajectory.append(dict(step=step+1, critic_loss=critic_loss, actor_loss=actor_loss,behavior_kl=kl_value))
            durable_json(a.out / ('step-' + str(step+1) + '.json'), trajectory[-1])
    for key, value in actor.state_dict().items():
        assert bool(torch.isfinite(value).all())
        if not key.startswith('sampled_motion_aim_branch.'): assert torch.equal(value, initial[key]), key
    assert torch.equal(std[:2], initial_std[:2])
    total_actor_steps=previous_actor_steps+a.steps;total_critic_steps=previous_critic_steps+a.steps+a.warmup
    for group, expected in ((actor_optimizer, total_actor_steps), (critic_optimizer, total_critic_steps)):
        assert group.state
        for state in group.state.values():
            assert float(state['step']) == expected
            for value in state.values():
                if torch.is_tensor(value): assert value.is_cuda and bool(torch.isfinite(value).all())
    # Save the training boundary, before export/read-back constructors consume
    # RNG. Closed 1+1 continuation must equal an uninterrupted two-step update.
    training_rng=torch.get_rng_state().clone();training_cuda_rng=[v.clone() for v in torch.cuda.get_rng_state_all()]
    updated = copy.deepcopy(model); export_spatial(updated, actor)
    updated['log_std'] = std.detach().cpu().tolist()
    for key, value in build(updated, 'actor').state_dict().items(): assert torch.equal(value, actor.state_dict()[key])
    durable_json(a.out / 'weights.json', updated)
    inference = copy.deepcopy(cp)
    inference.update(actor=actor.state_dict(), log_std=std.detach(), weights_sha256=sha(a.out / 'weights.json'))
    inference['actor_training'] = dict(scope='conditional-aim', learning_rate=cp['config']['actor_lr'],
        weapon_state_residual=cp['actor_training']['weapon_state_residual'])
    inference['sac_training'] = dict(version=protocol['version'], protocol_sha256=sha(a.out / 'protocol.json'),
        actor_steps=total_actor_steps, critic_steps=total_critic_steps, ppo_optimizer_reset=True, explicit_algorithm_fork_required=True)
    inference['consumed_rollouts'] = list(cp['consumed_rollouts'])
    for item in catalog:
        if item['rollout_sha256'] not in inference['consumed_rollouts']:inference['consumed_rollouts'].append(item['rollout_sha256'])
    durable_write(a.out / 'checkpoint.pt', lambda f: torch.save(inference, f))
    sac_checkpoint = dict(version=protocol['version'], actor=actor.state_dict(), log_std=std.detach(),
        critic=critic.state_dict(), target_critic=target_critic.state_dict(), actor_optimizer=actor_optimizer.state_dict(),
        critic_optimizer=critic_optimizer.state_dict(), actor_steps=total_actor_steps, critic_steps=total_critic_steps,
        rng=training_rng, cuda_rng=training_cuda_rng, weights_sha256=sha(a.out / 'weights.json'))
    if teacher_branch is not None:sac_checkpoint['behavior_anchor']=dict(branch=teacher_branch.state_dict(),log_std=teacher_std,weights_sha256=anchor_weights_sha256)
    durable_write(a.out / 'sac-checkpoint.pt', lambda f: torch.save(sac_checkpoint, f))
    restored = torch.load(a.out / 'sac-checkpoint.pt', map_location='cuda', weights_only=True)
    for name in ('actor', 'critic', 'target_critic'):
        for key, value in sac_checkpoint[name].items(): assert torch.equal(value, restored[name][key])
    for path, digest in pins.items(): assert sha(path) == digest
    durable_json(a.out / 'report.json', dict(state='complete', device='cuda', algorithm=protocol['version'],
        training_performed=True, promotion=False, rows=rows, training_cases=protocol['training_cases'], actor_steps=a.steps,
        critic_steps=a.steps+a.warmup,total_actor_steps=total_actor_steps,total_critic_steps=total_critic_steps,
        resumed=a.resume,replay_catalog=catalog,policy_family_sha256=family,trainable_actor_parameters=trainable, seconds=time.perf_counter()-started,
        behavior_kl_weight=a.behavior_kl,behavior_anchor_weights_sha256=anchor_weights_sha256 if a.behavior_kl else None,
        warmup_boundary_sha256=sha(a.out/'warmup-boundary.pt') if a.record_warmup_boundary else None,
        source_weights_sha256=seal['weights_sha256'], protocol_sha256=sha(a.out / 'protocol.json'),
        sac_checkpoint_sha256=sha(a.out / 'sac-checkpoint.pt'), history=trajectory,
        frozen_other_actor_tensors_and_movement_std=True, optimizer_steps_and_finiteness_verified=True,
        scope=('Sealed conditional SAC feedback continuation' if a.resume else 'Offline warm conditional SAC aim update')
            +' on TRAIN replay, clipped twin Q/soft target/normalized tanh Gaussian entropy. Other policy factors and encoder frozen. Q-only cached history uses current/past observations. Inference checkpoint retains legacy container; SAC state separate, PPO counters not incremented. Independent native evaluation and measured environment/compute budgets required for any quality or efficiency claim.'))
    durable_json(a.out / 'complete.json', dict(version='combat_update_complete_v1',
        weights_sha256=sha(a.out / 'weights.json'), checkpoint_sha256=sha(a.out / 'checkpoint.pt'), report_sha256=sha(a.out / 'report.json')))
    print(dict(state='complete', actor_steps=a.steps, critic_steps=a.steps+a.warmup, rows=rows, seconds=time.perf_counter()-started))


if __name__ == '__main__': main()

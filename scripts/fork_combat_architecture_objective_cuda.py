"""Explicit CUDA objective fork for sealed GRU/temporal PPO checkpoints.

Actor and exploration are preserved; both Adam states are reset and the critic
output is zeroed. Run identically for control and changed-reward A/B branches.
Fresh native on-policy collection is required before any update.
"""
import argparse, copy, pathlib, shutil
from process_combat_architecture_pool import read, save, sha
from prepare_combat_target_heads import build, forward
from train_combat_bc import torch


def main():
    ap = argparse.ArgumentParser()
    for name in ('parent', 'config', 'reward', 'out'):
        ap.add_argument('--'+name, type=pathlib.Path, required=True)
    ap.add_argument('--preserve-critic-optimizer', action='store_true',
                    help='Explicit warm objective fork; keep critic and both Adam states')
    ap.add_argument('--migration-parent', action='store_true',
                    help='Use verified CUDA navigation migration receipt instead of a training completion seal')
    a = ap.parse_args()
    assert torch.cuda.is_available() and not a.out.exists()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    seal = read(a.parent/('migration.json' if a.migration_parent else 'complete.json'))
    if a.migration_parent:
        assert seal['version'] == 'combat_navigation_checkpoint_migration_v1'
        assert seal['device'] == 'cuda' and seal['persisted_restoration_verified']
        assert not seal['training_performed'] and seal['persisted_optimizer_steps'] == 0
        assert all(v['optimizer_restored'] and v['disposable_cuda_step'] for v in seal['optimizer_checks'].values())
    files = [('weights.json','weights_sha256'), ('checkpoint.pt','checkpoint_sha256')]
    if not a.migration_parent: files.append(('report.json','report_sha256'))
    for filename, key in files:
        assert sha(a.parent/filename) == seal[key]
    model = read(a.parent/'weights.json')
    assert not model['deterministic'] and bool(model.get('memory') or model.get('attention'))
    state = torch.load(a.parent/'checkpoint.pt', map_location='cuda', weights_only=True)
    assert state['version'] == 'combat_architecture_checkpoint_v1'
    assert state['weights_sha256'] == sha(a.parent/'weights.json')
    config, reward = read(a.config), read(a.reward)
    assert config['objective_reward_sha256'] == sha(a.reward)
    assert reward['version'] in ('combat_reward_v4','combat_reward_v5','combat_reward_v6','combat_reward_v7','combat_reward_v8','combat_reward_v9','combat_reward_v10','combat_reward_v11')
    assert reward['monster_kill'] > 0 and reward['aim_gamma'] == config['gamma']
    if reward['version'] in ('combat_reward_v5','combat_reward_v9'):
        assert -0.1 <= reward['blaster_miss'] < 0
    if reward['version'] in ('combat_reward_v6','combat_reward_v7','combat_reward_v8','combat_reward_v9','combat_reward_v10','combat_reward_v11'):
        assert all(-0.05 <= reward[name] < 0 for name in ('off_target_attack','turn_away','stalled_movement'))
    if reward['version'] == 'combat_reward_v7':
        assert -0.05 <= reward['target_churn'] < 0
    if reward['version'] == 'combat_reward_v9':
        assert -0.1 <= reward['machinegun_miss'] < 0
    if reward['version'] in ('combat_reward_v10','combat_reward_v11'):
        assert 0 < reward['navigation_potential'] <= 0.1
    if reward['version'] == 'combat_reward_v11':
        assert reward['peer_death'] < 0
    assert {k:v for k,v in state['config'].items() if k!='objective_reward_sha256'} == {k:v for k,v in config.items() if k!='objective_reward_sha256'}
    actor, value = build(model,'actor'), build(model,'value')
    for name, module in [('actor',actor), ('value',value)]:
        expected, actual = module.state_dict(), state[name]
        assert expected.keys() == actual.keys()
        assert all(v.is_cuda and torch.equal(v,actual[k]) for k,v in expected.items())
    assert torch.equal(state['log_std'], torch.tensor(model['log_std'],device='cuda'))
    child = copy.deepcopy(model)
    optimizers_before = {name:copy.deepcopy(state[name]) for name in ('actor_optimizer','value_optimizer')}
    if not a.preserve_critic_optimizer:
        with torch.no_grad():
            for module in (value.head, value.residual if hasattr(value,'residual') else value.output):
                module.weight.zero_(); module.bias.zero_()
        child['value'], cell = value.export()
        key = 'attention' if child.get('attention') else 'memory'
        child[key]['value'].update(cell)
    state['value'] = value.state_dict()
    state['config'] = config
    for name, lr in [('actor_optimizer','actor_lr'), ('value_optimizer','value_lr')]:
        if not a.preserve_critic_optimizer:
            state[name]['state'] = {}
        for group in state[name]['param_groups']:
            group['lr'] = config[lr]
    restored_actor, restored_value = build(child,'actor'), build(child,'value')
    assert all(torch.equal(v,restored_actor.state_dict()[k]) for k,v in actor.state_dict().items())
    assert all(torch.equal(v,restored_value.state_dict()[k]) for k,v in value.state_dict().items())
    width = len(model['actor'][0]['weight'][0])
    x = torch.randn(2,7,width,device='cuda')*.01
    with torch.no_grad():
        assert torch.equal(forward(actor,x),forward(restored_actor,x))
        if a.preserve_critic_optimizer:
            assert torch.equal(forward(value,x),forward(restored_value,x))
        else:
            assert bool((forward(restored_value,x)==0).all())
    a.out.mkdir()
    if a.preserve_critic_optimizer:
        shutil.copyfile(a.parent/'weights.json',a.out/'weights.json')
        assert sha(a.out/'weights.json') == seal['weights_sha256']
    else:
        save(a.out/'weights.json',child)
    state['weights_sha256'] = sha(a.out/'weights.json')
    torch.save(state,a.out/'checkpoint.pt')
    restored = torch.load(a.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    def exact(left,right):
        if torch.is_tensor(left):
            assert left.is_cuda and right.is_cuda and torch.equal(left,right)
        elif isinstance(left,dict):
            assert left.keys()==right.keys()
            for key in left: exact(left[key],right[key])
        elif isinstance(left,(list,tuple)):
            assert len(left)==len(right)
            for x,y in zip(left,right): exact(x,y)
        else:
            assert left==right
    for name in ('actor','value','log_std','actor_optimizer','value_optimizer'):
        exact(state[name],restored[name])
    if a.preserve_critic_optimizer:
        for name,original in optimizers_before.items(): exact(original,restored[name])
    assert restored['config']==config and restored['consumed_rollouts']==state['consumed_rollouts']
    save(a.out/'report.json',dict(version='combat_architecture_objective_fork_cuda_v1',
        device='cuda',gpu=torch.cuda.get_device_name(),parent_weights_sha256=seal['weights_sha256'],
        parent_checkpoint_sha256=seal['checkpoint_sha256'],weights_sha256=sha(a.out/'weights.json'),
        checkpoint_sha256=sha(a.out/'checkpoint.pt'),config_sha256=sha(a.config),
        anchor_sha256=state['anchor_sha256'],bank_sha256=state['bank_sha256'],
        reward_config_sha256=sha(a.reward),actor_std_exact=True,critic_output_zero=not a.preserve_critic_optimizer,
        parent_receipt_sha256=sha(a.parent/('migration.json' if a.migration_parent else 'complete.json')),
        adam_reset=not a.preserve_critic_optimizer,critic_optimizer_preserved=a.preserve_critic_optimizer,
        checkpoint_readback_exact=True,updates_completed=state['updates_completed'],
        scope='Explicit objective fork, not training or a PPO update. Actor/std and consumed history/RNG retained. '
              +('Critic and both Adam states retained exactly; value predictions initially describe the parent objective. ' if a.preserve_critic_optimizer else 'Critic head/residual zeroed and both Adam states reset. ')
              +'CUDA read-back checked. No CPU/Go numerical checks. Fresh native rollouts required; same fork mode applies to the A/B control.'))


if __name__ == '__main__':
    main()

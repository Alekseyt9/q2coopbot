"""Explicit warm-start of a new reward objective, with fresh critic targets."""
import argparse
import json
import pathlib
import sys
sys.pycache_prefix=str(pathlib.Path(__file__).resolve().parents[1]/'workspace/build/python-cache')
from fork_combat_exploration import fork
from ppo_combat import network, layers, sha, torch


def rebase(model,state,config):
    before={k:v for k,v in state['config'].items() if k!='objective_reward_sha256'}
    after={k:v for k,v in config.items() if k!='objective_reward_sha256'}
    if before!=after or not config.get('objective_reward_sha256'):
        raise ValueError('Only explicitly pinned reward objective may change')
    child,state=fork(model,state,model['log_std'])
    critic=network(child['value'])
    with torch.no_grad():
        critic[-1].weight.zero_();critic[-1].bias.zero_()
    child['value']=layers(critic);state['value']=critic.state_dict()
    state['value_optimizer']['state']={}
    for group in state['value_optimizer']['param_groups']:group['lr']=config['value_lr']
    state['config']=config
    return child,state


def main():
    ap=argparse.ArgumentParser()
    for name in ['model','checkpoint','config','reward-config','out']:ap.add_argument('--'+name,required=True,type=pathlib.Path)
    a=ap.parse_args();model_sha,parent_sha=sha(a.model),sha(a.checkpoint)
    config=json.loads(a.config.read_text());reward=json.loads(a.reward_config.read_text())
    if config.get('objective_reward_sha256')!=sha(a.reward_config) or reward.get('version') not in ('combat_reward_v2','combat_reward_v3','combat_reward_v4') or reward.get('monster_kill',0)<=0:
        raise ValueError('Pinned kill reward v2/v3 required')
    if reward['version'] in ('combat_reward_v3','combat_reward_v4') and reward.get('aim_gamma')!=config['gamma']:
        raise ValueError('Shaping gamma differs from PPO')
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if state.get('weights_sha256')!=model_sha:raise ValueError('Parent weights SHA differs')
    child,state=rebase(json.loads(a.model.read_text()),state,config)
    a.out.mkdir(exist_ok=False);weights=a.out/'weights.json';weights.write_text(json.dumps(child,allow_nan=False))
    state['weights_sha256']=sha(weights);torch.save(state,a.out/'checkpoint.pt')
    report=dict(version='combat_objective_fork_v1',parent_model_sha256=model_sha,parent_checkpoint_sha256=parent_sha,
                weights_sha256=sha(weights),reward_config_sha256=sha(a.reward_config),config_sha256=sha(a.config),
                scope='Actor/log_std preserved; critic output zeroed; actor/value Adam reset. RNG/consumed history/counters retained; fresh on-policy rollout required. Fork is not PPO update.')
    if sha(a.model)!=model_sha or sha(a.checkpoint)!=parent_sha:raise ValueError('Parent changed')
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

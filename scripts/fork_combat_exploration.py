"""Fork a PPO v2 checkpoint with explicit exploration; never reuse old rollout."""
import argparse
import copy
import json
import math
import pathlib
import sys

sys.pycache_prefix = str(pathlib.Path(__file__).resolve().parents[1] / 'workspace/build/python-cache')
from ppo_combat import network, torch, sha


def fork(model, checkpoint, log_std):
    if checkpoint.get('version') != 'combat_ppo_checkpoint_v2':
        raise ValueError('PPO checkpoint v2 required')
    if model.get('kind') != 'combat_ppo_v1' or model.get('deterministic'):
        raise ValueError('Stochastic PPO model required')
    if len(log_std) != 4 or any(not math.isfinite(x) or not -8 <= x <= 1 for x in log_std):
        raise ValueError('Four finite log std values in [-8,1] required')
    for name in ['actor', 'value']:
        expected = network(model[name]).state_dict()
        actual = checkpoint[name]
        if expected.keys() != actual.keys() or any(not torch.equal(expected[k], actual[k].cpu()) for k in expected):
            raise ValueError('Checkpoint network differs from JSON')
    if not torch.equal(checkpoint['log_std'].cpu(), torch.tensor(model['log_std'])):
        raise ValueError('Checkpoint exploration differs from JSON')
    result = copy.deepcopy(model)
    state = copy.deepcopy(checkpoint)
    result['log_std'] = list(log_std)
    state['log_std'] = torch.tensor(log_std)
    # Intentional experimental branch: discard actor Adam moments after changing
    # its sampling distribution. Keep value optimizer, RNG and consumed history.
    state['actor_optimizer']['state'] = {}
    for group in state['actor_optimizer']['param_groups']:
        group['lr'] = state['config']['actor_lr']
    return result, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, type=pathlib.Path)
    ap.add_argument('--checkpoint', required=True, type=pathlib.Path)
    ap.add_argument('--log-std', required=True, type=float, nargs=4)
    ap.add_argument('--out', required=True, type=pathlib.Path)
    args = ap.parse_args()
    model_sha, parent_sha = sha(args.model), sha(args.checkpoint)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if state.get('weights_sha256') != model_sha:
        raise ValueError('Parent weights SHA differs')
    model = json.loads(args.model.read_text())
    result, state = fork(model, state, args.log_std)
    args.out.mkdir(exist_ok=False)
    weights = args.out / 'weights.json'
    weights.write_text(json.dumps(result, allow_nan=False))
    state['weights_sha256'] = sha(weights)
    torch.save(state, args.out / 'checkpoint.pt')
    report = dict(version='combat_exploration_fork_v1', parent_weights_sha256=model_sha,
                  parent_checkpoint_sha256=parent_sha, previous_log_std=model['log_std'],
                  log_std=args.log_std, weights_sha256=sha(weights),
                  updates_completed=state['updates_completed'], total_actor_steps=state['total_actor_steps'],
                  fork_script_sha256=sha(pathlib.Path(__file__)),
                  scope='Explicit experimental fork, not an optimizer update. Actor/value weights unchanged; actor Adam reset; value Adam/RNG/consumed rollouts retained. Only fresh on-policy rollout may train this distribution.')
    if sha(args.model) != model_sha or sha(args.checkpoint) != parent_sha:
        raise ValueError('Parent inputs changed')
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

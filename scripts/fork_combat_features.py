"""Append observed-origin angular features with zero input weights; no imitation."""
import argparse
import copy
import json
import pathlib
import sys
sys.pycache_prefix = str(pathlib.Path(__file__).resolve().parents[1] / 'workspace/build/python-cache')
from fork_combat_exploration import fork
from ppo_combat import network, torch, sha


def extend(model, checkpoint):
    migrations={'combat_features_v1':('combat_features_v2',386,40),'combat_features_v2':('combat_features_v3',426,40),'combat_features_v3':('combat_features_v4',466,344)}
    if model.get('feature_version') not in migrations:
        raise ValueError('Only consecutive v1/v2/v3 feature migrations supported')
    version,inputs,added=migrations[model['feature_version']]
    # Validates all parent tensors/std. Both Adam resets are explicit because
    # parameter shapes change; history/objective/RNG/counters remain intact.
    child, state = fork(model, checkpoint, model['log_std'])
    child['feature_version'] = version
    for name in ('actor', 'value'):
        if any(len(row) != inputs for row in child[name][0]['weight']):
            raise ValueError('Unexpected parent input width')
        for row in child[name][0]['weight']:
            row.extend([0.] * added)
        state[name] = network(child[name]).state_dict()
    actor, value = network(child['actor']), network(child['value'])
    std = torch.nn.Parameter(torch.tensor(child['log_std']))
    state['actor_optimizer'] = torch.optim.Adam(list(actor.parameters())+[std], lr=state['config']['actor_lr']).state_dict()
    state['value_optimizer'] = torch.optim.Adam(value.parameters(), lr=state['config']['value_lr']).state_dict()
    return child, state


def main():
    ap = argparse.ArgumentParser()
    for name in ('model', 'checkpoint', 'probe-rollout', 'out'):
        ap.add_argument('--'+name, type=pathlib.Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    inputs = {str(p): sha(p) for p in (a.model,a.checkpoint,a.probe_rollout)}
    model = json.loads(a.model.read_text())
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    if state.get('weights_sha256') != sha(a.model):
        raise ValueError('Checkpoint differs from parent weights')
    child, state = extend(model,state)
    rows = [json.loads(s) for s in a.probe_rollout.read_text().splitlines() if s]
    if not rows:
        raise ValueError('Nonempty probe required')
    x = torch.tensor([r['features'] for r in rows], dtype=torch.float32)
    # Nonzero arbitrary additions prove they cannot affect initial outputs.
    added=len(child['actor'][0]['weight'][0])-len(x[0])
    extra = torch.linspace(-1,1,len(rows)*added).reshape(len(rows),added)
    errors = {}
    with torch.no_grad():
        for name in ('actor','value'):
            errors[name] = float((network(model[name])(x)-network(child[name])(torch.cat((x,extra),1))).abs().max())
            if errors[name]>1e-5:
                raise ValueError('Initial output parity failed')
    a.out.mkdir(exist_ok=False)
    weights = a.out/'weights.json'
    weights.write_text(json.dumps(child,allow_nan=False))
    state['weights_sha256'] = sha(weights)
    torch.save(state,a.out/'checkpoint.pt')
    if any(sha(pathlib.Path(p))!=h for p,h in inputs.items()):
        raise ValueError('Parent inputs changed')
    report = dict(version='combat_feature_fork_v1',source_sha256=inputs,
                  weights_sha256=sha(weights),max_output_error=errors,probe_rows=len(rows),
                  inputs_before=len(x[0]),inputs_after=len(x[0])+added,
                  scope='Zero appended weights preserve initial actor/value; both Adam states reset. Objective/std/RNG/consumed history/counters preserved. Only fresh on-policy training; no live target correction.')
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()

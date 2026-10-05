"""Offline function-preserving widening; Go still owns all game decisions."""
import argparse
import copy
import json
import pathlib
import sys
sys.pycache_prefix = str(pathlib.Path(__file__).resolve().parents[1] / 'workspace/build/python-cache')
from fork_combat_exploration import fork
from ppo_combat import network, layers, torch, sha


def widen(source, width, generator):
    net = network(source)
    if len(source) != 3 or any(len(l['bias']) > width for l in source[:2]):
        raise ValueError('Widening requires three layers and cannot shrink hidden layers')
    for index in (0, 2):
        old, following = net[index], net[index + 2]
        size = old.out_features
        if size == width:
            continue
        # Every original unit survives. Duplicate incoming rows and distribute
        # each outgoing column among its copies with positive weights summing
        # to one. Unequal shares allow subsequent gradients to break symmetry.
        mapping = torch.cat((torch.arange(size), torch.randint(size, (width-size,), generator=generator)))
        shares = torch.rand(width, generator=generator, dtype=torch.float64) + .5
        totals = torch.zeros(size, dtype=torch.float64).scatter_add_(0, mapping, shares)
        shares = (shares / totals[mapping]).float()
        expanded = torch.nn.Linear(old.in_features, width)
        outgoing = torch.nn.Linear(width, following.out_features)
        with torch.no_grad():
            expanded.weight.copy_(old.weight[mapping]); expanded.bias.copy_(old.bias[mapping])
            outgoing.weight.copy_(following.weight[:, mapping] * shares)
            outgoing.bias.copy_(following.bias)
        net[index], net[index + 2] = expanded, outgoing
    return layers(net)


def resize(model, checkpoint, width, seed):
    if width not in (64, 128, 256):
        raise ValueError('Supported experiment widths: 64/128/256')
    child, state = fork(model, checkpoint, model['log_std'])
    generator = torch.Generator().manual_seed(seed)
    for name in ('actor', 'value'):
        child[name] = widen(child[name], width, generator)
        state[name] = network(child[name]).state_dict()
    actor, value = network(child['actor']), network(child['value'])
    std = torch.nn.Parameter(torch.tensor(child['log_std']))
    # Reset both optimizers for ALL arms, including the unchanged-width control.
    state['actor_optimizer'] = torch.optim.Adam(list(actor.parameters()) + [std], lr=state['config']['actor_lr']).state_dict()
    state['value_optimizer'] = torch.optim.Adam(value.parameters(), lr=state['config']['value_lr']).state_dict()
    return child, state


def main():
    ap = argparse.ArgumentParser()
    for key in ('model', 'checkpoint', 'probe-rollout', 'out'):
        ap.add_argument('--'+key, required=True, type=pathlib.Path)
    ap.add_argument('--width', required=True, type=int)
    ap.add_argument('--seed', type=int, default=20261005)
    a = ap.parse_args()
    torch.set_num_threads(2)
    inputs = {str(p): sha(p) for p in (a.model, a.checkpoint, a.probe_rollout)}
    model = json.loads(a.model.read_text())
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('weights_sha256') != inputs[str(a.model)]:
        raise ValueError('Checkpoint belongs to different parent weights')
    child, state = resize(model, checkpoint, a.width, a.seed)
    rows = [json.loads(line) for line in a.probe_rollout.read_text().splitlines() if line]
    x = torch.tensor([r['features'] for r in rows], dtype=torch.float32)
    errors = {}
    with torch.no_grad():
        for name in ('actor', 'value'):
            errors[name] = float((network(model[name])(x)-network(child[name])(x)).abs().max())
            if errors[name] > 1e-5:
                raise ValueError('Widening parity failed: '+str(errors))
    a.out.mkdir(exist_ok=False)
    weights = a.out/'weights.json'
    weights.write_text(json.dumps(child, allow_nan=False))
    state['weights_sha256'] = sha(weights)
    torch.save(state, a.out/'checkpoint.pt')
    counts = {name: sum(p.numel() for p in network(child[name]).parameters()) for name in ('actor', 'value')}
    report = dict(version='combat_capacity_fork_v1', width=a.width, seed=a.seed,
                  source_sha256=inputs, weights_sha256=sha(weights), parameters=counts,
                  total_parameters=sum(counts.values())+4, probe_rows=len(rows), max_output_error=errors,
                  scope='Function-preserving widening with unequal positive outgoing shares; both Adam states reset in every arm. Objective/log_std/RNG/history/counters preserved. Probe parity is finite coverage, not gameplay acceptance.')
    if any(sha(pathlib.Path(p)) != digest for p, digest in inputs.items()):
        raise ValueError('Parent inputs changed')
    (a.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

"""Offline geometric aim-head warm-start; never corrects a live command."""
import argparse
import json
import math
import pathlib
import sys
import time
sys.pycache_prefix = str(pathlib.Path(__file__).resolve().parents[1] / 'workspace/build/python-cache')
from fork_combat_exploration import fork
from ppo_combat import network, layers, torch, sha, training_devices


def aim_labels(rows, limit=20.):
    if not math.isfinite(limit) or not 0 < limit <= 45:
        raise ValueError('Aim step limit must be in (0,45] degrees')
    selected, targets = [], []
    for row in rows:
        features = row['features']
        if len(features) != 810 or not all(math.isfinite(x) for x in features):
            raise ValueError('Finite typed v4 features required')
        for slot in range(8):
            offset = 426 + 5*slot
            if features[offset] != 1:
                continue
            angles = []
            for sine, cosine in ((features[offset+1], features[offset+2]), (features[offset+3], features[offset+4])):
                if abs(math.hypot(sine, cosine)-1) > 1e-5:
                    raise ValueError('Invalid observed angular feature')
                error = math.degrees(math.atan2(sine, cosine))
                angles.append(max(-limit, min(limit, error)))
            selected.append(features)
            targets.append(angles)
            break
    if len(selected) < 2:
        raise ValueError('At least two observed bbox targets required')
    return torch.tensor(selected, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


def finish_actor(child, state, actor):
    child['actor'] = layers(actor)
    state['actor'] = actor.state_dict()
    critic = network(child['value'])
    with torch.no_grad():
        critic[-1].weight.zero_()
        critic[-1].bias.zero_()
    child['value'] = layers(critic)
    state['value'] = critic.state_dict()
    state['value_optimizer']['state'] = {}


def fit_joint(child, state, x, target, epochs, limit, distill_weight):
    parent = network(child['actor'])
    with torch.no_grad(): old = parent(x)
    other = [0, 1, 4, 5, 6, 7]
    desired = torch.atanh(target/180)
    benchmark = {}

    def setup(device):
        actor = network(child['actor']).to(device)
        optimizer = torch.optim.Adam(actor.parameters(), lr=.0003)
        inputs, labels, teacher = x.to(device), desired.to(device), old.to(device)
        def step():
            optimizer.zero_grad()
            raw = actor(inputs)
            loss = (raw[:, 2:4]-labels).square().mean() + distill_weight*(raw[:, other]-teacher[:, other]).square().mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite joint aim fit')
            loss.backward()
            optimizer.step()
        return actor, step

    for device in training_devices():
        actor, step = setup(device)
        for _ in range(10): step()
        if device == 'cuda': torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50): step()
        if device == 'cuda': torch.cuda.synchronize()
        benchmark[device] = (time.perf_counter()-start)/50
    device = min(benchmark, key=benchmark.get)
    actor, step = setup(device)
    for _ in range(epochs): step()
    actor = actor.cpu()
    with torch.no_grad():
        updated = actor(x)
        movement_mae = (updated[:, :2].tanh()-old[:, :2].tanh()).abs().mean(0)
        attack_mae = (updated[:, 4].sigmoid()-old[:, 4].sigmoid()).abs().mean()
        teacher_vertical = old[:, 5:].log_softmax(1)
        vertical_kl = (teacher_vertical.exp()*(teacher_vertical-updated[:, 5:].log_softmax(1))).sum(1).mean()
        if float(movement_mae.max()) > .05 or float(attack_mae) > .05 or float(vertical_kl) > .02:
            raise ValueError('Joint fit failed training-state non-aim retention limits')
        report = dict(rows=len(x), epochs=epochs, step_limit_degrees=limit, mode='joint', distill_weight=distill_weight,
                      train_command_mae_before_degrees=(old[:, 2:4].tanh()*180-target).abs().mean(0).tolist(),
                      train_command_mae_after_degrees=(updated[:, 2:4].tanh()*180-target).abs().mean(0).tolist(),
                      non_aim_raw_max_error=float((updated[:, other]-old[:, other]).abs().max()),
                      movement_command_mae=movement_mae.tolist(), attack_probability_mae=float(attack_mae),
                      vertical_kl=float(vertical_kl), device=device, benchmark_seconds_per_step=benchmark,
                      retention_scope='Training observations only; not preserved closed-loop behavior')
    finish_actor(child, state, actor)
    return child, state, report


def fit(model, checkpoint, rows, epochs=1500, limit=20., mode='head', distill_weight=1.):
    if model.get('feature_version') != 'combat_features_v4' or not 1 <= epochs <= 10000 or mode not in ('head', 'joint'):
        raise ValueError('Typed v4 model and bounded fixed epochs required')
    if not math.isfinite(distill_weight) or not 0 < distill_weight <= 100:
        raise ValueError('Finite positive bounded distillation weight required')
    child, state = fork(model, checkpoint, model['log_std'])
    x, target = aim_labels(rows, limit)
    if mode == 'joint': return fit_joint(child, state, x, target, epochs, limit, distill_weight)
    actor = network(child['actor'])
    with torch.no_grad():
        hidden = actor[:-1](x)
        old = actor(x)
    width = hidden.shape[1]

    def head_on(device):
        head = torch.nn.Linear(width, 2).to(device)
        with torch.no_grad():
            head.weight.copy_(actor[-1].weight[2:4].to(device))
            head.bias.copy_(actor[-1].bias[2:4].to(device))
        return head

    latent_target = torch.atanh(target/180)
    benchmark = {}
    for device in training_devices():
        head = head_on(device)
        h, y = hidden.to(device), latent_target.to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=.003)
        def step():
            optimizer.zero_grad()
            loss = (head(h)-y).square().mean()
            loss.backward()
            optimizer.step()
        for _ in range(10): step()
        if device == 'cuda': torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50): step()
        if device == 'cuda': torch.cuda.synchronize()
        benchmark[device] = (time.perf_counter()-start)/50
    device = min(benchmark, key=benchmark.get)
    head = head_on(device)
    h, y = hidden.to(device), latent_target.to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=.003)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = (head(h)-y).square().mean()
        if not torch.isfinite(loss): raise ValueError('Nonfinite aim fit')
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        actor[-1].weight[2:4].copy_(head.weight.cpu())
        actor[-1].bias[2:4].copy_(head.bias.cpu())
        updated = actor(x)
        unchanged = [0, 1, 4, 5, 6, 7]
        error = float((updated[:, unchanged]-old[:, unchanged]).abs().max())
        if error != 0: raise ValueError('Non-aim outputs changed')
        before = (old[:, 2:4].tanh()*180-target).abs().mean(0).tolist()
        after = (updated[:, 2:4].tanh()*180-target).abs().mean(0).tolist()
    finish_actor(child, state, actor)
    report = dict(rows=len(x), epochs=epochs, step_limit_degrees=limit,
                  train_command_mae_before_degrees=before, train_command_mae_after_degrees=after,
                  non_aim_raw_max_error=error, device=device, benchmark_seconds_per_step=benchmark)
    return child, state, report


def main():
    ap = argparse.ArgumentParser()
    for name in ('model', 'checkpoint', 'out'): ap.add_argument('--'+name, type=pathlib.Path, required=True)
    ap.add_argument('--rollout', type=pathlib.Path, action='append', required=True)
    ap.add_argument('--epochs', type=int, default=1500)
    ap.add_argument('--limit', type=float, default=20.)
    ap.add_argument('--mode', choices=('head', 'joint'), default='head')
    ap.add_argument('--distill-weight', type=float, default=1.)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20261006)
    torch.use_deterministic_algorithms(True)
    inputs = {str(p): sha(p) for p in [a.model, a.checkpoint]}
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    if state.get('weights_sha256') != sha(a.model): raise ValueError('Checkpoint differs from parent weights')
    model = json.loads(a.model.read_text())
    rows = []
    seeds = set()
    seen = set()
    for path in a.rollout:
        meta_path = path/'report.json'
        meta = json.loads(meta_path.read_text())
        data = path/'rollout.jsonl'
        if meta['version'] != 'combat_ppo_rollout_v1' or meta['feature_version'] != 'combat_features_v4' or sha(data) != meta['rollout_sha256']:
            raise ValueError('Verified typed rollout required')
        if meta['rollout_sha256'] in seen: raise ValueError('Duplicate rollout')
        seen.add(meta['rollout_sha256'])
        for name, digest in meta['source_sha256'].items():
            if sha(pathlib.Path(name)) != digest: raise ValueError('Changed rollout proof')
        inputs[str(meta_path)], inputs[str(data)] = sha(meta_path), sha(data)
        batch = [json.loads(line) for line in data.read_text().splitlines()]
        if len(batch) != meta['rows']: raise ValueError('Rollout row count differs')
        rows.extend(batch)
        seeds.update(row['seed'] for row in batch)
    child, state, report = fit(model, state, rows, a.epochs, a.limit, a.mode, a.distill_weight)
    a.out.mkdir(exist_ok=False)
    weights = a.out/'weights.json'
    weights.write_text(json.dumps(child, allow_nan=False))
    state['weights_sha256'] = sha(weights)
    torch.save(state, a.out/'checkpoint.pt')
    if any(sha(pathlib.Path(p)) != digest for p, digest in inputs.items()): raise ValueError('Parent input changed')
    report.update(version='combat_aim_head_fork_v1' if a.mode=='head' else 'combat_aim_joint_fork_v1', source_sha256=inputs, training_seeds=sorted(seeds),
                  weights_sha256=sha(weights), script_sha256=sha(pathlib.Path(__file__)),
                  scope=('Offline geometric supervision on observed bbox angular errors, not PPO or human demonstrations. '
                         + ('Only final actor rows2/3 fitted; encoder and other actor rows/std unchanged. ' if a.mode=='head' else 'Actor encoder and heads fitted jointly, other outputs distilled from parent on training observations; std unchanged. ')
                         + 'Critic output zero; both Adam states reset. RNG/consumed history/counters retained; old PPO rows cannot be reused on this policy. Training fit is not held-out combat acceptance.'))
    (a.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()

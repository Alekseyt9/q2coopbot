"""Offline observed-geometry curriculum; runtime commands remain learned in Go."""
import argparse
import json
import math
import pathlib
import sys
import time
sys.pycache_prefix = str(pathlib.Path(__file__).resolve().parents[1] / 'workspace/build/python-cache')
from fork_combat_exploration import fork
from fork_combat_aim import finish_actor
from ppo_combat import network, torch, sha


def movement_label(f, raw):
    """v4 layout: eight current-view probes, nearest enemies, four observed props.

    Return None when grounded geometry is unavailable. Probe reach is only 64
    units: these labels are local curriculum hints, not a guaranteed safe path.
    Current-view left axis is opposite Quake's positive sidemove. Compensate
    the parent's simultaneous yaw/pitch command before constructing commands.
    """
    if len(f) != 810 or any(not math.isfinite(v) for v in f):
        raise ValueError('Finite typed v4 features required')
    if len(raw) != 8 or any(not math.isfinite(v) for v in raw):
        raise ValueError('Finite eight-output parent required')
    if f[2] != 1 or f[21] != 1:
        return None
    repel = [0., 0.]
    for slot in range(8):
        at = 73 + 12*slot
        if f[at] != 1 or f[at+9] != 1:  # Visible Parasite only.
            continue
        x, y = f[at+2]*512, f[at+3]*512
        distance = math.hypot(x, y)
        if 0 < distance < 320:
            weight = 2*(1-distance/320)
            repel[0] -= weight*x/distance
            repel[1] -= weight*y/distance
    barrels = []
    if f[259] == 1:  # Props list known, after projectiles/teammate/pickups.
        for slot in range(4):
            at = 260 + 9*slot
            if f[at] != 1 or f[at+5] != 1:
                continue
            x, y = f[at+2]*512, f[at+3]*512
            distance = math.hypot(x, y)
            barrels.append((x, y))
            if 0 < distance < 384:
                weight = 3*(1-distance/384)
                repel[0] -= weight*x/distance
                repel[1] -= weight*y/distance
    candidates = []
    for direction in range(8):
        at = 25 + 6*direction
        if f[at] != 1 or f[at+2] != 1 or f[at+3]*64 < 40 or f[at+4] != 1:
            continue
        angle = direction*math.pi/4
        x, y = math.cos(angle), math.sin(angle)
        # Avoid stepping through the observed padded barrel's horizontal box.
        if any(abs(bx-40*x) < 48 and abs(by-40*y) < 48 for bx, by in barrels):
            continue
        score = x*repel[0]+y*repel[1] + .25*f[at+3] + .1*f[at+1]
        candidates.append((score, direction))
    if not candidates:
        return None
    _, direction = max(candidates)  # Stable, documented tie break.
    yaw_step = math.radians(180*math.tanh(raw[2]))
    angle = direction*math.pi/4-yaw_step
    pitch = max(-89., min(89., math.degrees(math.atan2(f[8], f[9]))+180*math.tanh(raw[3])))
    forward = .75*math.cos(angle)/math.cos(math.radians(pitch/3))
    side = -.75*math.sin(angle)
    return [max(-.95, min(.95, forward)), side]


def verified_rows(paths, receipts):
    rows, seen = [], set()
    for path in paths:
        report, data = path/'report.json', path/'rollout.jsonl'
        meta = json.loads(report.read_text())
        if meta['version'] != 'combat_ppo_rollout_v1' or meta['feature_version'] != 'combat_features_v4' or sha(data) != meta['rollout_sha256']:
            raise ValueError('Verified typed rollout required')
        if meta['rollout_sha256'] in seen:
            raise ValueError('Duplicate rollout')
        seen.add(meta['rollout_sha256'])
        for name, digest in meta['source_sha256'].items():
            if sha(pathlib.Path(name)) != digest:
                raise ValueError('Changed rollout proof')
        batch = [json.loads(line) for line in data.read_text().splitlines()]
        if len(batch) != meta['rows']:
            raise ValueError('Rollout row count differs')
        receipts[str(report)], receipts[str(data)] = sha(report), sha(data)
        rows.extend(batch)
    return rows


def fit(model, checkpoint, rows, retain_rows, epochs=1500):
    if model.get('feature_version') != 'combat_features_v4' or not 1 <= epochs <= 10000:
        raise ValueError('Typed model and bounded epochs required')
    child, state = fork(model, checkpoint, model['log_std'])
    all_rows = rows+retain_rows
    x = torch.tensor([r['features'] for r in all_rows], dtype=torch.float32)
    if x.ndim != 2 or x.shape[1] != 810 or not torch.isfinite(x).all():
        raise ValueError('Finite typed observations required')
    with torch.no_grad(): old = network(model['actor'])(x)
    target = old[:, :2].tanh().clone()
    changed = []
    for i, row in enumerate(rows):
        if row['features'][808]*8 < 2:
            continue  # Distill parent movement when only one enemy is visible.
        label = movement_label(row['features'], old[i].tolist())
        if label is not None:
            target[i] = torch.tensor(label)
            changed.append(i)
    if len(changed) < 2 or len(retain_rows) < 2:
        raise ValueError('Mixed geometry labels and solo retention observations required')
    desired = torch.atanh(target)
    benchmark = {}
    def setup(device):
        actor = network(child['actor']).to(device)
        optimizer = torch.optim.Adam(actor.parameters(), lr=.0003)
        inputs, labels, teacher = x.to(device), desired.to(device), old.to(device)
        def step():
            optimizer.zero_grad()
            raw = actor(inputs)
            loss = (raw[:, :2]-labels).square().mean() + 10*(raw[:, 2:]-teacher[:, 2:]).square().mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite maneuver fit')
            loss.backward()
            optimizer.step()
        return actor, step
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
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
        new = actor(x)
        solo_error = (new[len(rows):, :2].tanh()-old[len(rows):, :2].tanh()).abs().mean(0)
        aim_error = (new[:, 2:4].tanh()-old[:, 2:4].tanh()).abs().mean(0)*180
        attack_error = (new[:, 4].sigmoid()-old[:, 4].sigmoid()).abs().mean()
        log_old, log_new = old[:, 5:].log_softmax(1), new[:, 5:].log_softmax(1)
        vertical_kl = (log_old.exp()*(log_old-log_new)).sum(1).mean()
        if solo_error.max() > .05 or aim_error.max() > 2 or attack_error > .05 or vertical_kl > .02:
            raise ValueError('Training-state command retention gate failed')
        report = dict(rows=len(rows), retention_rows=len(retain_rows), labels=len(changed), epochs=epochs,
                      movement_mae_before=(old[changed, :2].tanh()-target[changed]).abs().mean(0).tolist(),
                      movement_mae_after=(new[changed, :2].tanh()-target[changed]).abs().mean(0).tolist(),
                      solo_movement_mae=solo_error.tolist(), aim_command_mae_degrees=aim_error.tolist(),
                      attack_probability_mae=float(attack_error), vertical_kl=float(vertical_kl),
                      device=device, benchmark_seconds_per_step=benchmark)
    finish_actor(child, state, actor)
    return child, state, report


def main():
    ap = argparse.ArgumentParser()
    for name in ('model', 'checkpoint', 'out'):
        ap.add_argument('--'+name, type=pathlib.Path, required=True)
    for name in ('rollout', 'retain-rollout'):
        ap.add_argument('--'+name, type=pathlib.Path, action='append', required=True)
    ap.add_argument('--epochs', type=int, default=1500)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20261006)
    torch.use_deterministic_algorithms(True)
    receipts = {str(p): sha(p) for p in (a.model, a.checkpoint)}
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    if state.get('weights_sha256') != sha(a.model): raise ValueError('Parent checkpoint differs')
    rows = verified_rows(a.rollout, receipts)
    retain = verified_rows(a.retain_rollout, receipts)
    if {r['seed'] for r in rows} & {r['seed'] for r in retain}:
        raise ValueError('Mixed and solo seeds must be disjoint')
    child, state, report = fit(json.loads(a.model.read_text()), state, rows, retain, a.epochs)
    a.out.mkdir(exist_ok=False)
    weights = a.out/'weights.json'
    weights.write_text(json.dumps(child, allow_nan=False))
    state['weights_sha256'] = sha(weights)
    torch.save(state, a.out/'checkpoint.pt')
    if any(sha(pathlib.Path(p)) != digest for p, digest in receipts.items()):
        raise ValueError('Input changed')
    report.update(version='combat_maneuver_fork_v1', source_sha256=receipts,
                  training_seeds=sorted({r['seed'] for r in rows}), retention_seeds=sorted({r['seed'] for r in retain}),
                  weights_sha256=sha(weights), script_sha256=sha(pathlib.Path(__file__)),
                  scope='Offline local geometry supervision, not PPO or demonstrations. Visible mixed groups only; 64-unit probes and observed barrels, no hidden state. Parent aim/fire/vertical and solo movement distilled on training states. No live correction. Both Adam states reset, critic output zero, RNG/history/counters/std retained. Only fresh rollout may train PPO; closed-loop retention requires held-out evaluation.')
    (a.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()

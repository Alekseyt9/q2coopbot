"""CUDA pilot: structured H-JEPA adaptation with BC proposals, not PPO.

Three chained encoders predict at 1/4/16 frames in distinct latent spaces.
VICReg-style variance/covariance and inverse dynamics protect useful variation;
this intentionally differs from the paper's visual encoder and SIGReg.
"""
import argparse
import hashlib
import json
import math
import pathlib
import time

import torch
from torch import nn
from torch.nn import functional as F

STRIDES = (1, 4, 16)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def network(n, out, hidden=64):
    return nn.Sequential(nn.Linear(n, hidden), nn.ReLU(), nn.Linear(hidden, out))


def export(net):
    return [dict(weight=m.weight.detach().cpu().double().tolist(),
                 bias=m.bias.detach().cpu().double().tolist())
            for m in net if isinstance(m, nn.Linear)]


def serialized_forward(x, layers):
    for i, layer in enumerate(layers):
        x=F.linear(x,torch.tensor(layer['weight'],dtype=x.dtype),torch.tensor(layer['bias'],dtype=x.dtype))
        if i<len(layers)-1:x=F.relu(x)
    return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([network(845 if i == 0 else 32, 32) for i in range(3)])
        self.predictors = nn.ModuleList([network(32 + 8 * s, 32) for s in STRIDES])
        self.utilities = nn.ModuleList([network(32, 1, 32) for _ in STRIDES])
        self.actor = network(32, 8)
        self.inverse = network(64, 8)

    def encode(self, x):
        zs = []
        for encoder in self.encoders:
            x = encoder(x)
            zs.append(x)
        return zs


def load(path, device):
    meta = json.loads(path.with_name(path.name + '.meta.json').read_text())
    if meta['version'] != 'hjepa_data_v1' or meta['feature_version'] != 'combat_features_v6':
        raise ValueError('unsupported dataset')
    if meta.get('data_sha256') and sha(path) != meta['data_sha256']:
        raise ValueError('exported data changed')
    # Bind every retained source; do not silently train after captures change.
    for source, digest in meta['sources_sha256'].items():
        if sha(pathlib.Path(source)) != digest:
            raise ValueError('native source changed: ' + source)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != meta['rows']:
        raise ValueError('row count changed')
    groups = {}
    for r in rows:
        if len(r['features']) != 845 or len(r['next_features']) != 845 or len(r['action']) != 8:
            raise ValueError('feature/action width')
        groups.setdefault(r['episode'], []).append(r)
    clips = []
    for group in groups.values():
        group.sort(key=lambda r: r['frame'])
        if len({r['frame'] for r in group}) != len(group):
            raise ValueError('duplicate episode/frame')
        for start in range(len(group) - 15):
            chunk = group[start:start + 16]
            if any(b['frame'] != a['frame'] + 1 or a['next_features'] != b['features']
                   for a, b in zip(chunk, chunk[1:])):
                continue
            clips.append(chunk)
    if not clips:
        raise ValueError('no contiguous 16-frame clips')
    x = torch.tensor([[r['features'] for r in c] + [c[-1]['next_features']]
                      for c in clips], device=device)
    a = torch.tensor([[r['action'] for r in c] for c in clips], device=device)
    terminal_targets = all('next_utility' in r for r in rows)
    u = torch.tensor([[r['utility'] for r in c]+([c[-1]['next_utility']] if terminal_targets else []) for c in clips], device=device)
    behavior = torch.tensor([[r.get('behavior_weight', 1.) for r in c] for c in clips], device=device)
    if not all(torch.isfinite(v).all() for v in (x, a, u)):
        raise ValueError('nonfinite data')
    if (a[..., :4].abs() > 1).any() or not ((a[..., 4:] == 0) | (a[..., 4:] == 1)).all() or not (a[..., 5:].sum(-1) == 1).all():
        raise ValueError('invalid primitive')
    return (x, a, u, behavior), set(groups), rows


def action_loss(raw, a):
    # Degree-sized errors otherwise disappear next to binary fire loss.
    return ((raw[..., :2].tanh() - a[..., :2]).square().mean()
            + 200 * (raw[..., 2:4].tanh() - a[..., 2:4]).square().mean()
            + F.binary_cross_entropy_with_logits(raw[..., 4], a[..., 4])
            + F.cross_entropy(raw[..., 5:].reshape(-1, 3), a[..., 5:].argmax(-1).reshape(-1)))


def losses(model, data):
    x, a, u, behavior = data
    zs = model.encode(x)
    prediction, regularizer, utility = x.new_zeros(()), x.new_zeros(()), x.new_zeros(())
    for i, stride in enumerate(STRIDES):
        z = zs[i]
        actions = a[:, :stride].flatten(1)
        pred = model.predictors[i](torch.cat((z[:, 0], actions), -1))
        prediction = prediction + F.mse_loss(pred, z[:, stride].detach())
        flat = z.reshape(-1, 32)
        regularizer = regularizer + F.relu(1 - (flat.var(0, unbiased=False) + 1e-4).sqrt()).mean()
        centered = flat - flat.mean(0)
        cov = centered.T @ centered / max(1, len(flat) - 1)
        regularizer = regularizer + (cov.square().sum() - cov.diag().square().sum()) / 32
        utility = utility + F.mse_loss(model.utilities[i](z[:, :u.shape[1]]).squeeze(-1), u)
    mask = behavior > 0
    bc = action_loss(model.actor(zs[0][:, :16])[mask], a[mask]) if mask.any() else x.new_zeros(())
    inverse = action_loss(model.inverse(torch.cat((zs[0][:, :-1], zs[0][:, 1:]), -1)), a)
    total = prediction + .2 * regularizer + utility + bc + .2 * inverse
    return total, dict(prediction_mse=float(prediction.detach()), anti_collapse=float(regularizer.detach()),
                       utility_mse=float(utility.detach()), behavior_loss=float(bc.detach()),
                       inverse_loss=float(inverse.detach()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', type=pathlib.Path, required=True)
    ap.add_argument('--validation', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--seed', type=int, default=251115)
    ap.add_argument('--motor-weights',type=pathlib.Path,help='preserve an independently trained direct motor exactly')
    args = ap.parse_args()
    if not 1 <= args.epochs <= 2000:
        raise ValueError('epochs outside pilot budget')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no silent CPU training')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train, train_ids, rows = load(args.train, 'cuda')
    val, val_ids, _ = load(args.validation, 'cuda')
    if train_ids & val_ids:
        raise ValueError('train/validation episode leakage')
    # Compare declared seed cohorts independently of connection/spawn identities.
    seed_ids = lambda ids: {i.split(':', 1)[0].split('@',1)[0] for i in ids}
    if seed_ids(train_ids) & seed_ids(val_ids):
        raise ValueError('train/validation seed leakage')
    model = Model().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    with torch.no_grad():
        before = {'train': losses(model, train)[1], 'validation': losses(model, val)[1]}
    history = []
    start = time.perf_counter()
    # Fixed epoch budget. Validation never selects a checkpoint or changes it.
    for epoch in range(args.epochs):
        order = torch.randperm(len(train[0]), device='cuda')
        for batch in order.split(128):
            opt.zero_grad()
            loss, _ = losses(model, tuple(t[batch] for t in train))
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
        if epoch % 25 == 0 or epoch + 1 == args.epochs:
            with torch.no_grad():
                entry = dict(epoch=epoch + 1, train=losses(model, train)[1], validation=losses(model, val)[1])
            history.append(entry)
            print(json.dumps(entry), flush=True)
    model = model.cpu().double()
    weights = dict(kind='combat_hjepa_v1', feature_version='combat_features_v6', prior_weight=.25,
                   actor=export(model.actor), levels=[dict(stride=s, encoder=export(model.encoders[i]),
                   predictor=export(model.predictors[i]), utility=export(model.utilities[i]))
                   for i, s in enumerate(STRIDES)])
    save(args.out / 'weights.json', weights)
    if args.motor_weights:
        source_motor=json.loads(args.motor_weights.read_text())
        if not source_motor.get('motor'):raise ValueError('missing frozen direct motor')
        weights['motor']=source_motor['motor']
        save(args.out/'weights.json',weights)
    torch.save(dict(version='hjepa_checkpoint_v1', model=model.state_dict(), optimizer=opt.state_dict(),
                    epochs=args.epochs, seed=args.seed, rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all()), args.out / 'checkpoint.pt')
    # Export numerical fixtures for independent native-Go layer parity.
    fixture_x = torch.tensor([rows[i]['features'] for i in range(min(8, len(rows)))], dtype=torch.float64)
    with torch.no_grad():
        z = model.encode(fixture_x)
        raw = model.actor(z[0])
        if weights.get('motor'):raw=serialized_forward(fixture_x,weights['motor'])
        a = torch.tensor([rows[i]['action'] for i in range(len(fixture_x))], dtype=torch.float64)
        predictions = [model.predictors[i](torch.cat((z[i], a.repeat(1, s)), -1)).tolist()
                       for i, s in enumerate(STRIDES)]
    save(args.out / 'parity.json', dict(features=fixture_x.tolist(), latents=[v.tolist() for v in z],
                                      actor=raw.tolist(), actions=a.tolist(), predictions=predictions))
    save(args.out / 'report.json', dict(version='hjepa_training_v1', seed=args.seed, epochs=args.epochs,
         train_clips=len(train[0]), validation_clips=len(val[0]), train_episodes=len(train_ids),
         validation_episodes=len(val_ids), before=before, history=history, seconds=time.perf_counter()-start,
         train_sha256=sha(args.train), validation_sha256=sha(args.validation),
         trainer_sha256=sha(pathlib.Path(__file__)), weights_sha256=sha(args.out/'weights.json'),
         checkpoint_sha256=sha(args.out/'checkpoint.pt'), promotion=False,
         objective_version=json.loads(pathlib.Path(str(args.train)+'.meta.json').read_text()).get('objective_version','visibility_proxy_v1'),
         motor_source_sha256=sha(args.motor_weights) if args.motor_weights else None,
         scope='Structured observation adaptation. VICReg-style variance/covariance plus inverse dynamics, BC proposal prior, observed utility auxiliary; not the visual paper reproduction or SIGReg. No PPO, weapon choice, campaign readiness or superiority claim. Fixed final epoch.'))


if __name__ == '__main__':
    main()

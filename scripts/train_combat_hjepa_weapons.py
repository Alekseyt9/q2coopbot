"""Distil a separate masked weapon head; keep H-JEPA world and motor frozen."""
import argparse
import collections
import json
import pathlib
import time

import torch
from torch import nn
from torch.nn import functional as F
from train_combat_hjepa import export, save, sha, serialized_forward


def load_rows(path, split):
    meta = json.loads(pathlib.Path(str(path) + '.meta.json').read_text())
    if meta['version'] != 'hjepa_weapon_data_v1' or meta['split'] != split:
        raise ValueError('wrong dataset version/split')
    for field in ['sources_sha256', 'native_receipts_sha256']:
        for name, digest in meta[field].items():
            if sha(pathlib.Path(name)) != digest:
                raise ValueError('changed native source/proof')
    if sha(pathlib.Path(meta['teacher_source'])) != meta['teacher_source_sha256']:
        raise ValueError('changed teacher')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows or len(rows) != meta['rows']:
        raise ValueError('row count mismatch')
    seen = set()
    for r in rows:
        key = (r['seed'], json.dumps(r['identity'], sort_keys=True))
        if key in seen:
            raise ValueError('duplicate identity')
        seen.add(key)
        if r['split'] != split or r['label_kind'] != 'hypothetical_u1_weapon_distribution_v1':
            raise ValueError('wrong label/split')
        if len(r['features']) != 845 or len(r['teacher_weapon_probabilities']) != 12:
            raise ValueError('wrong dimensions')
    return rows, meta


def tensors(rows, device):
    x = torch.tensor([r['features'] for r in rows], dtype=torch.float32, device=device)
    y = torch.tensor([r['teacher_weapon_probabilities'] for r in rows], dtype=torch.float32, device=device)
    mask = x[:, 833:845]
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('nonfinite data')
    if not ((mask == 0) | (mask == 1)).all() or not (mask[:, 0] == 1).all():
        raise ValueError('invalid availability')
    if (y < 0).any() or not torch.allclose(y.sum(-1), torch.ones_like(y[:, 0]), atol=1e-5):
        raise ValueError('invalid probability distribution')
    if (y[mask == 0] != 0).any():
        raise ValueError('unavailable teacher choice')
    return x, y, mask.bool()


def loss_metrics(net, data):
    x, y, mask = data
    logits = net(x).masked_fill(~mask, -1e9)
    logp = logits.log_softmax(-1)
    ce = -(y * logp).sum(-1).mean()
    entropy = -(y * y.clamp_min(1e-30).log()).sum(-1).mean()
    metrics = dict(kl=float((ce - entropy).detach()),
                   top1_agreement=float((logits.argmax(-1) == y.argmax(-1)).float().mean().detach()),
                   switch_fraction=float((logits.argmax(-1) != 0).float().mean().detach()))
    return ce, metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', type=pathlib.Path, required=True)
    ap.add_argument('--validation', type=pathlib.Path, required=True)
    ap.add_argument('--parent', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--seed', type=int, default=261116)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for neural training')
    if not 1 <= a.epochs <= 1000:
        raise ValueError('epoch budget')
    train_rows, tm = load_rows(a.train, 'train')
    val_rows, vm = load_rows(a.validation, 'validation')
    if {r['seed'] for r in train_rows} & {r['seed'] for r in val_rows}:
        raise ValueError('train/validation seed overlap')
    if tm['teacher_source_sha256'] != vm['teacher_source_sha256']:
        raise ValueError('different teachers')
    parent = json.loads(a.parent.read_text())
    if parent['kind'] != 'combat_hjepa_v1' or parent['feature_version'] != 'combat_features_v6':
        raise ValueError('incompatible parent')
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'trainer-frozen.py').write_bytes(pathlib.Path(__file__).read_bytes())
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    train, val = tensors(train_rows, 'cuda'), tensors(val_rows, 'cuda')
    net = nn.Sequential(nn.Linear(845, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 12)).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    history = []
    start = time.perf_counter()
    for epoch in range(a.epochs):
        net.train()
        for batch in torch.randperm(len(train_rows), device='cuda').split(128):
            opt.zero_grad()
            loss, _ = loss_metrics(net, tuple(t[batch] for t in train))
            if not torch.isfinite(loss):
                raise ValueError('nonfinite loss')
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1)
            opt.step()
        if epoch % 50 == 0 or epoch + 1 == a.epochs:
            with torch.no_grad():
                row = dict(epoch=epoch + 1, train=loss_metrics(net, train)[1], validation=loss_metrics(net, val)[1])
            history.append(row)
            print(json.dumps(row), flush=True)
    layers = export(net)
    model = dict(parent, weapon_mode='learned_masked_v1', weapon_head='combat_masked_weapon_v1', weapon_actor=layers,
                 weapon_sampling_seed=a.seed)
    for key in ['levels', 'motor', 'actor', 'planner', 'plan_axes', 'plan_step', 'consistency_margin', 'prior_weight']:
        if model.get(key) != parent.get(key):
            raise ValueError('frozen parent changed')
    save(a.out / 'model.json', model)
    # Serialized dense layers use double precision in Go. Compare against the
    # independently exported scalar representation before native replay.
    x = val[0][:64].cpu().double()
    cpu = net.cpu().double()
    with torch.no_grad():
        parity = float((cpu(x) - serialized_forward(x, layers)).abs().max())
    if parity > 1e-10:
        raise ValueError('serialized parity failure')
    save(a.out / 'parity-vectors.json', dict(features=x.tolist(), outputs=cpu(x).detach().tolist()))
    torch.save(dict(version='hjepa_weapon_checkpoint_v1', weapon_actor=net.state_dict(), optimizer=opt.state_dict(),
                    epochs=a.epochs, seed=a.seed, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()),
               a.out / 'checkpoint.pt')
    save(a.out / 'report.json', dict(version='hjepa_weapon_training_v1', promotion=False, train_rows=len(train_rows),
         validation_rows=len(val_rows), train_seeds=sorted({r['seed'] for r in train_rows}),
         validation_seeds=sorted({r['seed'] for r in val_rows}), parent_sha256=sha(a.parent),
         teacher_sha256=tm['teacher_source_sha256'], train_sha256=sha(a.train), validation_sha256=sha(a.validation),
         model_sha256=sha(a.out / 'model.json'), epochs=a.epochs, seed=a.seed, history=history,
         serialized_max_error=parity, seconds=time.perf_counter() - start,
         scope='Only weapon_actor trained on U1 hypothetical probabilities. World/motor frozen; '
               'no runtime teacher. Fixed epoch budget; validation never backpropagated. Not a combat quality claim.'))


if __name__ == '__main__':
    main()

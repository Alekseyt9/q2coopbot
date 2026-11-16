"""Calibrate only the last weapon layer on native initial-state preferences."""
import argparse
import copy
import json
import pathlib
import time
import torch
from torch import nn
from train_combat_hjepa import export, save, sha, serialized_forward
from train_combat_hjepa_weapons import load_rows, tensors, loss_metrics


def load_native(path, split):
    meta = json.loads(pathlib.Path(str(path) + '.meta.json').read_text())
    if meta['version'] != 'hjepa_native_weapon_preference_data_v1' or meta['split'] != split or not meta['diagnostic']:
        raise ValueError('wrong native preference contract/split')
    for field in ['sources_sha256', 'native_receipts_sha256']:
        for name, digest in meta[field].items():
            if sha(pathlib.Path(name)) != digest:
                raise ValueError('changed native source/receipt')
    for name, digest in meta.get('additional_inputs_sha256', {}).items():
        if sha(pathlib.Path(name)) != digest:
            raise ValueError('changed merged preference input')
    for source, digest in [('branch_states', 'branch_states_sha256'), ('completion', 'completion_sha256'), ('parent_source', 'parent_sha256')]:
        if sha(pathlib.Path(meta[source])) != meta[digest]:
            raise ValueError('changed preference input')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows or len(rows) != meta['rows'] or len({r['seed'] for r in rows}) != len(rows):
        raise ValueError('missing/duplicate native states')
    for r in rows:
        if r['split'] != split or r['label_kind'] != 'native_weapon_policy_preference_v1':
            raise ValueError('mixed labels/splits')
        r['teacher_weapon_probabilities'] = r['preference_probabilities']
    return rows


def network(layers):
    modules = []
    for i, layer in enumerate(layers):
        linear = nn.Linear(len(layer['weight'][0]), len(layer['bias']))
        with torch.no_grad():
            linear.weight.copy_(torch.tensor(layer['weight']))
            linear.bias.copy_(torch.tensor(layer['bias']))
        modules.append(linear)
        if i + 1 < len(layers):
            modules.append(nn.ReLU())
    return nn.Sequential(*modules)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', type=pathlib.Path, required=True)
    ap.add_argument('--validation', type=pathlib.Path, required=True)
    ap.add_argument('--retention', type=pathlib.Path, required=True)
    ap.add_argument('--parent', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--seed', type=int, default=261118)
    ap.add_argument('--full-head', action='store_true', help='train weapon head encoders as well; world/motor remain frozen')
    ap.add_argument('--weapon-context', action='store_true', help='append current weapon and own-request context to weapon head only')
    ap.add_argument('--supervised-retention', action='store_true', help='retain explicit offline training labels instead of parent predictions')
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    if not 1 <= a.epochs <= 1000:
        raise ValueError('epoch budget')
    tr = load_native(a.train, 'train')
    vr = load_native(a.validation, 'validation')
    tm = json.loads(pathlib.Path(str(a.train) + '.meta.json').read_text())
    vm = json.loads(pathlib.Path(str(a.validation) + '.meta.json').read_text())
    if tm.get('target_mode', 'soft_score') != vm.get('target_mode', 'soft_score'):
        raise ValueError('different native target contracts')
    rr, _ = load_rows(a.retention, 'train')
    if {r['seed'] for r in tr + rr} & {r['seed'] for r in vr}:
        raise ValueError('train/validation seed overlap')
    parent = json.loads(a.parent.read_text())
    if parent.get('weapon_mode') != 'learned_masked_v1' or parent.get('weapon_sampling_seed') is not None:
        raise ValueError('greedy learned parent required')
    if a.weapon_context and not a.full_head:
        raise ValueError('new zero-initialized context columns require full weapon-head fitting')
    initial_layers = copy.deepcopy(parent['weapon_actor'])
    if a.weapon_context:
        for rows in [tr, vr, rr]:
            for row in rows:
                context = row.get('weapon_features')
                if context is None or len(context) != 871 or context[:845] != row['features']:
                    raise ValueError('missing or inconsistent observed weapon context')
                row['features'] = context
        for weights in initial_layers[0]['weight']:
            if len(weights) != 845:
                raise ValueError('zero-extension requires a legacy weapon parent')
            weights.extend([0.] * 26)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'trainer-frozen.py').write_bytes(pathlib.Path(__file__).read_bytes())
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    train, val, retention = tensors(tr, 'cuda'), tensors(vr, 'cuda'), tensors(rr, 'cuda')
    net = network(initial_layers).cuda()
    # All ablations use identical retention batch streams despite different
    # temporary Linear constructor sizes; initialized values are overwritten.
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    for p in net.parameters():
        p.requires_grad_(a.full_head)
    for p in net[-1].parameters():
        p.requires_grad_(True)
    parameters = [p for p in net.parameters() if p.requires_grad]
    initial = [p.detach().clone() for p in parameters]
    with torch.no_grad():
        if not a.supervised_retention:
            retention = (retention[0], net(retention[0]).masked_fill(~retention[2], -1e9).softmax(-1), retention[2])
        before = dict(train=loss_metrics(net, train)[1], validation=loss_metrics(net, val)[1])
    opt = torch.optim.Adam(parameters, lr=1e-3)
    history = []
    start = time.perf_counter()
    for epoch in range(a.epochs):
        batch = torch.randperm(len(rr), device='cuda')[:128]
        opt.zero_grad()
        native_loss, _ = loss_metrics(net, train)
        retention_loss, _ = loss_metrics(net, tuple(t[batch] for t in retention))
        drift = sum((p - old).square().mean() for p, old in zip(parameters, initial))
        loss = native_loss + .25 * retention_loss + .01 * drift
        if not torch.isfinite(loss):
            raise ValueError('nonfinite calibration loss')
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 1.)
        opt.step()
        if epoch % 100 == 0 or epoch + 1 == a.epochs:
            with torch.no_grad():
                row = dict(epoch=epoch + 1, train=loss_metrics(net, train)[1], validation=loss_metrics(net, val)[1])
            history.append(row)
            print(json.dumps(row), flush=True)
    layers = export(net)
    if not a.full_head and layers[:-1] != parent['weapon_actor'][:-1]:
        raise ValueError('frozen weapon encoder changed')
    model = dict(parent, weapon_actor=layers)
    if a.weapon_context:
        model['weapon_feature_version'] = 'hjepa_weapon_features_v1'
    frozen = lambda d: {k: v for k, v in d.items() if k not in ['weapon_actor', 'weapon_feature_version']}
    if frozen(model) != frozen(parent):
        raise ValueError('non-weapon model field changed')
    save(a.out / 'model.json', model)
    cpu = net.cpu().double()
    x = val[0].cpu().double()
    with torch.no_grad():
        error = float((cpu(x) - serialized_forward(x, layers)).abs().max())
    if error > 1e-10:
        raise ValueError('serialized parity failure')
    save(a.out / 'parity-vectors.json', dict(features=x.tolist(), outputs=cpu(x).detach().tolist()))
    torch.save(dict(version='hjepa_native_weapon_calibration_v1', weapon_actor=net.state_dict(), optimizer=opt.state_dict(),
                    full_head=a.full_head, weapon_context=a.weapon_context, supervised_retention=a.supervised_retention, epochs=a.epochs, seed=a.seed, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), a.out / 'checkpoint.pt')
    save(a.out / 'report.json', dict(version='hjepa_native_weapon_calibration_v1', promotion=False,
        train_states=len(tr), validation_states=len(vr), retention_states=len(rr), epochs=a.epochs, seed=a.seed,
        train_sha256=sha(a.train), validation_sha256=sha(a.validation), retention_sha256=sha(a.retention),
        parent_sha256=sha(a.parent), model_sha256=sha(a.out / 'model.json'), serialized_max_error=error,
        trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),
        full_head=a.full_head, weapon_context=a.weapon_context, supervised_retention=a.supervised_retention, target_mode=tm.get('target_mode', 'soft_score'),
        before=before, history=history, seconds=time.perf_counter() - start,
        scope='Only weapon head parameters in the declared full_head scope trained on initial-state native policy contrasts with the declared supervised_retention target. '
              'No per-frame optimality claim, no world/motor updates, no fitting on validation/test/confirmation. '
              'Offline label agreement is not a native combat result.'))


if __name__ == '__main__':
    main()

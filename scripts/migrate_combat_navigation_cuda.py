"""Append navigation inputs without changing existing encoders; CUDA verification.

Creates weights only. Optimizer checkpoints are deliberately not migrated and
must not be treated as resumable by this tool.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from ppo_combat import torch, network
from ppo_recurrent import Recurrent
from combat_attention import CausalAttention
from combat_spatial_aim import wrap, validate as validate_spatial
from combat_precision_head import validate_precision_model
from combat_target_head import validate_model, availability


def validate(model):
    if model.get('spatial_aim'):
        validate_spatial(model)
    elif model.get('aim_mode_head'):
        validate_precision_model(model)
    else:
        validate_model(model)


def migrate(model):
    validate(model)
    if model['feature_version'] != 'combat_features_v7':
        raise ValueError('Expected unmigrated V7 model')
    result = copy.deepcopy(model)
    for name in ('actor', 'value'):
        for row in result[name][0]['weight']:
            if len(row) != 854:
                raise ValueError('Invalid source input width')
            row.extend([0.] * 27)
    result['feature_version'] = 'combat_features_v8'
    validate(result)
    return result


def branch(model, name):
    if model.get('memory'):
        net = Recurrent(model[name], model['memory'][name])
    elif model.get('attention'):
        spec = model['attention']
        net = CausalAttention(model[name], spec[name], spec['heads'], spec['window'])
    else:
        net = network(model[name])
    return (wrap(net, model) if name == 'actor' else net).to('cuda')


def output(net, x):
    value = net(x)
    return value[0] if isinstance(value, tuple) else value


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU fallback forbidden')
    torch.set_default_device('cuda')
    torch.manual_seed(20261010)
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for name in ('mlp64', 'gru128', 'attention64', 'attention128'):
        source = args.source / (name + '.weights.json')
        model = json.loads(source.read_text(encoding='utf-8-sig'))
        migrated = migrate(model)
        # Sequence shape tests temporal context as well as ordinary MLP inputs.
        x = torch.randn(3, 12, 854, device='cuda')
        x[...,426:466:5] = torch.randint(0, 2, (3,12,8), device='cuda')
        nav = torch.randn(3,12,27, device='cuda')
        nx = torch.cat((x,nav), -1)
        diffs = {}
        gradient_norms = {}
        for role in ('actor', 'value'):
            old, new = branch(model, role), branch(migrated, role)
            assert all(p.device.type == 'cuda' for p in new.parameters())
            with torch.no_grad():
                a, b = output(old,x), output(new,nx)
                diff = float((a-b).abs().max())
                torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)
                diffs[role] = diff
            new.zero_grad()
            y = output(new,nx)
            (y.square().mean()+y.mean()).backward()
            candidates = [p for p in new.parameters() if p.ndim==2 and p.shape[1]==881]
            if len(candidates)!=1:
                raise ValueError('Ambiguous first layer')
            g = candidates[0].grad[:,854:]
            assert bool(torch.isfinite(g).all()) and float(g.norm())>0
            gradient_norms[role] = float(g.norm())
        assert torch.equal(availability(x.reshape(-1,854)),availability(nx.reshape(-1,881)))
        dest = args.output / (name + '.weights.json')
        with dest.open('x', encoding='utf-8') as f:
            json.dump(migrated,f,allow_nan=False,separators=(',',':'))
        results.append(dict(name=name,source=str(source.resolve()),source_sha256=sha(source),
                            weights=str(dest.resolve()),weights_sha256=sha(dest),
                            max_output_difference=diffs,navigation_gradient_norm=gradient_norms))
        print(name, 'CUDA migration and gradients verified', flush=True)
    report = dict(device=torch.cuda.get_device_name(),feature_version='combat_features_v8',
                  prefix_width=854,input_width=881,training_performed=False,
                  optimizer_checkpoint_migrated=False,results=results)
    (args.output/'migration-verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()

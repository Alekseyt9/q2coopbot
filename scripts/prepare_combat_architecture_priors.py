"""CUDA-only matched teacher distillation for fresh combat architecture arms.

No PPO rollout reuse: exported priors require their own fresh on-policy battles.
Validation is measured, never used for gradients. Final test remains deferred.
"""
import argparse, copy, json, pathlib, time
from ppo_combat import torch, nn, layers, network, sha
from ppo_recurrent import Recurrent, initialize, VERSION, durable_json, durable_write
from combat_attention import CausalAttention, attention_cell, TEMPORAL
from train_combat_sequence_bc import read, tensors


class FeedForward(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.net = network(base)

    def forward(self, x):
        return self.net(x)

    def export(self):
        return layers(self.net), None


def build(model, name, device='cuda'):
    if model.get('attention'):
        cell = model['attention']
        module = CausalAttention(model[name], cell[name], cell['heads'], cell['window'])
    elif model.get('memory'):
        module = Recurrent(model[name], model['memory'][name])
    else:
        module = FeedForward(model[name])
    return module.to(device)


def output(module, x):
    result = module(x)
    return result[0] if isinstance(result, tuple) else result


def fresh_model(parent, architecture, width):
    result = {k: copy.deepcopy(v) for k, v in parent.items()
              if k not in ('actor', 'value', 'attention', 'memory', 'entity_attention')}
    for name in ('actor', 'value'):
        inputs = len(parent[name][0]['weight'][0])
        outputs = len(parent[name][-1]['bias'])
        result[name] = layers(nn.Sequential(nn.Linear(inputs, width, device='cuda'), nn.ReLU(),
                                           nn.Linear(width, width, device='cuda'), nn.ReLU(),
                                           nn.Linear(width, outputs, device='cuda')))
    if architecture == 'attention':
        result['attention'] = dict(version=TEMPORAL, heads=4, window=32,
            **{name: attention_cell(width, len(result[name][-1]['bias']), 4)
               for name in ('actor', 'value')})
    elif architecture == 'gru':
        result = initialize(result, width)
    result['sampling_seed'] = 0
    result['deterministic'] = False
    return result


def main():
    ap = argparse.ArgumentParser()
    for arg in ('parent', 'data', 'config', 'out'):
        ap.add_argument('--'+arg, type=pathlib.Path, required=True)
    args = ap.parse_args()
    assert torch.cuda.is_available(), 'CUDA required'
    assert not args.out.exists(), 'Fresh output required'
    config, parent, meta = read(args.config), read(args.parent), read(args.data/'report.json')
    assert config['version'] == 'combat_architecture_prior_v1' and meta['test_deferred']
    assert parent['feature_version'] == meta['feature_version'] and parent.get('attention')
    assert len(set(config['seeds'])) == len(config['seeds']) >= 2
    assert 1 <= config['epochs'] <= 2000 and config['learning_rate'] > 0
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    receipts = {str(args.parent): sha(args.parent), str(args.config): sha(args.config),
                str(args.data/'report.json'): sha(args.data/'report.json')}
    rows, data = {}, {}
    for split in ('train', 'validation'):
        path = args.data/(split+'.jsonl')
        receipts[str(path)] = sha(path)
        assert receipts[str(path)] == meta['data_sha256'][split]
        rows[split] = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
        data[split] = tensors(rows[split], 'cuda', require_attack=False)
    assert not {r['seed'] for r in rows['train']} & {r['seed'] for r in rows['validation']}
    teacher = {name: build(parent, name) for name in ('actor', 'value')}
    with torch.no_grad():
        targets = {s: {name: output(module, d[0]).detach() for name, module in teacher.items()}
                   for s, d in data.items()}
    args.out.mkdir()
    history = []
    for seed in config['seeds']:
        for arm in config['arms']:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = fresh_model(parent, arm['architecture'], arm['width'])
            modules = {name: build(model, name) for name in ('actor', 'value')}
            parameters = [p for module in modules.values() for p in module.parameters()]
            optimizer = torch.optim.Adam(parameters, lr=config['learning_rate'])

            def loss(split):
                x, _, _, _, _, valid, _ = data[split]
                predicted = {n: output(m, x) for n, m in modules.items()}
                actor_error = predicted['actor'][valid]-targets[split]['actor'][valid]
                value_error = predicted['value'][valid]-targets[split]['value'][valid]
                objective = actor_error.square().mean()+config['value_weight']*value_error.square().mean()
                metrics = dict(actor_raw_rmse=float(actor_error.detach().square().mean().sqrt()),
                    value_rmse=float(value_error.detach().square().mean().sqrt()),
                    movement_rmse=float((predicted['actor'][..., :2].tanh()[valid]-
                                         targets[split]['actor'][..., :2].tanh()[valid]).square().mean().sqrt()),
                    aim_rmse_degrees=float((predicted['actor'][..., 2:4].tanh()[valid]-
                                            targets[split]['actor'][..., 2:4].tanh()[valid]).square().mean().sqrt()*180),
                    fire_probability_rmse=float((predicted['actor'][...,4].sigmoid()[valid]-
                                                  targets[split]['actor'][...,4].sigmoid()[valid]).square().mean().sqrt()))
                return objective, metrics

            with torch.no_grad():
                before = {s: loss(s)[1] for s in data}
            start = time.perf_counter()
            curve = []
            for epoch in range(config['epochs']):
                optimizer.zero_grad()
                objective, _ = loss('train')
                assert torch.isfinite(objective)
                objective.backward()
                nn.utils.clip_grad_norm_(parameters, 1.)
                optimizer.step()
                if epoch % 50 == 0 or epoch+1 == config['epochs']:
                    with torch.no_grad():
                        curve.append(dict(epoch=epoch+1, **{s: loss(s)[1] for s in data}))
            torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            with torch.no_grad():
                after = {s: loss(s)[1] for s in data}
            for name, module in modules.items():
                base, cell = module.export()
                model[name] = base
                if model.get('attention'): model['attention'][name] = cell
                if model.get('memory'): model['memory'][name] = cell
            # CUDA roundtrip and causal-prefix checks; no Go replay.
            checks = {}
            with torch.no_grad():
                for name, module in modules.items():
                    restored = build(model, name)
                    x = data['validation'][0][:2, :40]
                    y = output(module, x)
                    roundtrip = float((y-output(restored, x)).abs().max())
                    causal = float((y[:, :16]-output(module, x[:, :16])).abs().max())
                    assert roundtrip < 3e-5 and causal < 3e-5, (name, roundtrip, causal)
                    checks[name] = dict(cuda_export_max_error=roundtrip, causal_prefix_max_error=causal)
            arm_root = args.out/(arm['id']+'-seed-'+str(seed))
            arm_root.mkdir()
            durable_json(arm_root/'weights.json', model)
            durable_write(arm_root/'initialization.pt', lambda f: torch.save(dict(
                version='combat_distilled_prior_checkpoint_v1', architecture=arm['architecture'],
                actor=modules['actor'].state_dict(), value=modules['value'].state_dict(),
                optimizer=optimizer.state_dict(), epochs=config['epochs'],
                parent_sha256=receipts[str(args.parent)], weights_sha256=sha(arm_root/'weights.json'),
                data_sha256=meta['data_sha256'], ppo_updates_completed=0,
                scope='Distillation checkpoint; not a PPO resume checkpoint'), f))
            report = dict(version='combat_distilled_prior_v1', arm=arm, seed=seed, device='cuda',
                actor_parameters=sum(p.numel() for p in modules['actor'].parameters()),
                value_parameters=sum(p.numel() for p in modules['value'].parameters()),
                epochs=config['epochs'], training_contexts=meta['counts']['train'],
                validation_contexts=meta['counts']['validation'], before=before, after=after,
                curve=curve, checks=checks, seconds=seconds, source_sha256=receipts,
                ppo_updates_completed=0, weights_sha256=sha(arm_root/'weights.json'),
                scope='Matched fixed-budget parent distillation. Teacher has earlier experience; all fresh arms receive identical corpus/epochs. Validation only measured. Not hit accuracy, live improvement or PPO experience; own-policy rollouts required.')
            durable_json(arm_root/'report.json', report)
            durable_json(arm_root/'complete.json', dict(weights_sha256=sha(arm_root/'weights.json'),
                checkpoint_sha256=sha(arm_root/'initialization.pt'), report_sha256=sha(arm_root/'report.json')))
            history.append(report)
            print(json.dumps(dict(arm=arm['id'], seed=seed, after=after, seconds=seconds)), flush=True)
    for path, digest in receipts.items():
        assert sha(pathlib.Path(path)) == digest, 'Source changed during distillation'
    durable_json(args.out/'report.json', dict(version='combat_architecture_prior_comparison_v1',
        state='complete', source_unchanged=True, config=config, results=history,
        trainer_sha256=sha(pathlib.Path(__file__)), final_test_deferred=True,
        scope='Offline GPU comparison only. No architecture promoted. Fresh native PPO and live paired evaluation remain required.'))


if __name__ == '__main__':
    main()

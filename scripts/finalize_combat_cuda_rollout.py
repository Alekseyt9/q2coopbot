"""Verify behavior likelihood/context and finalize exact native bootstrap on CUDA."""
import argparse
import json
import pathlib
import shutil

from process_combat_architecture_pool import read, save, sha
from ppo_combat import network, torch
from ppo_recurrent import prepare_context, sequence, Recurrent
from combat_attention import CausalAttention, EntityAttention
from combat_cuda_bootstrap import next_values


def finalize(model_path, data, out):
    assert torch.cuda.is_available(), 'CUDA is required; no CPU fallback'
    assert not out.exists(), 'Fresh CUDA output required'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    model, meta = read(model_path), read(data / 'report.json')
    assert meta['version'] == 'combat_ppo_native_cuda_pending_v1'
    assert meta['numerical_verification'] == 'cuda_pending'
    assert meta['model_sha256'] == sha(model_path) and not model['deterministic']
    assert meta['feature_version'] == model['feature_version']
    assert sha(data / 'native-rollout.jsonl') == meta['native_rollout_sha256']
    sources = dict(meta['source_sha256'])
    sources.update({str((data / 'report.json').resolve()): sha(data / 'report.json'),
                    str((data / 'native-rollout.jsonl').resolve()): meta['native_rollout_sha256'],
                    str(pathlib.Path(__file__).resolve()): sha(__file__),
                    str(pathlib.Path(__file__).with_name('combat_cuda_bootstrap.py').resolve()):
                        sha(pathlib.Path(__file__).with_name('combat_cuda_bootstrap.py'))})
    for path, digest in sources.items():
        assert sha(path) == digest, 'Native source changed: ' + path
    rows = [json.loads(line) for line in (data / 'native-rollout.jsonl').read_text().splitlines()]
    assert len(rows) == meta['rows'] > 0
    assert all(r['sample']['version'] == meta['policy_version'] for r in rows)
    context, prepared = [], None
    if meta.get('recurrent_version'):
        assert sha(data / 'sequence.jsonl') == meta['sequence_sha256']
        context = [json.loads(line) for line in (data / 'sequence.jsonl').read_text().splitlines()]
        assert len(context) == meta['sequence_rows']
        sources[str((data / 'sequence.jsonl').resolve())] = meta['sequence_sha256']
        prepared = prepare_context(context, rows, 'cuda')

    def build(name):
        if model.get('memory'):
            module = Recurrent(model[name], model['memory'][name])
        elif model.get('attention'):
            spec = model['attention']
            module = CausalAttention(model[name], spec[name], spec['heads'], spec['window'])
        elif model.get('entity_attention'):
            spec = model['entity_attention']
            module = EntityAttention(model[name], spec[name], spec['heads'])
        else:
            module = network(model[name])
        if name == 'actor' and model.get('spatial_aim'):
            from combat_spatial_aim import wrap
            module = wrap(module, model)
        return module.to('cuda')

    actor, value = build('actor'), build('value')
    x = torch.tensor([r['features'] for r in rows], device='cuda', dtype=torch.float32)
    nx = torch.tensor([r['next_features'] for r in rows], device='cuda', dtype=torch.float32)
    assert x.shape == nx.shape and bool(torch.isfinite(x).all()) and bool(torch.isfinite(nx).all())
    reset = torch.tensor([r['next_reset'] for r in rows], device='cuda', dtype=torch.bool)
    zero = torch.tensor([r['bootstrap_zero'] for r in rows], device='cuda', dtype=torch.bool)
    assert all(not r['terminal'] or r['bootstrap_zero'] for r in rows)
    assert all(r['next_frame'] == r['frame'] + 1 or r['next_reset'] for r in rows)
    z = torch.tensor([r['sample']['latent'] for r in rows], device='cuda', dtype=torch.float32)
    attack = torch.tensor([float(r['sample']['attack']) for r in rows], device='cuda')
    vertical = torch.tensor([r['sample']['vertical'] for r in rows], device='cuda', dtype=torch.long)
    weapon = torch.tensor([r['sample'].get('weapon', 0) for r in rows], device='cuda', dtype=torch.long)
    target = torch.tensor([r['sample'].get('target', 0) for r in rows], device='cuda', dtype=torch.long)
    mode = torch.tensor([r['sample'].get('aim_mode', 0) for r in rows], device='cuda', dtype=torch.long)
    std = torch.tensor(model['log_std'], device='cuda', dtype=torch.float32)

    def evaluate(module, name):
        if prepared is not None:
            return sequence(module, prepared, 32, True, name)
        if model.get('entity_attention'):
            return module.single(x), 0.
        return module(x), 0.

    with torch.no_grad():
        raw, actor_memory_error = evaluate(actor, 'actor')
        predicted, critic_memory_error = evaluate(value, 'value')
        if model.get('aim_mode_head'):
            from combat_precision_head import probabilities
            lp, _ = probabilities(raw, std, z, attack, vertical, weapon, x, target, mode)
        elif model.get('target_head'):
            from combat_target_head import probabilities
            lp, _ = probabilities(raw, std, z, attack, vertical, weapon, x, target)
        else:
            from combat_weapon_head import probabilities, feature_mask
            lp, _ = probabilities(raw, std, z, attack, vertical,
                                   weapon if model.get('weapon_head') else None,
                                   feature_mask(x) if model.get('weapon_head') else None)
        log_error = float((lp - torch.tensor([r['sample']['log_probability'] for r in rows], device='cuda')).abs().max())
        value_error = float((predicted[:, 0] - torch.tensor([r['sample']['value'] for r in rows], device='cuda')).abs().max())
        assert log_error < .003 and value_error < 1e-4 and max(actor_memory_error, critic_memory_error) < 2e-5, (
            log_error, value_error, actor_memory_error, critic_memory_error)
        bootstrap = next_values(value, nx, prepared, reset=reset)
        bootstrap[zero] = 0
        assert bool(torch.isfinite(bootstrap).all())
        values = bootstrap.cpu().tolist()  # Serialization only; neural computation stays on CUDA.
    for path, digest in sources.items():
        assert sha(path) == digest, 'Input changed during CUDA verification: ' + path
    out.mkdir()
    with (out / 'rollout.jsonl').open('w', encoding='utf-8') as stream:
        for row, nv in zip(rows, values):
            row['next_value'] = nv
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    if prepared is not None:
        shutil.copy2(data / 'sequence.jsonl', out / 'sequence.jsonl')
    audit = dict(state='passed', device='cuda', rows=len(rows), context_rows=len(context),
                 log_probability_max_error=log_error, value_max_error=value_error,
                 actor_memory_max_error=actor_memory_error, critic_memory_max_error=critic_memory_error,
                 next_resets=int(reset.sum()), zero_bootstraps=int(zero.sum()),
                 native_rollout_sha256=meta['native_rollout_sha256'], model_sha256=sha(model_path),
                 scope='Exact native Step.Next features and reset flags; numerical checks and bootstrap on CUDA. Go performs only native proof, feature extraction and sampled action contract checks.')
    save(out / 'cuda-verification.json', audit)
    sources[str((out / 'cuda-verification.json').resolve())] = sha(out / 'cuda-verification.json')
    meta.update(version='combat_ppo_rollout_v1', numerical_verification='cuda_verified_v1',
                rollout_sha256=sha(out / 'rollout.jsonl'), source_sha256=sources,
                cuda_verification_sha256=sha(out / 'cuda-verification.json'))
    save(out / 'report.json', meta)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=pathlib.Path)
    parser.add_argument('--data', required=True, type=pathlib.Path)
    parser.add_argument('--out', required=True, type=pathlib.Path)
    args = parser.parse_args()
    print(json.dumps(finalize(args.model.resolve(), args.data.resolve(), args.out.resolve())), flush=True)


if __name__ == '__main__':
    main()

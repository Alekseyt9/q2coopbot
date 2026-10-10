"""Scale only categorical aim logits, including every additive residual; CUDA audit."""
import argparse
import copy
import json
import pathlib

from prepare_combat_target_heads import build, forward
from process_combat_architecture_pool import read, save, sha
from train_combat_bc import torch


def transform(source, target_scale, mode_scale):
    assert target_scale in (1, 4) and mode_scale in (1, 4)
    assert source['aim_mode_head'] == 'combat_target_coarse_fine_aim_v1'
    assert source['target_head'] == 'combat_target_conditioned_aim_v1'
    assert not source.get('entity_attention') and not source['deterministic']
    result = copy.deepcopy(source)
    scales = {i: target_scale for i in range(20, 29)}
    scales.update({i: mode_scale for i in range(63, 81)})

    def scale_layer(layer, rows):
        for index, factor in rows.items():
            layer['weight'][index] = [value * factor for value in layer['weight'][index]]
            layer['bias'][index] *= factor

    assert len(result['actor'][-1]['bias']) == 81
    scale_layer(result['actor'][-1], scales)
    if result.get('memory'):
        scale_layer(result['memory']['actor']['output'], scales)
    if result.get('attention'):
        scale_layer(result['attention']['actor']['residual'], scales)
    if result.get('spatial_aim'):
        scale_layer(result['spatial_aim']['layers'][-1], {2: mode_scale, 3: mode_scale})
    assert result['value'] == source['value'] and result['log_std'] == source['log_std']
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('model', 'sequence', 'out'):
        parser.add_argument('--' + name, type=pathlib.Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'Numerical model validation requires CUDA'
    assert not args.out.exists() or not any(args.out.iterdir()), 'Refuse existing prepared artifacts'
    source = read(args.model)
    features = []
    with args.sequence.open(encoding='utf-8-sig') as stream:
        for line in stream:
            features.append(json.loads(line)['features'])
            if len(features) == 32:
                break
    assert len(features) == 32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    x = torch.tensor(features, device='cuda', dtype=torch.float32)[None, :, :]
    actor, critic = build(source, 'actor').eval(), build(source, 'value').eval()
    with torch.no_grad():
        baseline, values = forward(actor, x), forward(critic, x)
    assert baseline.device.type == 'cuda' and bool(torch.isfinite(baseline).all())
    args.out.mkdir(exist_ok=True)
    records = []
    for name, target_scale, mode_scale in [('baseline', 1, 1), ('mode4', 1, 4), ('targetmode4', 4, 4)]:
        result = transform(source, target_scale, mode_scale)
        factors = torch.ones(81, device='cuda')
        factors[20:29] = target_scale
        factors[63:81] = mode_scale
        with torch.no_grad():
            actual = forward(build(result, 'actor').eval(), x)
            actual_value = forward(build(result, 'value').eval(), x)
        assert actual.device.type == actual_value.device.type == 'cuda'
        error = float((actual - baseline * factors).abs().max())
        assert error < 2e-5 and bool(torch.isfinite(actual).all()), (name, error)
        unchanged = factors == 1
        assert torch.equal(actual[..., unchanged], baseline[..., unchanged])
        assert torch.equal(actual_value, values)
        if name == 'baseline':
            assert result == source
        path = args.out / (name + '-weights.json')
        save(path, result)
        records.append(dict(name=name, model=str(path.resolve()), model_sha256=sha(path),
            target_logit_scale=target_scale, mode_logit_scale=mode_scale,
            raw_scaling_max_error=error, unchanged_outputs_exact=True, critic_exact=True))
    save(args.out / 'cuda-audit.json', dict(state='passed', device='cuda', gpu=torch.cuda.get_device_name(),
        parent_model=str(args.model.resolve()), parent_model_sha256=sha(args.model), sequence_sha256=sha(args.sequence),
        real_sequence_rows=32, models=records,
        scope='Inference temperature transform only;32 real feature rows on CUDA. All additive actor/temporal/GRU/spatial logit contributions scaled. '
              'Means/fire/vertical/weapon/std/value outputs and argmax rankings unchanged for identical context, categorical confidence differs. '
              'Runtime observations can diverge. No PPO update, checkpoint/optimizer resume or native quality acceptance. No CPU/Go NN replay.'))
    print([(record['name'], record['raw_scaling_max_error']) for record in records], flush=True)


if __name__ == '__main__':
    main()

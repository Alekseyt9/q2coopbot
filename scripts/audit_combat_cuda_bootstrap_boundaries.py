"""CUDA-only full-prefix reference for bootstrap reset/window boundaries."""
import argparse
import json
import pathlib

from combat_cuda_bootstrap import next_values
from combat_attention import CausalAttention
from prepare_combat_target_heads import build, forward
from process_combat_architecture_pool import read, save, sha
from ppo_recurrent import prepare_context, torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=pathlib.Path, required=True)
    parser.add_argument('--data', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available() and not args.out.exists()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    meta = read(args.data / 'report.json')
    assert meta['model_sha256'] == sha(args.model)
    assert sha(args.data / 'rollout.jsonl') == meta['rollout_sha256']
    assert sha(args.data / 'sequence.jsonl') == meta['sequence_sha256']
    rows = [json.loads(line) for line in (args.data / 'rollout.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    context = [json.loads(line) for line in (args.data / 'sequence.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    prepared = prepare_context(context, rows, 'cuda')
    x, indices, inverse, segments = prepared
    locations = indices[inverse].cpu().tolist()  # Integer indexing metadata, no model calculations.
    window = read(args.model).get('attention', {}).get('window', 32)
    selected = []
    for batch in range(min(8, len(segments))):
        wanted = {0, 1, window - 1, window, 2 * window, len(segments[batch]) - 1}
        selected.extend(i for i, (b, t) in enumerate(locations) if b == batch and t in wanted)
    assert selected and any(locations[i][1] >= window for i in selected)
    chosen = torch.tensor(selected, dtype=torch.long, device='cuda')
    value = build(read(args.model), 'value')
    features = torch.stack([x[locations[i][0], locations[i][1]] for i in selected]) + .01
    references, reset_references = [], []
    with torch.no_grad():
        for j, i in enumerate(selected):
            batch, time = locations[i]
            prefix = torch.cat((x[batch:batch + 1, :time + 1], features[j:j + 1, None, :]), dim=1)
            references.append(forward(value, prefix)[0, -1, 0])
            reset_references.append(forward(value, features[j:j + 1, None, :])[0, -1, 0])
        reference = torch.stack(references)
        reset_reference = torch.stack(reset_references)
        continued = next_values(value, features, prepared, chosen)
        reset = torch.ones(len(selected), dtype=torch.bool, device='cuda')
        cleared = next_values(value, features, prepared, chosen, reset)
        mixed_mask = torch.arange(len(selected), device='cuda') % 2 == 0
        mixed = next_values(value, features, prepared, chosen, mixed_mask)
        # A large change to future padding/frames must not change current bootstrap.
        changed = x.clone()
        for batch in range(len(segments)):
            times = [locations[i][1] for i in selected if locations[i][0] == batch]
            if times:
                changed[batch, max(times) + 1:] += 100
        causal = next_values(value, features, (changed, indices, inverse, segments), chosen)
        errors = dict(continue_full_prefix=float((continued - reference).abs().max()),
                      reset_empty_prefix=float((cleared - reset_reference).abs().max()),
                      mixed_reset=float((mixed - torch.where(mixed_mask, reset_reference, reference)).abs().max()),
                      future_padding=float((causal - continued).abs().max()))
        assert torch.isfinite(continued).all() and torch.isfinite(cleared).all()
    report = dict(state='passed' if max(errors.values()) < 3e-5 else 'mismatch', device='cuda',
        checked_locations=len(selected), attention_window=window, errors=errors,
        model_sha256=sha(args.model), rollout_sha256=meta['rollout_sha256'], sequence_sha256=meta['sequence_sha256'],
        helper_sha256=sha(pathlib.Path(__file__).with_name('combat_cuda_bootstrap.py')),
        scope='Synthetic next-feature perturbations at real prefix/window/end locations, compared with independent full-prefix CUDA replay and empty-prefix reset. Future padding invariance only; actual Step.Next/reset/handoff native export remains unverified. No Go model execution, training or pipeline replacement.')
    save(args.out, report)
    print(json.dumps(report), flush=True)
    assert report['state'] == 'passed', errors


if __name__ == '__main__':
    main()

"""Seal train-only native transitions with causal frozen-encoder CUDA caches.

Bootstrap follows the existing finite provider-segment contract, including
explicit handoff truncations. A missing successor is never guessed terminal.
"""
import argparse
import collections
import gzip
import hashlib
import json
import pathlib

from process_combat_architecture_pool import read, save, sha


def adjacent_rows(rows, context):
    lookup = {(c['seed'], c['index']): i for i, c in enumerate(context)}
    assert len(lookup) == len(context)
    pairs, missing = [], []
    for i, row in enumerate(rows):
        key = row['seed'], row['index']
        assert key in lookup
        current = lookup[key]
        assert context[current]['features'] == row['features'] and context[current]['frame'] == row['frame']
        assert row['next_frame'] == row['frame'] + 1
        stop = row['bootstrap_zero']
        assert isinstance(stop, bool) and (not row['terminal'] or stop)
        if stop:
            assert row['terminal'] or row['truncated'], 'Unexplained zero bootstrap'
            pairs.append((i, current, current, False))
            continue
        next_index = lookup.get((row['seed'], row['index'] + 1))
        if next_index is None:
            missing.append(key)
            continue
        successor = context[next_index]
        assert not row['next_reset'] and not successor['memory']['reset'], 'Cross-reset bootstrap'
        assert successor['frame'] == row['next_frame']
        assert successor['features'] == row['next_features'], 'Successor differs from native next observation'
        pairs.append((i, current, next_index, True))
    assert pairs
    return pairs, missing


def main():
    ap = argparse.ArgumentParser()
    for name in ('source', 'corpus', 'out'): ap.add_argument('--' + name, type=pathlib.Path, required=True)
    a = ap.parse_args()
    from ppo_recurrent import torch, prepare_context, sequence, durable_json, durable_write
    from prepare_combat_target_heads import build
    from combat_precision_head import probabilities
    assert torch.cuda.is_available() and not a.out.exists()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    source, corpus = a.source.resolve(), a.corpus.resolve()
    pins = {}

    def pin(path, expected=None):
        path = pathlib.Path(path); digest = sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path.resolve())] = digest
        return path

    receipt = read(pin(source / 'complete.json'))
    for file, key in (('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'), ('report.json', 'report_sha256')):
        pin(source / file, receipt[key])
    seal = read(pin(corpus / 'complete.json')); assert seal['state'] == 'complete' and not seal['training_performed']
    protocol = read(pin(corpus / 'protocol.json', seal['protocol_sha256']))
    assert pathlib.Path(protocol['source']) == source and protocol['train_only'] and not protocol['diagnostic_intervention']
    objective = protocol['ppo_training_objective']; assert objective
    cp = torch.load(source / 'checkpoint.pt', map_location='cuda', weights_only=True)
    assert cp['config']['objective_reward_sha256'] == objective['reward_sha256']
    assert cp['weights_sha256'] == receipt['weights_sha256']
    pool = read(pin(corpus / 'pool/report.json', seal['pool_sha256']))
    assert pool['state'] == 'complete' and pool['source_unchanged'] and not any(j['error'] for j in pool['jobs'])
    assert len(seal['processed']) == 1
    binding = seal['processed'][0]; assert binding['label'] == 'train'
    processing = pin(binding['report'], binding['sha256']).parent
    closed = read(processing / 'report.json')
    assert closed['device'] == 'cuda' and closed['state'] == 'complete' and not closed['training_performed']
    assert all(c['split'] == 'train' for c in closed['cases'])
    merged = processing / 'merged'; meta = read(pin(merged / 'report.json'))
    assert meta['model_sha256'] == receipt['weights_sha256'] and meta['reward_config_sha256'] == objective['reward_sha256']
    for file, digest in meta['source_sha256'].items(): pin(file, digest)

    def stream(name):
        path = merged / (name + '.jsonl')
        if not path.exists(): path = pathlib.Path(str(path) + '.gz')
        pin(path); digest = hashlib.sha256(); items = []
        with (gzip.open(path, 'rb') if path.suffix == '.gz' else path.open('rb')) as f:
            for line in f:
                digest.update(line); items.append(json.loads(line))
        assert digest.hexdigest() == meta[name + '_sha256']
        return items

    rows, context = stream('rollout'), stream('sequence')
    assert len(rows) == meta['rows'] and len(context) == meta['sequence_rows']
    seeds = {s for assignment in protocol['assignments'] for s in assignment['seeds']}
    assert all(c['seed'] in seeds for c in context) and {r['seed'] for r in rows} <= seeds
    pairs, missing = adjacent_rows(rows, context)
    model = read(source / 'weights.json'); assert model.get('attention') and not model.get('shared_target')
    actor = build(model, 'actor'); prepared = prepare_context(context, rows, 'cuda')
    with torch.no_grad():
        raw, memory_error = sequence(actor, prepared, check_states=True)
        assert memory_error < 2e-5
        x, _, _, segments = prepared
        core = actor.base
        _, tokens = core(x)
        length = tokens.shape[1]; positions = torch.arange(length, device='cuda')
        mask = (positions[None, :] > positions[:, None]) | (positions[None, :] < positions[:, None] - core.window + 1)
        attended, _ = core.mha(tokens, tokens, tokens, attn_mask=mask, need_weights=False)
        output = actor(x)[0]
        locations = [(b, t) for b, segment in enumerate(segments) for t in range(len(segment))]
        flat_keys = [(c['seed'], c['index']) for segment in segments for c in segment]
        assert flat_keys == [(c['seed'], c['index']) for c in context]
        indices = torch.tensor(locations, device='cuda', dtype=torch.long)
        features = x[indices[:, 0], indices[:, 1]]
        logits = output[indices[:, 0], indices[:, 1]]
        history = attended[indices[:, 0], indices[:, 1]]
        samples = [r['sample'] for r in rows]
        z = torch.tensor([s['latent'] for s in samples], device='cuda', dtype=torch.float32)
        condition = torch.tensor([[s.get('target', 0), s.get('aim_mode', 0), s['vertical'], s.get('weapon', 0), int(s['attack'])] for s in samples], device='cuda', dtype=torch.long)
        lp, _ = probabilities(raw, cp['log_std'], z, condition[:, 4].float(), condition[:, 2], condition[:, 3],
            torch.tensor([r['features'] for r in rows], device='cuda', dtype=torch.float32), condition[:, 0], condition[:, 1], actor.sampled_motion_aim_branch)
        log_error = float((lp - torch.tensor([s['log_probability'] for s in samples], device='cuda')).abs().max())
        assert log_error < .003
        selected = torch.tensor([p[0] for p in pairs], device='cuda', dtype=torch.long)
        data = dict(version='combat_causal_replay_v1', source_weights_sha256=receipt['weights_sha256'],
            reward_sha256=objective['reward_sha256'], features=features, raw=logits, history=history,
            current=torch.tensor([p[1] for p in pairs], device='cuda', dtype=torch.long),
            successor=torch.tensor([p[2] for p in pairs], device='cuda', dtype=torch.long),
            bootstrap=torch.tensor([p[3] for p in pairs], device='cuda', dtype=torch.bool),
            latent=z[selected], condition=condition[selected],
            reward=torch.tensor([rows[p[0]]['reward'] for p in pairs], device='cuda', dtype=torch.float32))
        for value in data.values():
            if torch.is_tensor(value): assert value.is_cuda and bool(torch.isfinite(value).all())
    a.out.mkdir(parents=True)
    durable_write(a.out / 'replay.pt', lambda f: torch.save(data, f))
    restored = torch.load(a.out / 'replay.pt', map_location='cuda', weights_only=True)
    for key, value in data.items():
        if torch.is_tensor(value): assert restored[key].is_cuda and torch.equal(value, restored[key])
        else: assert value == restored[key]
    for path, digest in pins.items(): assert sha(path) == digest
    durable_json(a.out / 'keys.json', [dict(seed=rows[p[0]]['seed'], index=rows[p[0]]['index'], frame=rows[p[0]]['frame'], bootstrap=p[3]) for p in pairs])
    durable_json(a.out / 'complete.json', dict(state='complete', version=data['version'], device='cuda',
        source=str(source), corpus=str(corpus), source_weights_sha256=receipt['weights_sha256'],
        reward_sha256=objective['reward_sha256'], replay_sha256=sha(a.out / 'replay.pt'), keys_sha256=sha(a.out / 'keys.json'),
        rows=len(pairs), context_rows=len(context), training_cases=len(closed['cases']),
        bootstrap_rows=sum(p[3] for p in pairs), zero_bootstrap_rows=sum(not p[3] for p in pairs),
        excluded_missing_successor=missing, log_probability_max_error=log_error, memory_max_error=memory_error,
        inputs=[dict(path=p, sha256=h) for p, h in pins.items()], script_sha256=sha(__file__),
        training_performed=False, promotion=False,
        scope='TRAIN-only actual native transitions. Causal frozen encoder cache, exact adjacent next observations; no invented successors/cross-reset bootstrap. Explicit finite provider-segment bootstrap_zero semantics, not inferred full-game completion. No native hidden state in current features.'))
    print(dict(state='complete', rows=len(pairs), excluded=len(missing), memory_error=memory_error, log_probability_error=log_error))


if __name__ == '__main__': main()

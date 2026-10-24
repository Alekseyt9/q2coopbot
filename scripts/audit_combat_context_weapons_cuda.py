"""Inspect actual masked weapon probabilities on sealed TRAIN history, CUDA only."""
import argparse
import json
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha
from ppo_combat import torch, training_devices
from ppo_recurrent import prepare_context, sequence
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model
from combat_weapon_head import feature_mask, distribution
from combat_weapon_features import WEAPONS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'data', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--frames', required=True)
    a = ap.parse_args()
    assert not a.out.exists()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    def lines(path):
        with path.open(encoding='utf-8-sig') as stream:
            return [json.loads(line) for line in stream]

    pin(__file__)
    source = read(pin(a.source/'complete.json'))
    model = read(pin(a.source/'weights.json', source['weights_sha256']))
    meta = read(pin(a.data/'cuda/report.json'))
    assert meta['model_sha256'] == source['weights_sha256']
    assert meta['numerical_verification'] == 'cuda_verified_v1' and not meta.get('diagnostic_only')
    rows = lines(pin(a.data/'cuda/rollout.jsonl', meta['rollout_sha256']))
    histories = lines(pin(a.data/'cuda/sequence.jsonl', meta['sequence_sha256']))
    paths = list((a.data/'native').glob('replay-*/steps.jsonl'))
    assert len(paths) == 1
    steps = {s['index']: s for s in lines(pin(paths[0], meta['source_sha256'][str(paths[0].resolve())]))}
    for row in rows:
        assert steps[row['index']]['owner'] == 'provider' and steps[row['index']]['observation']['identity']['life'] == 1
    training_devices()
    torch.set_default_device('cuda')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    actor = branch(base_model(model, allow_shared=True), 'actor').requires_grad_(False)
    context = prepare_context(histories, rows, 'cuda')
    x = torch.tensor([r['features'] for r in rows], device='cuda', dtype=torch.float32)
    with torch.no_grad():
        raw, memory_error = sequence(actor, context, 32, True, 'actor')
        assert memory_error < 2e-5 and raw.is_cuda
        mask = feature_mask(x[:, :845])
        probabilities = distribution(raw[:, 8:20], mask).probs
        assert probabilities.is_cuda
    wanted = sorted({int(f) for f in a.frames.split(',')})
    details = []
    for frame in wanted:
        indexes = [i for i, row in enumerate(rows) if row['frame'] == frame]
        assert len(indexes) == 1
        i = indexes[0]
        observation = steps[rows[i]['index']]['observation']
        active = [j for j, enabled in enumerate(mask[i].tolist()) if enabled]
        details.append(dict(frame=frame, index=rows[i]['index'], equipped=observation['weapon'],
            inventory=observation.get('inventory'), inventory_age_frames=observation.get('inventory_age_frames'),
            sampled_weapon=WEAPONS[rows[i]['sample'].get('weapon', 0)],
            probabilities={WEAPONS[j] or 'keep': float(probabilities[i, j]) for j in active}))
    assert all(sha(path) == digest for path, digest in pins.items())
    a.out.mkdir(parents=True)
    save(a.out/'report.json', dict(state='complete', version='combat_context_weapon_distribution_v1',
        device='cuda', model_sha256=source['weights_sha256'], memory_replay_max_error=memory_error,
        details=details, inputs=pins, source_ordinary_ppo_training_ready=meta['ppo_training_ready'],
        training_performed=False, new_games=False,
        scope='Actual actor wrapper includes trained weapon residual and observed-state temperature. '
              'Masked categorical probabilities on the sealed own-policy TRAIN history; no selected action or tensor changed. '
              'Availability/probability diagnosis only, not proof that a weapon switch improves survival.'))
    save(a.out/'complete.json', dict(state='complete', report_sha256=sha(a.out/'report.json')))
    print(dict(device='cuda', details=details))


if __name__ == '__main__':
    main()

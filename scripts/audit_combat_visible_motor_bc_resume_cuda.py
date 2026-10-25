"""Actual full-vs-interrupted CUDA BC optimizer/RNG continuation proof."""
import argparse
from pathlib import Path
from ppo_combat import torch, training_devices
from combat_checkpoint_scope import require_checkpoint_scope
from train_combat_visible_balanced_motor_bc import SCOPE, CHECKPOINT
from evaluate_registered_combat_models import read, save, sha


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert left.is_cuda and right.is_cuda
        assert torch.equal(left, right)
        return 1
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        return sum(same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        assert type(left) == type(right) and len(left) == len(right)
        return sum(same(x, y) for x, y in zip(left, right))
    assert left == right
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--full', required=True, type=Path)
    ap.add_argument('--resumed', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    assert not a.out.exists()
    training_devices()
    full = torch.load(a.full/'checkpoint.pt', map_location='cuda', weights_only=True)
    resumed = torch.load(a.resumed/'checkpoint.pt', map_location='cuda', weights_only=True)
    for cp, folder in ((full, a.full), (resumed, a.resumed)):
        require_checkpoint_scope(cp, SCOPE)
        assert cp['version'] == CHECKPOINT and cp['completed_steps'] == 150
        assert cp['weights_sha256'] == sha(folder/'weights.json')
    assert read(a.resumed/'report.json')['resumed_steps'] == 75
    tensors = same(full, resumed)
    assert sha(a.full/'weights.json') == sha(a.resumed/'weights.json')
    rejected = []
    for scope in (None, 'combat_movement_demonstration_bc_v1', 'combat_memory_route_ppo_v1'):
        try:
            require_checkpoint_scope(full, scope)
        except ValueError:
            rejected.append(scope)
        else:
            raise AssertionError('Checkpoint admitted to wrong trainer')
    save(a.out, dict(version='combat_visible_motor_bc_resume_cuda_audit_v1', state='passed',
         device=torch.cuda.get_device_name(), compared_tensors=tensors, full_and_resumed_steps=150,
         split_steps=[75, 75], exported_weights_identical=True, optimizer_and_rng_exact=True,
         rejected_scopes=rejected, training_performed=False, promotion=False,
         inputs={str((folder/name).resolve()): sha(folder/name)
                 for folder in (a.full, a.resumed) for name in ('weights.json', 'checkpoint.pt', 'report.json')}))
    print(dict(compared_tensors=tensors, weights_identical=True, optimizer_and_rng_exact=True), flush=True)


if __name__ == '__main__':
    main()

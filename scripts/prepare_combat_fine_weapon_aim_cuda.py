"""Zero-initialize optional SG fine branch and verify CUDA means/gate isolation."""
import argparse
import copy
from pathlib import Path
from ppo_combat import torch, training_devices
from combat_fine_weapon_aim import initialize, active, apply
from combat_joint_aim_policy import base_model
from migrate_combat_navigation_cuda import branch
from combat_precision_head import selected_means
from train_combat_stationary_shotgun_aim_cuda import load
from evaluate_registered_combat_models import read, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args(); assert not a.out.exists()
    training_devices(); torch.set_default_device('cuda'); torch.set_num_threads(2); torch.manual_seed(20261025)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    original = read(a.model); prepared = initialize(original)
    before = branch(base_model(original, allow_shared=True), 'actor').eval().requires_grad_(False)
    after = branch(base_model(prepared, allow_shared=True), 'actor').eval().requires_grad_(False)
    rows, gated = 0, 0
    report = read(a.data)
    with torch.no_grad():
        for split in ('train', 'diagnostic'):
            for e in load(report, split, before):
                f, raw, z, t, m, v = (e[k] for k in ('features', 'raw', 'movement', 'target', 'mode', 'vertical'))
                left = selected_means(raw, t, m, f, z, before.sampled_motion_aim_branch, v, before.motor_branch)
                right = selected_means(raw, t, m, f, z, after.sampled_motion_aim_branch, v, after.motor_branch)
                assert torch.equal(left, right)
                gate = active(f, t, m); rows += len(f); gated += int(gate.sum())
                # Nonzero head: off-gate preservation must be structural, too.
                head = copy.deepcopy(after.fine_weapon_aim_branch)
                head[-1].bias.fill_(.5)
                changed = apply(head, f, raw, z, t, m, left[:, 2:], before.motor_branch)
                assert torch.equal(changed[~gate], left[~gate, 2:])
                if bool(gate.any()):
                    torch.testing.assert_close(changed[gate], left[gate, 2:]+.5, atol=0, rtol=0)
                # The branch must not be registered/executed inside Sequential.
                assert len(after.sampled_motion_aim_branch) == len(before.sampled_motion_aim_branch)
                assert not any('fine_weapon' in k for k in after.sampled_motion_aim_branch.state_dict())
    assert gated and rows > gated
    a.out.mkdir(parents=True); save(a.out/'weights.json', prepared)
    save(a.out/'report.json', dict(version='combat_fine_weapon_aim_zero_cuda_v1', state='passed',
         device=torch.cuda.get_device_name(), rows=rows, gated_rows=gated, zero_means_exact=True,
         nonzero_outside_gate_exact=True, source_model=str(a.model.resolve()), source_sha256=sha(a.model),
         weights=str((a.out/'weights.json').resolve()), weights_sha256=sha(a.out/'weights.json'),
         inputs={str(a.data.resolve()): sha(a.data)}, training_performed=False, promotion=False))
    print(dict(rows=rows, gated_rows=gated, zero_means_exact=True, outside_gate_exact=True), flush=True)


if __name__ == '__main__':
    main()

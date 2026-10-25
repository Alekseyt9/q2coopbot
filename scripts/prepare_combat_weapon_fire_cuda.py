"""Prepare factual SG hit labels and zero-residual CUDA identity evidence."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import numpy as np
from ppo_combat import torch, training_devices
from ppo_recurrent import prepare_context, sequence
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model
from combat_conditional_fire import inputs, logits
from combat_weapon_fire_residual import initialize, active, apply, VERSION
from evaluate_registered_combat_models import read, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--corpus', required=True, type=Path, nargs='+')
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(); assert not args.out.exists()
    training_devices(); torch.set_default_device('cuda'); torch.set_num_threads(2); torch.manual_seed(20261025)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    original = read(args.model); zero = initialize(original)
    actor = branch(base_model(original, allow_shared=True), 'actor').eval().requires_grad_(False)
    migrated = branch(base_model(zero, allow_shared=True), 'actor').eval().requires_grad_(False)
    assert all(p.is_cuda for p in actor.parameters())
    xs, ys, baselines, splits, shot_ids, worlds = [], [], [], [], [], []
    pins = {str(args.model.resolve()): sha(args.model)}
    zero_rows = off_gate_rows = labels = 0
    with torch.no_grad():
        for root in args.corpus:
            report = read(root/'report.json')
            assert report['state'] == 'passed' and read(root/'independent-readback.json')['state'] == 'passed'
            pins[str((root/'report.json').resolve())] = sha(root/'report.json')
            for e in report['entries']:
                for key in ('pack', 'labels', 'compressed_native', 'compressed_sequence', 'verification'):
                    assert sha(e[key]) == e[key+'_sha256']
                proof = read(e['verification'])['cuda_verification']
                assert proof['state'] == 'passed' and proof['device'] == 'cuda' and proof['model_sha256'] == e['source_model_sha256']
                def load(name):
                    with gzip.open(e[name], 'rt', encoding='utf-8-sig') as f:
                        return [json.loads(line) for line in f]
                history, native, shots = load('compressed_sequence'), load('compressed_native'), load('labels')
                native_by_frame = {s['observation']['identity']['frame']: s for s in native}
                with np.load(e['pack'], allow_pickle=False) as pack:
                    frame_order = list(map(int, pack['frames']))
                    cached_features = pack['inputs_features'].copy()
                by_frame = {h['frame']: h for h in history}
                rows = []
                for frame in frame_order:
                    context = by_frame[frame]; step = native_by_frame[frame]
                    assert step['owner'] == 'provider'
                    rows.append(dict(seed=context['seed'], index=context['index'], frame=frame,
                                     features=context['features'], sample=step['sample']))
                assert np.array_equal(cached_features, np.asarray([r['features'] for r in rows], dtype=np.float32))
                # Evaluate the frozen parent on full observed histories, including
                # off-policy controller histories. No PPO/on-policy claim.
                raw, _ = sequence(actor, prepare_context(history, rows, 'cuda'), 32, False, 'actor')
                f = torch.tensor([r['features'] for r in rows], device='cuda', dtype=torch.float32)
                samples = [r['sample'] for r in rows]
                category = lambda key: torch.tensor([s.get(key, 0) for s in samples], device='cuda', dtype=torch.long)
                target, mode, vertical, weapon = (category(k) for k in ('target', 'aim_mode', 'vertical', 'weapon'))
                latent = torch.tensor([s['latent'] for s in samples], device='cuda', dtype=torch.float32)
                x = inputs(f, target, mode, latent, vertical, weapon)
                old = logits(actor.conditional_fire_branch, f, raw, target, mode, latent, vertical, weapon)
                unchanged = logits(migrated.conditional_fire_branch, f, raw, target, mode, latent, vertical, weapon)
                assert torch.equal(old, unchanged)
                assert len(actor.conditional_fire_branch) == len(migrated.conditional_fire_branch)
                assert not any('weapon_fire' in k for k in migrated.conditional_fire_branch.state_dict())
                # Structural isolation is checked with a deliberately nonzero
                # head and synthetic other-weapon/dead-player observations.
                probe = copy.deepcopy(migrated.weapon_fire_residual_branch)
                probe[-1].bias.fill_(.5)
                for variant in ('other_weapon', 'dead_player'):
                    altered = f.clone()
                    altered[:, 1123 if variant == 'other_weapon' else 0] = 0
                    assert not bool(active(altered).any())
                    assert torch.equal(apply(probe, altered, x, old), old)
                    off_gate_rows += len(f)
                y = np.full(len(rows), -1, dtype=np.float32)
                for shot in shots:
                    assert shot['policy_label_eligible']
                    index = shot['input_row']
                    assert frame_order[index] == shot['window']['begin_frame'] and y[index] == -1
                    y[index] = float(shot['direct_monster_damage'] > 0)
                    assert bool(active(f[index:index+1]).all())
                    labels += 1
                xs.append(x.cpu().numpy()); ys.append(y); baselines.append(old.cpu().numpy())
                splits.extend([e['split']] * len(rows)); worlds.extend([f"{e['map']}:{e['seed']}"] * len(rows))
                shot_ids.extend([f"{root.name}/{e['index']}:{frame}" for frame in frame_order])
                zero_rows += len(rows)
                if len(xs) % 16 == 0:
                    print(dict(cases=len(xs), zero_rows=zero_rows, labels=labels), flush=True)
    train_worlds = {w for w, s in zip(worlds, splits) if s == 'train'}
    diagnostic_worlds = {w for w, s in zip(worlds, splits) if s == 'diagnostic'}
    assert train_worlds and diagnostic_worlds and not train_worlds & diagnostic_worlds
    args.out.mkdir(parents=True)
    np.savez_compressed(args.out/'data.npz', x=np.concatenate(xs), y=np.concatenate(ys), legacy=np.concatenate(baselines),
                        train=np.asarray([s == 'train' for s in splits]), world=np.asarray(worlds), sample_id=np.asarray(shot_ids))
    save(args.out/'weights.json', zero)
    save(args.out/'report.json', dict(version='combat_sg_fire_outcome_data_v1', state='passed', device='cuda',
         cases=len(xs), rows=zero_rows, labelled_triggers=labels, zero_logits_exact=True,
         nonzero_off_gate_rows=off_gate_rows, split_worlds=dict(train=len(train_worlds), diagnostic=len(diagnostic_worlds)),
         inputs=pins, data_sha256=sha(args.out/'data.npz'), source_model=str(args.model.resolve()), source_sha256=sha(args.model),
         weights=str((args.out/'weights.json').resolve()), weights_sha256=sha(args.out/'weights.json'),
         residual_version=VERSION, script_sha256=sha(__file__), training_performed=False, promotion=False,
         scope='Factual actual SG hit/miss labels; unexecuted decisions have label-1, never invented misses. Current observation and sampled action181inputs only. Source controller histories may differ. No future/hidden label as input; not ordinary PPO data.'))
    (args.out/'preparer-source.py').write_bytes(Path(__file__).read_bytes())
    print(dict(zero_rows=zero_rows, labels=labels, off_gate_rows=off_gate_rows), flush=True)


if __name__ == '__main__':
    main()

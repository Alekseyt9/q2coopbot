"""Decompose decision-scope updates on their own TRAIN histories, CUDA only."""
import argparse
import json
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha
from train_combat_bc import torch, training_devices
from ppo_recurrent import prepare_context, sequence
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model, events, probabilities
from combat_geometry_ppo import GeometryJoint
from combat_precision_head import selected_means, targets, mode_distribution
from combat_weapon_head import distribution as weapons, feature_mask
from combat_conditional_fire import logits as fire_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    training_devices()
    torch.set_default_device('cuda')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    pins, cases = {}, []

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    pin(__file__)
    matched = read(pin(args.workflow / 'matched-update-counts.json'))
    assert matched['state'] == 'matched' and matched['new_updates_per_arm'] == 1 and matched['cases_per_arm'] == 24
    for label in ('trio', 'balanced'):
        for number in (1,):
            root = args.workflow / label
            seal = read(pin(root / 'update/complete.json'))
            update = read(pin(root / 'update/report.json', seal['report_sha256']))
            source = Path(update['source'])
            source_seal = read(pin(source / 'complete.json'))
            parent = read(pin(source / 'weights.json', source_seal['weights_sha256']))
            child = read(pin(root / 'update/weights.json', seal['weights_sha256']))
            assert source_seal['weights_sha256'] == update['behavior_sha256']
            corpus_seal = read(pin(root / 'corpus/complete.json'))
            protocol = read(pin(root / 'corpus/protocol.json', corpus_seal['protocol_sha256']))
            assert protocol['train_only'] and protocol['assignments'][0]['split'] == 'train'
            data = root / 'corpus/processing-train/merged'
            meta = read(pin(data / 'report.json'))
            assert meta['rollout_sha256'] == update['rollout_sha256']
            with pin(data / 'rollout.jsonl', meta['rollout_sha256']).open(encoding='utf-8-sig') as stream:
                rows = list(map(json.loads, stream))
            with pin(data / 'sequence.jsonl', meta['sequence_sha256']).open(encoding='utf-8-sig') as stream:
                context = list(map(json.loads, stream))
            assert len(rows) == update['rows']
            prepared = prepare_context(context, rows, 'cuda')
            x = torch.tensor([row['features'] for row in rows], device='cuda', dtype=torch.float32)
            category = lambda key: torch.tensor([row['sample'].get(key, 0) for row in rows], device='cuda', dtype=torch.long)
            target, mode, vertical, weapon = (category(key) for key in ('target', 'aim_mode', 'vertical', 'weapon'))
            final = torch.tensor([row['sample']['latent'] for row in rows], device='cuda', dtype=torch.float32)
            attack = torch.tensor([float(row['sample']['attack']) for row in rows], device='cuda')
            source_z, choice, timing, active = events(rows, x, target, mode, vertical, weapon)

            def evaluate(model, verify_memory):
                actor = branch(base_model(model, allow_shared=True), 'actor').eval()
                joint = GeometryJoint(model).eval()
                raw, memory_error = sequence(actor, prepared, 32, verify_memory, 'actor')
                std = torch.tensor(model['log_std'], device='cuda', dtype=raw.dtype)
                means = selected_means(raw, target, mode, x, source_z[:, :2], actor.sampled_motion_aim_branch,
                                       vertical, actor.motor_branch)
                d = torch.distributions
                distributions = dict(normal=d.Normal(means, std.exp()), target=targets(raw[:, :45], x),
                    mode=mode_distribution(raw, target), pose=d.Categorical(logits=raw[:, 5:8]),
                    weapon=weapons(raw[:, 8:20], feature_mask(x[:, :845])),
                    fire=d.Bernoulli(logits=fire_logits(actor.conditional_fire_branch, x, raw, target, mode, final, vertical, weapon)))
                lp, _, logits, current_active = probabilities(joint, raw, std, source_z, final, attack, vertical, weapon,
                    x, target, mode, choice, timing, actor.sampled_motion_aim_branch, actor.motor_branch, actor.conditional_fire_branch)
                assert torch.equal(current_active, active)
                distributions['geometry'] = d.Categorical(logits=logits)
                normal_lp = distributions['normal'].log_prob(source_z).sum(-1)
                summed = (normal_lp + distributions['target'].log_prob(target) + distributions['mode'].log_prob(mode)
                    + distributions['pose'].log_prob(vertical) + distributions['weapon'].log_prob(weapon)
                    + distributions['fire'].log_prob(attack)
                    + torch.where(active, distributions['geometry'].log_prob(choice), torch.zeros_like(lp)))
                error = float((summed-lp).abs().max())
                assert error < 1e-4
                return distributions, lp, error, memory_error

            with torch.no_grad():
                before, old_lp, old_sum_error, memory_error = evaluate(parent, True)
                after, new_lp, new_sum_error, _ = evaluate(child, False)
                native = torch.tensor([row['sample']['log_probability'] for row in rows], device='cuda')
                replay = float((old_lp-native).abs().max())
                assert replay < .003 and memory_error < 2e-5
                kl = {name: torch.distributions.kl_divergence(before[name], after[name]) for name in before}
                assert all(bool(torch.isfinite(value).all()) for value in kl.values())
                assert all(bool((torch.distributions.kl_divergence(value, value) == 0).all()) for value in before.values())
                coordinate = kl.pop('normal').mean(0)
                factors = dict(movement=float(coordinate[:2].sum()), conditional_aim=float(coordinate[2:].sum()),
                    **{name:float(value[active].mean() if name=='geometry' else value.mean()) for name,value in kl.items()})
                delta = new_lp-old_lp
                full_sample_k3 = float((delta.exp()-1-delta).mean())
                assert abs(full_sample_k3-update['full_sample_k3']) < 5e-5
                cases.append(dict(branch=label, round=number, rows=len(rows), conditional_factor_kl=factors,
                    gaussian_coordinate_kl=coordinate.tolist(), old_log_std=parent['log_std'], new_log_std=child['log_std'],
                    source_lp_native_max_error=replay, factor_sum_max_error=max(old_sum_error,new_sum_error),
                    full_sample_k3=full_sample_k3, trainer_full_sample_k3=update['full_sample_k3']))
    import sys
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and Path(path).resolve().parent == Path(__file__).resolve().parent and Path(path).suffix == '.py':
            pin(path)
    assert all(sha(path) == digest for path,digest in pins.items())
    args.out.mkdir(parents=True)
    save(args.out / 'report.json', dict(state='complete', version='combat_decision_factor_change_audit_v1', device='cuda',
        cases=cases, inputs=pins, training_performed=False, new_games=False, weights_written=False,
        scope='Read-only decomposition on each update own TRAIN histories. Conditional KL averages use recorded upstream samples; '
              'geometry is averaged only on active rows. They are not an exact additive decomposition of sampled joint K3, '
              'nor a causal explanation of the optimizer stopping or battle outcomes. No fresh PPO update or checkpoint selection.'))
    save(args.out / 'complete.json', dict(state='complete', report_sha256=sha(args.out / 'report.json')))
    print(cases)


if __name__ == '__main__':
    main()

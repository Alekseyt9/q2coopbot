"""Check complete temporal actors and joint target densities on real source TRAIN histories."""
import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha
from ppo_combat import torch, training_devices
from ppo_recurrent import prepare_context, sequence
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import JointAim, base_model, events, probabilities
from combat_target_sampling import VERSION, correction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'corpus', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    args.out.mkdir(parents=True)
    pins, live = {}, {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = pins.get(str(path)) or sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    def snapshot():
        scripts = Path(__file__).resolve().parent
        for module in list(sys.modules.values()):
            name = getattr(module, '__file__', None)
            if not name:
                continue
            path = Path(name).resolve()
            if path.parent != scripts or path.suffix != '.py' or str(path) in live:
                continue
            target = args.out / 'source' / path.name
            target.parent.mkdir(exist_ok=True)
            digest = sha(path)
            shutil.copy2(path, target)
            pin(target, digest)
            live[str(path)] = digest

    protocol = read(pin(args.protocol))
    assert protocol['version'] == 'combat_target_temperature_calibration_v1'
    source = Path(protocol['source']).resolve()
    seal = read(pin(source / 'complete.json'))
    model = read(pin(source / 'weights.json', seal['weights_sha256']))
    pin(source / 'report.json', seal['report_sha256'])
    assert seal['weights_sha256'] == protocol['source_weights_sha256'] and not model.get('target_sampling')
    closed = read(pin(args.corpus / 'complete.json'))
    collection = read(pin(args.corpus / 'protocol.json', closed['protocol_sha256']))
    assert closed['native_cases'] == closed['cuda_cases'] == 32 and collection['train_only']
    assert Path(collection['source']).resolve() == source
    pool = read(pin(args.corpus / 'pool/report.json', closed['pool_sha256']))
    assert pool['state'] == 'complete' and pool['source_unchanged']
    training_devices()
    torch.set_default_device('cuda')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    definitions = dict(source=model)
    models = {}
    for label, temperature in (('zero', 1.), ('calibrated', protocol['temperature'])):
        changed = copy.deepcopy(model)
        changed['target_sampling'] = dict(version=VERSION, temperature=temperature)
        unchanged = copy.deepcopy(changed)
        unchanged.pop('target_sampling')
        assert unchanged == model
        directory = args.out / label
        directory.mkdir()
        save(directory / 'weights.json', changed)
        definitions[label] = changed
        models[label] = dict(path=str((directory / 'weights.json').resolve()), sha256=sha(directory / 'weights.json'))
    actors = {k: branch(base_model(v, allow_shared=True, allow_target_sampling=True), 'actor').requires_grad_(False)
              for k, v in definitions.items()}
    joints = {k: JointAim(v, allow_shared=True, allow_target_sampling=True).requires_grad_(False)
              for k, v in definitions.items()}
    reference = copy.deepcopy(actors['source'].state_dict())
    assert all(all(torch.equal(t, actor.state_dict()[key]) for key, t in reference.items()) for actor in actors.values())
    snapshot()
    rows, lp_max, density_max = 0, 0., 0.
    for index, job in enumerate(sorted(pool['jobs'], key=lambda j: j['job'])):
        assert job['mode'] == 'learned'
        folder = args.corpus / 'processing-train' / f"case-{job['job']:04d}" / 'cuda'
        meta = read(pin(folder / 'report.json'))
        assert meta['model_sha256'] == protocol['source_weights_sha256'] and meta['numerical_verification'] == 'cuda_verified_v1'
        proof = read(pin(folder / 'cuda-verification.json', meta['cuda_verification_sha256']))
        assert proof['state'] == 'passed' and proof['device'] == 'cuda'
        with pin(folder / 'rollout.jsonl', meta['rollout_sha256']).open(encoding='utf-8-sig') as stream:
            recorded = list(map(json.loads, stream))
        with pin(folder / 'sequence.jsonl', meta['sequence_sha256']).open(encoding='utf-8-sig') as stream:
            history = list(map(json.loads, stream))
        x = torch.tensor([r['features'] for r in recorded], device='cuda')
        cat = lambda key: torch.tensor([r['sample'].get(key, 0) for r in recorded], device='cuda', dtype=torch.long)
        target, mode, vertical, weapon = (cat(k) for k in ('target', 'aim_mode', 'vertical', 'weapon'))
        final = torch.tensor([r['sample']['latent'] for r in recorded], device='cuda')
        attack = torch.tensor([float(r['sample']['attack']) for r in recorded], device='cuda')
        with torch.no_grad():
            context = prepare_context(history, recorded, 'cuda')
            outputs = {}
            for label, actor in actors.items():
                raw, error = sequence(actor, context, 32, True, 'actor')
                assert error < 2e-5
                outputs[label] = raw
            raw = outputs['source']
            assert torch.equal(raw, outputs['zero']) and torch.equal(raw, outputs['calibrated'])
            source_z, choice, timing, _ = events(recorded, x, target, mode, vertical, weapon)
            before_raw = raw.clone()
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
            results = {}
            for label, actor in actors.items():
                results[label] = probabilities(joints[label], outputs[label], torch.tensor(model['log_std'], device='cuda'),
                    source_z, final, attack, vertical, weapon, x, target, mode, choice, timing,
                    actor.sampled_motion_aim_branch, actor.motor_branch, actor.conditional_fire_branch)
            assert torch.equal(cpu_rng, torch.get_rng_state())
            assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
            assert torch.equal(before_raw, raw) and torch.equal(results['source'][0], results['zero'][0])
            assert torch.equal(results['source'][1], results['zero'][1])
            assert torch.equal(results['source'][2], results['calibrated'][2]) and torch.equal(results['source'][3], results['calibrated'][3])
            delta, entropy_delta = correction(raw, x, target, definitions['calibrated']['target_sampling'])
            error = float(((results['calibrated'][0] - results['source'][0]) - delta).abs().max())
            assert error < .003
            assert float(((results['calibrated'][1] - results['source'][1]) - entropy_delta).abs().max()) < 1e-4
            density_max = max(density_max, error)
            error = float((results['source'][0] - torch.tensor([r['sample']['log_probability'] for r in recorded], device='cuda')).abs().max())
            assert error < .003
            lp_max = max(lp_max, error)
        rows += len(recorded)
        save(args.out / 'progress.json', dict(stage='cuda-target-factor-proof', cases_done=index + 1, cases=32))
    snapshot()
    for path, digest in {**pins, **live}.items():
        assert sha(path) == digest, path
    report = dict(state='complete', version='combat_target_sampling_cuda_preparation_v1', device='cuda',
        source=str(source), source_weights_sha256=protocol['source_weights_sha256'], corpus=str(args.corpus.resolve()),
        models=models, source_parameters_exact=True, zero_source_raw_exact=True, conditional_other_factors_exact=True,
        zero_source_full_density_exact=True, zero_source_full_entropy_exact=True, no_density_rng_draws=True,
        source_lp_max_error=lp_max, target_density_delta_max_error=density_max, rows=rows, cases=32, inputs=pins,
        training_performed=False, ppo_checkpoint_written=False, native_compatibility_proven=False, promotion=False)
    save(args.out / 'report.json', report)
    save(args.out / 'complete.json', dict(state='complete', report_sha256=sha(args.out / 'report.json')))
    save(args.out / 'progress.json', dict(stage='complete', rows=rows))
    print(dict(rows=rows, source_lp_error=lp_max, density_delta_error=density_max))


if __name__ == '__main__':
    main()

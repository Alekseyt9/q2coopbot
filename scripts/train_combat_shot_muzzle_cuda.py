"""Offline shot-muzzle forward-model experiment; never a runtime aim policy."""
import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np

SCOPE = 'combat_shot_muzzle_forward_aux_v1'
CONFIG = dict(seed=20261025, planned_steps=500, learning_rate=.002, hidden=[64, 32],
              output_scale_units=32, checkpoint_selection='fixed_final_step',
              split='corpus_world_split', action_ablation='identical_capacity_zero_action_inputs',
              unknown_range_sentinel=-1)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def local(vector, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    x, y, z = vector
    return [c*x+s*y, -s*x+c*y, z]


def observed_range(value, scale):
    # Physical distances are nonnegative; -1 encodes unknown without adding
    # inputs or changing any previously valid value in the fixed-width model.
    if value is None:
        return -1.
    assert math.isfinite(value) and value >= 0
    return value/scale


def state_inputs(observation):
    """Only the current client observation; no map, world IDs or absolute position."""
    o = observation
    pitch, yaw = (float(v)*2*math.pi/65536 for v in o['view_angles'][:2])
    previous = o['previous_applied_command']
    geometry = o['local_geometry']
    x = [v/400 for v in local(o['velocity'], yaw)]
    x += [float(o['on_ground']), float(o['ducked']), math.sin(pitch), math.cos(pitch), o['gun_frame']/20]
    x += [previous[k]/400 for k in ['Forward', 'Side', 'Up']]
    x += [v/15 for v in o['kick_angles_degrees']]
    drop = geometry['ground_drop']
    x += [0 if drop is None else drop/64, float(drop is not None), observed_range(geometry['up_eye_distance'], 64)]
    assert len(geometry['probes']) == 8
    for probe in geometry['probes']:
        drop = probe['ground_drop_at_40']
        x += [observed_range(probe['eye_ray_distance'], 256), observed_range(probe['standing_hull_clearance'], 64),
              0 if drop is None else drop/64, float(drop is not None)]
    assert len(x) == 49 and all(math.isfinite(v) for v in x)
    return x


def action_inputs(action):
    yaw = math.radians(action['yaw_delta_degrees'])
    return [action['forward'], action['side'], math.sin(yaw), math.cos(yaw),
            action['pitch_delta_degrees']/180, float(action['attack'])] + [
                float(action['vertical'] == name) for name in ['release', 'jump', 'crouch']]


def muzzle_label(observation, shot):
    point = [float(v) for v in shot['begin']['start'].split(',')]
    yaw = float(observation['view_angles'][1])*2*math.pi/65536
    return local([a-b for a, b in zip(point, observation['position'])], yaw)


def load_corpus(root):
    report = json.loads((root/'report.json').read_text())
    assert report['state'] == 'passed' and report['version'] == 'combat_sg_shot_time_corpus_v1'
    assert json.loads((root/'independent-readback.json').read_text())['state'] == 'passed'
    rows = []
    worlds = {'train': set(), 'diagnostic': set()}
    for entry in report['entries']:
        for key in ['compressed_native', 'labels', 'pack', 'verification']:
            assert sha(entry[key]) == entry[key+'_sha256']
        with gzip.open(entry['compressed_native'], 'rt', encoding='utf-8-sig') as stream:
            steps = [json.loads(line) for line in stream]
        with gzip.open(entry['labels'], 'rt') as stream:
            shots = [json.loads(line) for line in stream]
        with np.load(entry['pack'], allow_pickle=False) as pack:
            frames = pack['frames'].copy()
        for shot in shots:
            assert shot['policy_label_eligible'] and shot['input_row'] is not None
            step = steps[shot['native_trace_row']]
            o, action = step['observation'], step['action']
            assert o['identity']['frame'] == shot['window']['begin_frame'] == int(frames[shot['input_row']])
            assert step['native_step']['sequence'] == shot['window']['sequence']
            yaw = float(o['view_angles'][1])*2*math.pi/65536
            # Diagnostic simple baseline, not runtime motor/aim intervention.
            baseline = [v*.1 for v in local(o['velocity'], yaw)]
            baseline[2] += -10 if o['ducked'] else 14
            rows.append(dict(x=state_inputs(o)+action_inputs(action), y=muzzle_label(o, shot), baseline=baseline,
                split=entry['split'], world=f"{entry['map']}:{entry['seed']}", case=entry['index'],
                vertical=action['vertical'], phase=shot['native_phase_offset']))
        worlds[entry['split']].add((entry['map'], entry['seed']))
    assert not worlds['train'] & worlds['diagnostic'] and len(rows) == report['totals']['triggers']
    return rows


def training_extension(original, extra):
    protected = {r['world'] for r in original if r['split'] == 'diagnostic'}
    known = {r['world'] for r in original}
    assert extra and all(r['split'] == 'train' for r in extra), 'Extension must be entirely TRAIN'
    assert not protected & {r['world'] for r in extra}, 'Diagnostic worlds cannot become training examples'
    assert not known & {r['world'] for r in extra}, 'Extension worlds must be independent of existing worlds'
    return original+extra


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--resume', type=Path)
    ap.add_argument('--extra-train-corpus', type=Path, action='append', default=[])
    args = ap.parse_args()
    assert not args.out.exists() and 0 < args.steps <= CONFIG['planned_steps']
    # Shared F: cache setup and CUDA-only admission precede neural construction.
    from ppo_combat import torch, nn, training_devices
    from combat_checkpoint_scope import require_checkpoint_scope
    training_devices()
    torch.set_default_device('cuda'); torch.set_num_threads(2); torch.manual_seed(CONFIG['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = load_corpus(args.corpus)
    extension_pins = {}
    for extension in args.extra_train_corpus:
        rows = training_extension(rows, load_corpus(extension))
        extension_pins[str(extension.resolve()/'report.json')] = sha(extension/'report.json')
    x = torch.tensor([r['x'] for r in rows], dtype=torch.float32, device='cuda')
    y = torch.tensor([r['y'] for r in rows], dtype=torch.float32, device='cuda')
    baseline = torch.tensor([r['baseline'] for r in rows], dtype=torch.float32, device='cuda')
    train = torch.tensor([r['split'] == 'train' for r in rows], device='cuda')
    center = x[train].mean(0); scale = x[train].std(0).clamp_min(.05)
    normalized = (x-center)/scale
    blind = normalized.clone(); blind[:, 49:] = 0
    initial = nn.Sequential(nn.Linear(58, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))
    models = {name: copy.deepcopy(initial) for name in ['action', 'blind']}
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate']) for name, model in models.items()}
    inputs = {'action': normalized, 'blind': blind}
    pin = sha(args.corpus/'report.json'); script_sha = sha(__file__)
    completed = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cuda', weights_only=True)
        require_checkpoint_scope(checkpoint, SCOPE)
        assert checkpoint['config'] == CONFIG and checkpoint['corpus_sha256'] == pin and checkpoint['script_sha256'] == script_sha
        assert checkpoint['extension_pins'] == extension_pins
        assert torch.equal(checkpoint['center'], center) and torch.equal(checkpoint['scale'], scale)
        completed = checkpoint['completed_steps']; assert 0 < completed < args.steps
        for name in models:
            models[name].load_state_dict(checkpoint['models'][name])
            optimizers[name].load_state_dict(copy.deepcopy(checkpoint['optimizers'][name]))
    assert all(p.is_cuda for model in models.values() for p in model.parameters())
    for step in range(completed, args.steps):
        for name, model in models.items():
            prediction = model(inputs[name][train])
            loss = (prediction-y[train]/32).square().mean()
            assert torch.isfinite(loss)
            optimizers[name].zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizers[name].step()
    def metrics(prediction, mask):
        error = prediction[mask]-y[mask]
        return dict(rows=int(mask.sum()), xyz_mae_units=error.abs().mean(0).tolist(),
                    mean_position_error_units=float(error.norm(dim=1).mean()))
    with torch.no_grad():
        predictions = {name: model(inputs[name])*32 for name, model in models.items()}
        predictions.update(constant_velocity_baseline=baseline, train_mean_baseline=y[train].mean(0).expand_as(y))
        results = {}
        for name, prediction in predictions.items():
            results[name] = {}
            for split, mask in [('train', train), ('diagnostic', ~train)]:
                groups = {'all': mask}
                for vertical in ['release', 'jump', 'crouch']:
                    groups[vertical] = mask & torch.tensor([r['vertical'] == vertical for r in rows], device='cuda')
                groups.update({f'phase_{phase}': mask & torch.tensor([r['phase'] == phase for r in rows], device='cuda') for phase in [0, 1]})
                results[name][split] = {k: metrics(prediction, m) for k, m in groups.items() if bool(m.any())}
                world_errors = []
                for world in sorted({r['world'] for r in rows if r['split'] == split}):
                    m = mask & torch.tensor([r['world'] == world for r in rows], device='cuda')
                    world_errors.append(metrics(prediction, m)['mean_position_error_units'])
                results[name][split]['world_balanced_position_error_units'] = sum(world_errors)/len(world_errors)
    args.out.mkdir(parents=True)
    checkpoint = dict(checkpoint_training_scope=SCOPE, version=SCOPE, config=CONFIG, completed_steps=args.steps,
        corpus_sha256=pin, script_sha256=script_sha, center=center, scale=scale,
        extension_pins=extension_pins,
        models={n:m.state_dict() for n,m in models.items()}, optimizers={n:o.state_dict() for n,o in optimizers.items()})
    torch.save(checkpoint, args.out/'checkpoint.pt')
    report = dict(version=SCOPE, state='passed', device='cuda', config=CONFIG, completed_steps=args.steps,
        corpus_sha256=pin, script_sha256=script_sha, checkpoint_sha256=sha(args.out/'checkpoint.pt'),
        cases=len({(r['world'], r['case']) for r in rows}), triggers=len(rows), input_features=58,
        extension_pins=extension_pins, split_worlds={s: len({r['world'] for r in rows if r['split'] == s}) for s in ['train', 'diagnostic']},
        neural_parameters_per_model=sum(p.numel() for p in initial.parameters()), metrics=results,
        runtime_policy_changed=False, promotion=False,
        scope='Offline forward-model diagnostic only. Inputs=current client state and proposed action; labels=actual shot muzzle. No hidden/future inputs. Paired controllers share world split. Four diagnostic seeds on TRAIN maps; no independent combat-quality or unseen-map evidence.')
    (args.out/'report.json').write_text(json.dumps(report, indent=2))
    (args.out/'trainer-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

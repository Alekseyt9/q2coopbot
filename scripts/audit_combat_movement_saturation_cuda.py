"""Read-only CUDA audit of Gaussian movement and native command quantization."""
import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path

from combat_full_joint_ppo import FullJointScope
from ppo_combat import torch, training_devices
from ppo_recurrent import prepare_context, sequence
from process_combat_architecture_pool import read, save, sha


def rows(path):
    with path.open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def run(screen, config, out):
    assert not out.exists()
    training_devices()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tasks = read(screen/'cuda-request.json')['tasks'][:8]
    model_path = Path(tasks[0]['model'])
    model = read(model_path)
    assert not model.get('wide_aim')
    scope = FullJointScope(model, read(config))
    pins = {str(model_path): sha(model_path), str(config): sha(config),
            str(Path(__file__).resolve()): sha(Path(__file__))}
    cases = []
    threshold = math.atanh(399.5/400)
    with torch.no_grad():
        std = scope.std[:2].exp().double()
        for task in tasks:
            data, native = Path(task['out']), Path(task['data'])
            proof = read(data/'cuda-verification.json')
            assert proof['state'] == 'passed' and proof['device'] == 'cuda'
            assert proof['model_sha256'] == sha(model_path) == task['model_sha256']
            for name in ('rollout.jsonl', 'sequence.jsonl'):
                pins[str(data/name)] = sha(data/name)
            rollout, history = rows(data/'rollout.jsonl'), rows(data/'sequence.jsonl')
            report = read(native/'report.json')
            step_path = next(Path(p) for p in report['source_sha256'] if Path(p).name == 'steps.jsonl')
            assert sha(step_path) == report['source_sha256'][str(step_path)]
            pins[str(step_path)] = sha(step_path)
            steps = {r['index']: r for r in rows(step_path)}
            raw, error = sequence(scope.actor, prepare_context(history, rollout, 'cuda'), 32, True, 'actor')
            assert error < 2e-5
            means = raw[:, :2].double()
            latent = torch.tensor([r['sample']['latent'][:2] for r in rollout], device='cuda', dtype=torch.float64)
            cdf = lambda z: .5*(1+torch.erf(z/math.sqrt(2)))
            endpoint = cdf((-threshold-means)/std) + 1-cdf((threshold-means)/std)
            requested = torch.sign(torch.tanh(latent))*torch.floor(torch.tanh(latent).abs()*400+.5)
            sent = torch.tensor([[steps[r['index']]['sent_command'][k] for k in ('Forward', 'Side')]
                                 for r in rollout], device='cuda', dtype=torch.float64)
            reasons = Counter(str(i) for r in rollout for i in (r.get('interventions') or []))
            changed = (requested != sent).any(-1)
            stops = []
            unsupported = Counter()
            longest = current = 0
            for i, row in enumerate(rollout):
                step = steps[row['index']]
                before, after = step['observation'], step['next_observation']
                zero = bool((sent[i] == 0).all())
                current = current+1 if zero else 0
                longest = max(longest, current)
                if zero:
                    stops.append(math.dist(before['position'][:2], after['position'][:2]))
                if any(v.get('reason') == 'unsupported_motion_guard' for v in (row.get('interventions') or [])):
                    unsupported['on_ground' if before['on_ground'] else 'airborne'] += 1
                    unsupported['requested_'+step['action']['vertical']] += 1
                    unsupported['observed_ducked' if before['ducked'] else 'observed_standing'] += 1
            case = dict(seed=rollout[0]['seed'], map=steps[rollout[0]['index']]['observation']['identity']['map'],
                rows=len(rollout), memory_max_error=error,
                mean_abs_p50=torch.quantile(means.abs(), .5, dim=0).cpu().tolist(),
                mean_abs_p95=torch.quantile(means.abs(), .95, dim=0).cpu().tolist(),
                endpoint_mass_gt99=(endpoint > .99).double().mean(0).cpu().tolist(),
                sample_endpoint_fraction=(requested.abs() == 400).double().mean(0).cpu().tolist(),
                requested_vs_sent_changed_rows=int(changed.sum()),
                requested_nonzero_sent_zero_rows=int(((requested != 0).any(-1) & (sent == 0).all(-1)).sum()),
                sent_zero_rows=int((sent == 0).all(-1).sum()), interventions=dict(reasons))
            case.update(unsupported_motion_state_counts=dict(unsupported),
                longest_consecutive_zero_command_rows=longest,
                zero_command_actual_xy_distance=sum(stops),
                zero_command_moved_gt1unit_rows=sum(d > 1 for d in stops))
            cases.append(case)
    assert all(sha(Path(p)) == digest for p, digest in pins.items())
    result = dict(state='complete', device='cuda', source_model_sha256=sha(model_path),
        scope='Existing eight validation worlds; read-only mechanism audit, not TRAIN or causal quality evidence.',
        endpoint_latent_threshold=threshold, movement_std=std.cpu().tolist(), cases=cases,
        total_rows=sum(c['rows'] for c in cases), source_sha256=pins, promotion=False)
    out.mkdir(parents=True)
    shutil.copyfile(__file__, out/Path(__file__).name)
    save(out/'report.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--screen', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    run(args.screen, args.config, args.out)

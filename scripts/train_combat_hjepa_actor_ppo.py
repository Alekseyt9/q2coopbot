"""Float64 CUDA primitive PPO on source-bound native H-JEPA rollouts.

Only encoder0, actor and Gaussian log_std change. Weapon and auxiliary world
heads stay frozen. Native outcomes supply labels, never observation features.
"""
import argparse
import collections
import copy
import hashlib
import json
import math
import pathlib

import torch
from torch import nn
from torch.nn import functional as F

from train_combat_hjepa import export, save, sha
from hjepa_weapon_returns import episode_ledger, at_frame

VERSION = 'hjepa_primitive_ppo_v1'
ALGORITHM = dict(learning_rate=1e-4, clip=.1, gradient_clip=1., max_updates=4,
                 max_mean_kl=.02, backtrack_power=12, advantage_scale=100,
                 credit='first_life_native_rtg_other_stream_time_baseline_v1')


def network(layers):
    modules = []
    for i, layer in enumerate(layers):
        weight = torch.tensor(layer['weight'], dtype=torch.float64)
        linear = nn.Linear(weight.shape[1], weight.shape[0], dtype=torch.float64)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(torch.tensor(layer['bias'], dtype=torch.float64))
        modules.append(linear)
        if i + 1 < len(layers):
            modules.append(nn.ReLU())
    return nn.Sequential(*modules)


class Actor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = network(model['levels'][0]['encoder'])
        self.actor = network(model['actor'])
        self.log_std = nn.Parameter(torch.tensor(model['primitive_sampling']['log_std'], dtype=torch.float64))
        self.temperature = model['primitive_sampling']['categorical_temperature']

    def forward(self, features):
        return self.actor(self.encoder(features))


def log_probability(raw, log_std, latent, attack, vertical, temperature):
    normal = (-.5 * ((latent - raw[:, :4]) / log_std.exp()).square()
              - log_std - .5 * math.log(2 * math.pi)).sum(-1)
    fire = raw[:, 4] / temperature
    pose = F.log_softmax(raw[:, 5:] / temperature, -1)
    return normal + F.logsigmoid(torch.where(attack, fire, -fire)) + pose.gather(1, vertical[:, None])[:, 0]


def distribution_kl(old, old_std, new, new_std, temperature):
    """KL(old || new), before fixed tanh transform, summed over factors."""
    normal = (new_std - old_std + (old_std.mul(2).exp()
              + (old[:, :4] - new[:, :4]).square()) / (2 * new_std.mul(2).exp()) - .5).sum(-1)
    old_fire, new_fire = old[:, 4] / temperature, new[:, 4] / temperature
    p = old_fire.sigmoid()
    fire = p * (F.logsigmoid(old_fire) - F.logsigmoid(new_fire))
    fire += (1-p) * (F.logsigmoid(-old_fire) - F.logsigmoid(-new_fire))
    old_pose, new_pose = F.log_softmax(old[:, 5:] / temperature, -1), F.log_softmax(new[:, 5:] / temperature, -1)
    return normal + fire + (old_pose.exp() * (old_pose-new_pose)).sum(-1)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def eligible(row, ledger, life):
    return (row['exclusive'] and life['start_frame'] <= row['identity']['frame'] <= life['end_frame']
            and (ledger['terminal_step'] is None or row['step_index'] <= ledger['terminal_step']))


def load_rollouts(paths):
    rows, episodes, inputs, sources, streams = [], {}, {}, {}, set()
    parent, worlds = None, None
    for path in paths:
        mp = pathlib.Path(str(path)+'.meta.json')
        meta = json.loads(mp.read_text())
        if (meta['version'] != 'hjepa_onpolicy_actor_rollout_v1' or meta['split'] != 'train'
                or meta['distribution'] != 'hjepa_primitive_distribution_v1' or sha(path) != meta['data_sha256']):
            raise ValueError('typed unchanged TRAIN actor rollouts required')
        for name, digest in meta['sources_sha256'].items():
            if sha(pathlib.Path(name)) != digest:
                raise ValueError('changed native source: '+name)
            sources[name] = digest
        inputs[str(path.resolve())], inputs[str(mp.resolve())] = sha(path), sha(mp)
        model_path = pathlib.Path(meta['model'])
        if sha(model_path) != meta['model_sha256']:
            raise ValueError('changed behavior weights')
        model = json.loads(model_path.read_text())
        if model.get('motor') or model['planner'] != 'bc' or model.get('weapon_option_frames', 0):
            raise ValueError('shared frame-wise actor required')
        stream = model['primitive_sampling']['seed']
        if stream in streams:
            raise ValueError('distinct primitive streams required')
        streams.add(stream)
        comparable = copy.deepcopy(model)
        comparable['primitive_sampling']['seed'] = 0
        if parent is not None and comparable != parent:
            raise ValueError('behavior weights differ across streams')
        parent = comparable
        current = set()
        for ep in meta['episodes']:
            result, seed = ep['native_result'], ep['seed']
            if (seed in current or result['seed'] != seed or not result['capture_valid']
                    or result['generated_fixture']['split'] != 'train' or not result['generated_start']['confirmed']):
                raise ValueError('invalid native episode')
            current.add(seed)
            episodes[stream, seed] = dict(result=result, expected=ep['rows'])
        if worlds is not None and current != worlds:
            raise ValueError('incomplete paired worlds')
        worlds = current
        exported = [json.loads(line) for line in path.open()]
        if len(exported) != meta['rows']:
            raise ValueError('row count changed')
        cache, counts, seen = {}, collections.Counter(), set()
        for row in exported:
            key = stream, row['seed']
            if (key not in episodes or row['version'] != 'hjepa_actor_rollout_row_v1' or row['split'] != 'train'
                    or len(row['features']) != 871 or any(not math.isfinite(v) for v in row['features'])):
                raise ValueError('invalid actor row')
            source = row['source']
            if source not in sources or row['source_sha256'] != sources[source]:
                raise ValueError('unbound source')
            if source not in cache:
                steps = [json.loads(line) for line in pathlib.Path(source).open()]
                cache[source] = {s['index']: s for s in steps}
                op = pathlib.Path(source).parent/'server_outcomes.jsonl'
                ledger = episode_ledger(episodes[key]['result'], steps, [json.loads(line) for line in op.open()])
                episodes[key]['ledger'] = ledger
                sources[str(op)] = sha(op)
            step = cache[source][row['step_index']]
            sample = row['sample']
            if ((source, row['step_index']) in seen or step['owner'] != 'provider'
                    or step['sample'] != sample or step['observation']['identity'] != row['identity']
                    or not step['server_execution']['matched'] or step['server_execution']['recovery_commands'] != 0
                    or row['exclusive'] != step['server_execution']['window_exclusive']
                    or sample['version'] != 'hjepa_primitive_sample_v1'
                    or sample['sampling_seed'] != stream or len(sample['latent']) != 4
                    or not all(math.isfinite(v) for v in sample['latent']) or sample['vertical'] not in (0, 1, 2)):
                raise ValueError('native actor receipt mismatch')
            seen.add((source, row['step_index']))
            row['episode_key'] = key
            counts[key] += 1
            rows.append(row)
        if any(counts[stream, seed] != episodes[stream, seed]['expected'] for seed in current):
            raise ValueError('missing episode choices')
    if len(streams) < 2 or len(set(map(str, paths))) != len(paths):
        raise ValueError('independent paired streams required')
    for seed in worlds:
        group = [ep for (_, s), ep in episodes.items() if s == seed]
        if any(ep['result']['generated_fixture'] != group[0]['result']['generated_fixture'] for ep in group):
            raise ValueError('paired fixtures differ')
        first = {}
        for row in rows:
            if row['seed'] == seed:
                first.setdefault(row['episode_key'], row['features'])
        if len(first) != len(streams) or any(v != next(iter(first.values())) for v in first.values()):
            raise ValueError('paired initial observations differ')
    admitted, masked = [], collections.Counter()
    for row in rows:
        key, frame, index = row['episode_key'], row['identity']['frame'], row['step_index']
        ep = episodes[key]
        ledger, life = ep['ledger'], ep['result']['first_life']
        if not eligible(row, ledger, life):
            masked['outside_exclusive_first_life'] += 1
            continue
        other = [e['ledger'] for (stream, seed), e in episodes.items() if seed == key[1] and stream != key[0]]
        row['advantage'] = (ledger['suffix'][index] - sum(at_frame(v, frame) for v in other)/len(other))/100
        admitted.append(row)
    if not admitted:
        raise ValueError('no first-life actor samples')
    proof = dict(rows=len(rows), admitted=len(admitted), masked=dict(masked), worlds=len(worlds), episodes=len(episodes),
                 returns=[dict(stream=k[0], seed=k[1], value=v['ledger']['scalar_return']) for k, v in episodes.items()])
    return admitted, parent, dict(inputs=inputs, sources=sources), proof


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=pathlib.Path, action='append', required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    ap.add_argument('--updates', type=int, default=4)
    ap.add_argument('--resume', type=pathlib.Path)
    a = ap.parse_args()
    if not 1 <= a.updates <= ALGORITHM['max_updates'] or not torch.cuda.is_available():
        raise ValueError('CUDA and 1..4 total updates required')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    rows, parent, binding, proof = load_rollouts(a.data)
    model = Actor(parent).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=ALGORITHM['learning_rate'])
    tensor = lambda values, dtype=torch.float64: torch.tensor(values, dtype=dtype, device='cuda')
    x, latent = tensor([r['features'] for r in rows]), tensor([r['sample']['latent'] for r in rows])
    attack, vertical = tensor([r['sample']['attack'] for r in rows], torch.bool), tensor([r['sample']['vertical'] for r in rows], torch.long)
    old_raw, old_lp = tensor([r['actor_raw'] for r in rows]), tensor([r['primitive_log_probability'] for r in rows])
    old_std = model.log_std.detach().clone()
    advantage = tensor([r['advantage'] for r in rows])
    with torch.no_grad():
        raw_error = float((model(x)-old_raw).abs().max())
        lp_error = float((log_probability(model(x), old_std, latent, attack, vertical, model.temperature)-old_lp).abs().max())
    if raw_error > 1e-8 or lp_error > 1e-7:
        raise ValueError('initial Go/CUDA behavior likelihood mismatch')
    history = []
    if a.resume:
        ck = torch.load(a.resume, map_location='cuda', weights_only=False)
        if (ck['version'] != VERSION or ck['algorithm'] != ALGORITHM or ck['binding'] != binding
                or ck['parent_fingerprint'] != fingerprint(parent) or ck['stopped']):
            raise ValueError('resume binding/algorithm mismatch')
        model.load_state_dict(ck['state']); optimizer.load_state_dict(ck['optimizer']); history = ck['history']
        if not len(history) < a.updates:
            raise ValueError('resume must advance the same batch')

    def metrics():
        raw = model(x)
        lp = log_probability(raw, model.log_std, latent, attack, vertical, model.temperature)
        ratio = (lp-old_lp).exp()
        surrogate = torch.minimum(ratio*advantage, ratio.clamp(.9, 1.1)*advantage).mean()
        kl = distribution_kl(old_raw, old_std, raw, model.log_std, model.temperature)
        return surrogate, kl.mean(), kl.max(), ((ratio-1).abs() > .1).double().mean()

    stopped = False
    for update in range(len(history), a.updates):
        before = copy.deepcopy(model.state_dict())
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        optimizer.zero_grad(set_to_none=True)
        surrogate, _, _, _ = metrics()
        prior = float(surrogate.detach())
        (-surrogate).backward()
        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.))
        optimizer.step()
        with torch.no_grad():
            model.log_std.clamp_(-10, 1)
        proposed = copy.deepcopy(model.state_dict())
        accepted = False
        with torch.no_grad():
            for power in range(ALGORITHM['backtrack_power']+1):
                scale = 2.**(-power)
                model.load_state_dict({k: before[k]+scale*(proposed[k]-before[k]) for k in before})
                values = [float(v) for v in metrics()]
                if all(math.isfinite(v) for v in values) and values[1] <= .02 and values[0] >= prior-1e-12:
                    accepted = True
                    break
        if not accepted:
            model.load_state_dict(before); optimizer.load_state_dict(optimizer_before)
            stopped = True
            break
        history.append(dict(update=update+1, surrogate=values[0], mean_kl=values[1], max_kl=values[2],
                            clipped_fraction=values[3], step_scale=scale, gradient_norm=grad_norm))
    a.out.mkdir(parents=True, exist_ok=True)
    candidate = copy.deepcopy(parent)
    candidate['levels'][0]['encoder'], candidate['actor'] = export(model.encoder), export(model.actor)
    candidate['primitive_sampling']['log_std'] = model.log_std.detach().cpu().tolist()
    # Choose the first source stream's seed for reproducible future collection.
    source_meta = json.loads(pathlib.Path(str(a.data[0])+'.meta.json').read_text())
    candidate['primitive_sampling']['seed'] = json.loads(pathlib.Path(source_meta['model']).read_text())['primitive_sampling']['seed']
    save(a.out/'model.json', candidate)
    deterministic = copy.deepcopy(candidate); deterministic.pop('primitive_sampling')
    save(a.out/'deterministic.json', deterministic)
    torch.save(dict(version=VERSION, algorithm=ALGORITHM, binding=binding, parent_fingerprint=fingerprint(parent),
                    state=model.state_dict(), optimizer=optimizer.state_dict(), history=history, stopped=stopped,
                    model_sha256=sha(a.out/'model.json')), a.out/'checkpoint.pt')
    report = dict(version=VERSION, algorithm=ALGORITHM, admission=proof, initial_parity=dict(raw=raw_error, log_probability=lp_error),
                  history=history, stopped=stopped, model_sha256=sha(a.out/'model.json'), binding=binding,
                  scope='First primitive PPO pilot only. Frozen weapon and auxiliary world heads. No combat quality or promotion claim.')
    save(a.out/'report.json', report)
    print(json.dumps(dict(admission=proof | {'returns': 'see report'}, initial_parity=report['initial_parity'], history=history, stopped=stopped), indent=2))


if __name__ == '__main__':
    main()

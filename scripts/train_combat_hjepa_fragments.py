"""CUDA joint primitive/weapon update on source-bound short native fragments.

Only native-terminal same-stream Monte Carlo targets are accepted. Repeated
branches are grouped by checkpoint family; diagnostics never enter this loader.
"""
import argparse
import copy
import json
import math
import pathlib

import torch
from torch import nn
from torch.nn import functional as F

from train_combat_hjepa import export, save, sha
from train_combat_hjepa_actor_ppo import Actor, network, log_probability, distribution_kl, fingerprint

VERSION = 'hjepa_short_fragment_joint_ppo_v1'
ALGORITHM = dict(learning_rate=1e-4, clip=.1, max_mean_kl=.02, gradient_clip=1., advantage_scale=100., credit='same_root_other_stream_native_terminal_mc_v1')


def load(paths):
    inputs, families, rows, parent = {}, {}, [], None
    for path in paths:
        target = json.loads(path.read_text())
        if target['version'] != 'hjepa_fragment_mc_targets_v1' or not target['optimizer_ready'] or target['bootstrap_proof'] != 'native_terminal_in_same_stream_continuation':
            raise ValueError('verified native terminal targets required')
        for name, digest in target['sources_sha256'].items():
            if sha(pathlib.Path(name)) != digest:
                raise ValueError('changed target source: '+name)
            inputs[name] = digest
        inputs[str(path.resolve())] = sha(path)
        fragment = json.loads(pathlib.Path(target['prefix']).read_text())
        if fragment['version'] != 'hjepa_native_fragment_v2' or fragment['split'] != 'train' or not fragment['training_source_verified'] or not fragment['training_plan']:
            raise ValueError('typed verified TRAIN fragment required')
        required = {target['prefix'], target['tail'], fragment['behavior_model'], fragment['training_plan']}
        if not required.issubset(target['sources_sha256']) or any(
                target['sources_sha256'].get(name) != digest for name, digest in fragment['sources_sha256'].items()):
            raise ValueError('target must bind prefix, tail and all fragment sources')
        if (target['model_version'] != fragment['model_version'] or target['bootstrap_value'] != 0
                or target['terminal_reason'] not in ('native_death', 'native_source_roster_clear')
                or target['terminal_step'] < len(fragment['transitions'])-1):
            raise ValueError('native terminal/model binding differs')
        model_path = pathlib.Path(fragment['behavior_model'])
        model = json.loads(model_path.read_text())
        if ('hjepa:'+sha(model_path) != fragment['model_version'] or model['weapon_mode'] != 'learned_masked_v1'
                or model['weapon_head'] != 'combat_masked_weapon_v1' or model.get('weapon_sampling_seed') is None
                or model['planner'] != 'bc' or model.get('motor') or model.get('weapon_option_frames', 0)):
            raise ValueError('unchanged full joint behavior actor required')
        if parent is not None and model != parent:
            raise ValueError('different behavior weights')
        parent = model
        family, stream = fragment['checkpoint_family'], fragment['sampling_checkpoint']['sampling_stream']
        bank = families.setdefault(family, {})
        if stream in bank:
            raise ValueError('duplicate family/stream, including repeated mechanical runs')
        transitions = fragment['transitions']
        labels = target['targets']
        if len(transitions) != len(labels) or target['sampling_stream'] != stream or target['checkpoint_family'] != family:
            raise ValueError('target fragment binding/length')
        bank[stream] = {r['index']: t['return_to_go'] for r,t in zip(transitions, labels) if r['index'] == t['index']}
        if len(bank[stream]) != len(transitions):
            raise ValueError('target indexes differ')
        if not all(math.isfinite(value) for value in bank[stream].values()):
            raise ValueError('nonfinite terminal return')
        for row in transitions:
            if row['version'] != 'hjepa_native_fragment_transition_v2' or len(row['features']) != 871 or len(row['actor_raw']) != 8:
                raise ValueError('fragment feature/logit version')
            if not all(math.isfinite(v) for v in row['features']+row['actor_raw']):
                raise ValueError('nonfinite fragment inputs')
            rows.append((family, stream, row))
    if not rows:
        raise ValueError('empty TRAIN fragments')
    packed = []
    for family, stream, row in rows:
        peers = [bank[row['index']] for key,bank in families[family].items() if key != stream and row['index'] in bank]
        if not peers:
            raise ValueError('at least two paired streams per checkpoint family required')
        advantage = (families[family][stream][row['index']]-sum(peers)/len(peers))/100.
        packed.append((row, advantage))
    return parent, packed, inputs, dict(rows=len(rows), checkpoint_families=len(families), streams={key: sorted(bank) for key,bank in families.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--targets', type=pathlib.Path, nargs='+', required=True)
    p.add_argument('--out', type=pathlib.Path, required=True)
    p.add_argument('--updates', type=int, default=1)
    p.add_argument('--resume', type=pathlib.Path)
    a = p.parse_args()
    if not 1 <= a.updates <= 4 or a.out.exists() or not torch.cuda.is_available():
        raise ValueError('fresh output, CUDA and1..4 bounded updates required')
    parent, packed, inputs, admission = load(a.targets)
    torch.manual_seed(1729)
    model = Actor(parent).cuda()
    weapon = network(parent['weapon_actor']).cuda()
    parameters = list(model.parameters())+list(weapon.parameters())
    optimizer = torch.optim.Adam(parameters, lr=ALGORITHM['learning_rate'])
    tensor = lambda values, dtype=torch.float64: torch.tensor(values, dtype=dtype, device='cuda')
    rows = [r for r,_ in packed]
    x, old_raw = tensor([r['features'] for r in rows]), tensor([r['actor_raw'] for r in rows])
    latent = tensor([r['sample']['latent'] for r in rows])
    attack = tensor([r['sample']['attack'] for r in rows], torch.bool)
    pose = tensor([r['sample']['vertical'] for r in rows], torch.long)
    chosen = tensor([r['sample'].get('weapon', 0) for r in rows], torch.long)
    mask = x[:, 833:845] > .5
    old_lp = tensor([r['sample']['log_probability'] for r in rows])
    advantage = tensor([v for _,v in packed])
    old_std = model.log_std.detach().clone()
    with torch.no_grad():
        old_weapon = F.log_softmax(weapon(x).masked_fill(~mask, -torch.inf), -1)

    def current():
        raw = model(x)
        wp = F.log_softmax(weapon(x).masked_fill(~mask, -torch.inf), -1)
        lp = log_probability(raw, model.log_std, latent, attack, pose, model.temperature)+wp.gather(1, chosen[:,None])[:,0]
        ratio = (lp-old_lp).exp()
        surrogate = torch.minimum(ratio*advantage, ratio.clamp(.9,1.1)*advantage).mean()
        wk = torch.where(mask, old_weapon.exp()*(old_weapon-wp), 0.).sum(-1)
        kl = distribution_kl(old_raw, old_std, raw, model.log_std, model.temperature)+wk
        return surrogate, kl.mean(), kl.max(), lp

    with torch.no_grad():
        raw_error = float((model(x)-old_raw).abs().max())
        lp_error = float((current()[3]-old_lp).abs().max())
    if raw_error > 1e-8 or lp_error > 1e-7:
        raise ValueError(f'Go/CUDA initial joint density mismatch: {raw_error}, {lp_error}')
    binding = fingerprint(inputs)
    history = []
    if a.resume:
        ck = torch.load(a.resume, map_location='cuda', weights_only=False)
        if ck['version'] != VERSION or ck['algorithm'] != ALGORITHM or ck['binding'] != binding or ck['parent'] != fingerprint(parent):
            raise ValueError('resume algorithm/input/model mismatch')
        model.load_state_dict(ck['actor']);weapon.load_state_dict(ck['weapon']);optimizer.load_state_dict(ck['optimizer']);history=ck['history']
        if len(history) >= a.updates:
            raise ValueError('resume must advance same batch')
    for index in range(len(history),a.updates):
        before = copy.deepcopy(model.state_dict()), copy.deepcopy(weapon.state_dict()), copy.deepcopy(optimizer.state_dict())
        optimizer.zero_grad(set_to_none=True)
        prior = float(current()[0].detach())
        (-current()[0]).backward()
        grad = float(nn.utils.clip_grad_norm_(parameters,1.))
        optimizer.step()
        with torch.no_grad():model.log_std.clamp_(-10,1)
        proposal = copy.deepcopy(model.state_dict()), copy.deepcopy(weapon.state_dict())
        accepted = False
        with torch.no_grad():
            for power in range(13):
                scale = 2.**(-power)
                model.load_state_dict({k:v+scale*(proposal[0][k]-v) for k,v in before[0].items()})
                weapon.load_state_dict({k:v+scale*(proposal[1][k]-v) for k,v in before[1].items()})
                gain, mean_kl, max_kl, _ = current()
                metrics = [float(gain),float(mean_kl),float(max_kl)]
                if all(math.isfinite(v) for v in metrics) and metrics[1] <= .02 and metrics[0] >= prior-1e-12:
                    accepted = True;break
        if not accepted:
            model.load_state_dict(before[0]);weapon.load_state_dict(before[1]);optimizer.load_state_dict(before[2]);break
        history.append(dict(update=index+1,surrogate=metrics[0],mean_kl=metrics[1],max_kl=metrics[2],step_scale=scale,gradient_norm=grad))
    candidate = copy.deepcopy(parent)
    candidate['levels'][0]['encoder'], candidate['actor'], candidate['weapon_actor'] = export(model.encoder), export(model.actor), export(weapon)
    candidate['primitive_sampling']['log_std'] = model.log_std.detach().cpu().tolist()
    a.out.mkdir(parents=True)
    save(a.out/'model.json', candidate)
    torch.save(dict(version=VERSION,algorithm=ALGORITHM,binding=binding,parent=fingerprint(parent),actor=model.state_dict(),weapon=weapon.state_dict(),optimizer=optimizer.state_dict(),history=history),a.out/'checkpoint.pt')
    with torch.no_grad():
        save(a.out/'parity-actor.json',dict(features=x[:8].cpu().tolist(),outputs=model(x[:8]).cpu().tolist()))
        save(a.out/'parity-weapon.json',dict(features=x[:8].cpu().tolist(),outputs=weapon(x[:8]).cpu().tolist()))
    report = dict(version=VERSION,algorithm=ALGORITHM,admission=admission,initial_parity=dict(raw=raw_error,joint_log_probability=lp_error),history=history,binding=binding,inputs_sha256=inputs,model_sha256=sha(a.out/'model.json'),device=torch.cuda.get_device_name(),promotion=False,scope='Offline joint primitive/weapon fragment pilot; no independent combat quality evidence')
    save(a.out/'report.json',report)
    print(json.dumps({k:report[k] for k in ['admission','initial_parity','history','device','promotion']},indent=2))


if __name__ == '__main__':
    main()

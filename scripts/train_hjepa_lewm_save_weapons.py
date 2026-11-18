"""Offline four-frame weapon-option dynamics for paired restored Quake II saves.

Train an adapted LeWM loss and a matched H-JEPA-style EMA reference. Both
start from the same saved H-JEPA encoder. A separate frozen-latent readout
predicts first-life net HP over ten option steps; it cannot see test outcomes.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from train_combat_hjepa_branch_preferences import network


def sigreg(z):
    directions = F.normalize(torch.randn(z.shape[-1], 128, device=z.device), dim=0)
    frequencies = torch.linspace(0, 3, 17, device=z.device)
    normal = torch.exp(-frequencies.square() / 2)
    phases = (z @ directions).unsqueeze(-1) * frequencies
    deviation = (phases.cos().mean(0) - normal).square() + phases.sin().mean(0).square()
    return z.shape[0] * torch.trapezoid(deviation * normal, frequencies).mean()


class World(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encode = network(encoder)
        width = len(encoder[-1]['bias'])
        self.step = nn.Sequential(nn.Linear(width + 12, 64), nn.ReLU(), nn.Linear(64, width))
        self.readout = nn.Sequential(nn.Linear(2 * width, 32), nn.ReLU(), nn.Linear(32, 1))

    def forecast(self, root, option):
        start = z = self.encode(root)
        for _ in range(10):
            z = self.step(torch.cat((z, option), -1))
        return self.readout(torch.cat((start, z), -1)).flatten()


def tensor(x):
    return torch.tensor(x, device='cuda', dtype=torch.float32)


def arrays(cases):
    x, y, choices = [], [], []
    roots, options, outcomes = [], [], []
    for case in cases:
        for branch in case['options']:
            k = branch['choice']
            roots.append(branch['root'])
            options.append(k)
            outcomes.append(branch['score'])
            for prev, nxt in zip(branch['states'], branch['states'][1:]):
                if prev is not None and nxt is not None:
                    x.append(prev)
                    y.append(nxt)
                    choices.append(k)
    def categorical(ids):
        return F.one_hot(torch.tensor(ids, device='cuda'), 12).float()
    return tensor(x), tensor(y), categorical(choices), tensor(roots), categorical(options), tensor(outcomes)


def fit(kind, encoder, train, destination):
    torch.manual_seed(731)
    model = World(encoder).cuda()
    target = copy.deepcopy(model.encode).requires_grad_(False)
    optim = torch.optim.Adam([*model.encode.parameters(), *model.step.parameters()], lr=.0005)
    x, y, choices, roots, opts, scores = arrays(train)
    for _ in range(300):
        idx = torch.randint(len(x), (64,), device='cuda')
        z = model.encode(x[idx])
        nxt = model.encode(y[idx])
        pred = model.step(torch.cat((z, choices[idx]), -1))
        if kind == 'lewm':
            loss = F.mse_loss(pred, nxt) + .05 * sigreg(torch.cat((z, nxt)))
        else:
            states = torch.cat((z, nxt))
            centered = states - states.mean(0)
            cov = centered.T @ centered / (len(states)-1)
            penalty = F.relu(1 - (states.var(0) + 1e-4).sqrt()).mean()
            penalty = penalty + (cov - cov.diag().diag()).square().sum()/z.shape[-1]
            loss = F.mse_loss(pred, target(y[idx])) + .2 * penalty
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite objective')
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optim.step()
        if kind == 'ema':
            with torch.no_grad():
                for a,b in zip(target.parameters(), model.encode.parameters()):
                    a.lerp_(b, .01)
    model.encode.requires_grad_(False)
    model.step.requires_grad_(False)
    optim = torch.optim.Adam(model.readout.parameters(), lr=.002)
    scale = scores.std().clamp_min(1.)
    for _ in range(200):
        pred = model.forecast(roots, opts)
        delta = (pred - scores/scale).reshape(-1, 2)
        loss = delta.square().mean() + (delta[:,1]-delta[:,0]).square().mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
    torch.save(dict(kind=kind, version='quake_hjepa_option_world_v1', state=model.state_dict(),
                    score_scale=scale.item(), encoder_template=encoder, step_frames=4,
                    horizon_steps=10, train_seeds=[v['seed'] for v in train]), destination)


@torch.no_grad()
def predict(path, cases):
    cp=torch.load(path, map_location='cuda', weights_only=True)
    assert not set(cp['train_seeds']) & {r['seed'] for r in cases}
    model=World(cp['encoder_template']).cuda()
    model.load_state_dict(cp['state'])
    options=[b for c in cases for b in c['options']]
    choices=F.one_hot(torch.tensor([b['choice'] for b in options],device='cuda'),12).float()
    return (model.forecast(tensor([b['root'] for b in options]),choices)*cp['score_scale']).reshape(-1,2).tolist()


def run(folder):
    cases=json.loads((folder/'pairs.json').read_text())
    train=[c for c in cases if c['split']=='train']
    test=[c for c in cases if c['split']=='test']
    assert len({c['seed'] for c in cases})==len(cases)
    encoder=json.loads(Path(train[0]['model']).read_text())['levels'][0]['encoder']
    results={}
    for kind in ['lewm','ema']:
        path=folder/(kind+'.pt')
        if path.exists():raise FileExistsError(path)
        fit(kind,encoder,train,path)
        results[kind]=predict(path,test)
    logits=[]
    with torch.no_grad():
        for case in test:
            original=json.loads(Path(case['model']).read_text())
            head=network(original['weapon_actor']).cuda()
            x=tensor(case['options'][0]['root'])
            logits.append(head(x)[[b['choice'] for b in case['options']]].tolist())
    results['original_weapon_head']=logits
    outcomes=[[b['score'] for b in c['options']] for c in test]
    gaps=tensor(outcomes).diff().flatten()
    metrics={}
    for name,pred in results.items():
        d=tensor(pred).diff().flatten()
        signed=d[gaps!=0]*gaps[gaps!=0]
        metrics[name]=dict(correct=int((signed>0).sum()),wrong=int((signed<0).sum()),
                           tied_predictions=int((signed==0).sum()),unequal_native=len(signed),
                           mean_regret_hp=float(torch.where(d>0,(-gaps).clamp_min(0),
                                               torch.where(d<0,gaps.clamp_min(0),gaps.abs()/2)).mean()))
        if name!='original_weapon_head':
            metrics[name]['contrast_mae_hp']=float((d-gaps).abs().mean())
    result=dict(train=len(train),test=len(test),metrics=metrics,test_seeds=[c['seed'] for c in test],
                outcomes=outcomes,predictions=results,source_data_sha256=hashlib.sha256((folder/'pairs.json').read_bytes()).hexdigest())
    (folder/'evaluation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    torch.set_num_threads(2)
    run(parser.parse_args().folder)

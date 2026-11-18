"""Train or query the separate 40-frame Quake II damage readout.

train LABELS_JSON PARENT_LEWM_PT NEW_CHECKPOINT
predict CHECKPOINT OBSERVATION_JSON
"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from train_hjepa_lewm_save_weapons import World


VERSION = 'quake_lewm_weapon_damage_v2'


class DamageModel(nn.Module):
    def __init__(self, template):
        super().__init__()
        self.world = World(template)
        self.world.readout = nn.Identity()
        self.world.requires_grad_(False)
        width = len(template[-1]['bias'])
        self.head = nn.Sequential(nn.Linear(2 * width + 12, 32), nn.ReLU(), nn.Linear(32, 4))
        self.register_buffer('scales', torch.ones(3))

    @torch.no_grad()
    def context(self, roots, slots):
        option = F.one_hot(slots, 12).float()
        start = state = self.world.encode(roots)
        for _ in range(10):
            state = self.world.step(torch.cat((state, option), dim=-1))
        effective = torch.where(slots == 0, roots[:, 845:857].argmax(1), slots)
        return torch.cat((start, state, F.one_hot(effective, 12).float()), dim=-1)

    def components(self, context):
        raw = self.head(context)
        probability = raw[:, 3].sigmoid()
        amounts = F.softplus(raw[:, :3]) * self.scales
        expected = torch.stack((amounts[:, 0], probability * amounts[:, 1], amounts[:, 2]), dim=1)
        return expected, raw[:, 3], amounts[:, 1]

    def forward(self, roots, slots):
        expected, logits, _ = self.components(self.context(roots, slots))
        return expected, logits.sigmoid()


def net(amounts):
    return amounts[:, 0] - amounts[:, 1] - amounts[:, 2]


def pack(rows):
    roots, slots, labels = [], [], []
    for row in rows:
        if len(row['branches']) != 2:
            raise ValueError('Expected two native continuations per save')
        for branch in row['branches']:
            roots.append(row['root'])
            slots.append(branch['slot'])
            labels.append(branch['labels'])
    return torch.tensor(roots, dtype=torch.float32), torch.tensor(slots), torch.tensor(labels, dtype=torch.float32)


def train(dataset, parent, destination):
    rows = json.loads(dataset.read_text())
    train_rows = [r for r in rows if r['split'] == 'train']
    seeds = [r['seed'] for r in train_rows]
    cp = torch.load(parent, map_location='cpu', weights_only=True)
    if (cp['kind'] != 'lewm' or cp['step_frames'] != 4 or cp['horizon_steps'] != 10
            or set(seeds) != set(cp['train_seeds']) or len({r['seed'] for r in rows}) != len(rows)):
        raise ValueError('Incompatible parent model or save split')
    torch.manual_seed(731)
    model = DamageModel(cp['encoder_template'])
    model.world.load_state_dict({k: v for k, v in cp['state'].items() if not k.startswith('readout.')})
    roots, slots, labels = pack(train_rows)
    event = labels[:, 1] > 0
    if not event.any() or event.all():
        raise ValueError('Both blast-event classes are needed in training')
    model.scales.copy_(labels.mean(0).clamp_min(1))
    model.scales[1] = labels[event, 1].mean()
    context = model.context(roots, slots)
    score_scale = net(labels).std().clamp_min(1)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=.002)
    for _ in range(200):
        expected, logits, severity = model.components(context)
        err = (net(expected) - net(labels)).reshape(-1, 2) / score_scale
        loss = (F.smooth_l1_loss(expected / model.scales, labels / model.scales)
                + F.binary_cross_entropy_with_logits(logits, event.float())
                + F.smooth_l1_loss(severity[event] / model.scales[1], labels[event, 1] / model.scales[1])
                + err.diff(dim=1).square().mean())
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with destination.open('xb') as output:
        torch.save(dict(version=VERSION, template=cp['encoder_template'], state=model.state_dict(),
                        train_seeds=seeds), output)


def load(checkpoint):
    cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if cp['version'] != VERSION:
        raise ValueError('Unsupported damage model')
    model = DamageModel(cp['template'])
    model.load_state_dict(cp['state'])
    return model.eval()


@torch.inference_mode()
def predict(model, sample):
    root = torch.tensor(sample['features'], dtype=torch.float32)
    choices, available = sample['choices'], sample['available']
    if root.shape != (871,) or not torch.isfinite(root).all() or len(available) != 12:
        raise ValueError('Expected 871 root features and 12 availability flags')
    if (not choices or any(type(k) is not int or k not in range(12) or not available[k] for k in choices)
            or len(set(choices)) != len(choices)):
        raise ValueError('Invalid weapon options')
    exp, prob = model(root.expand(len(choices), -1), torch.tensor(choices))
    scores = net(exp)
    return dict(preferred_slot=choices[int(scores.argmax())], options=[
        dict(slot=k, monster_hp=float(x[0]), self_blast_probability=float(p),
             expected_self_blast_hp=float(x[1]), other_hp_lost=float(x[2]), net_hp=float(s))
        for k, x, p, s in zip(choices, exp, prob, scores)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('train')
    for arg in ('dataset', 'parent', 'destination'):
        fit.add_argument(arg, type=Path)
    forecast = sub.add_parser('predict')
    forecast.add_argument('checkpoint', type=Path)
    forecast.add_argument('observation', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == 'train':
        train(args.dataset, args.parent, args.destination)
    else:
        print(json.dumps(predict(load(args.checkpoint), json.loads(args.observation.read_text())), indent=2))

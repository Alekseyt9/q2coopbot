"""Score available Quake II weapon requests with an offline restored-save model."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from train_hjepa_lewm_save_weapons import World, tensor


def score(checkpoint, sample):
    cp = torch.load(checkpoint, map_location='cuda', weights_only=True)
    if cp['version'] != 'quake_hjepa_option_world_v1' or cp['horizon_steps'] != 10:
        raise ValueError('Unsupported weapon forecast checkpoint')
    model = World(cp['encoder_template']).cuda().eval()
    model.load_state_dict(cp['state'])
    x = tensor(sample['features'])
    mask = sample['available']
    slots = sample['choices']
    if x.shape != (871,) or not torch.isfinite(x).all() or len(mask) != 12:
        raise ValueError('Expected 871 H-JEPA context features and 12 weapon availability flags')
    if not slots or len(slots) != len(set(slots)) or any(type(k) is not int or k not in range(12) or not mask[k] for k in slots):
        raise ValueError('Invalid or unavailable weapon option')
    choices = F.one_hot(torch.tensor(slots, device='cuda'), 12).float()
    with torch.inference_mode():
        predictions = model.forecast(x.expand(len(slots), -1), choices) * cp['score_scale']
    return dict(net_hp_by_slot=dict(zip(slots, predictions.tolist())),
                preferred_slot=slots[int(predictions.argmax())])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('observation', type=Path)
    args = parser.parse_args()
    print(json.dumps(score(args.checkpoint, json.loads(args.observation.read_text())), indent=2))

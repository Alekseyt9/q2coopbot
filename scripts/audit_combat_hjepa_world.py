"""Measure held-out action sensitivity and temporal latent variation."""
import argparse
import json
import pathlib
import torch
from train_combat_hjepa import Model, STRIDES, load, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=pathlib.Path, required=True)
    ap.add_argument('--validation', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    a = ap.parse_args()
    cp = torch.load(a.checkpoint, map_location='cuda', weights_only=True)
    model = Model().cuda()
    model.load_state_dict(cp['model'])
    (x, actions, _, _), _, _ = load(a.validation, 'cuda')
    results = []
    with torch.no_grad():
        zs = model.encode(x)
        for i, stride in enumerate(STRIDES):
            z = zs[i]
            u = actions[:, :stride].flatten(1)
            actual = model.predictors[i](torch.cat((z[:, 0], u), -1))
            shuffled = model.predictors[i](torch.cat((z[:, 0], u.roll(1, 0)), -1))
            target = z[:, stride]
            results.append(dict(stride=stride, actual_mse=float((actual-target).square().mean()),
                                shuffled_action_mse=float((shuffled-target).square().mean()),
                                action_response_rms=float((actual-shuffled).square().mean().sqrt()),
                                mean_latent_std=float(z.flatten(0, 1).std(0).mean()),
                                within_clip_std=float(z.std(1).mean())))
    report = dict(version='hjepa_world_audit_v1', levels=results,
                  checkpoint_sha256=sha(a.checkpoint), validation_sha256=sha(a.validation),
                  scope='Held-out offline prediction and action permutation diagnostic; not causal gameplay benefit.')
    a.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

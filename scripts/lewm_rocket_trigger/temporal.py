"""Native reward-only search for a persistent latent controller. No action labels."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from build import HERE, REPO, substitute
from experiment import BANK, V2, collect, read, root_obs, save, totals


def build_temporal(out):
    source = REPO / 'internal/policy/hjepa.go'
    body = substitute(source.read_text(), 'type HJEPA struct {', 'type HJEPA struct {\n slow *slowPolicy')
    body = substitute(body, '\tprior := hPrimitive(raw)', '''
    if path := os.Getenv("Q2_LEWM_TEMPORAL_POLICY"); path != "" {
        if p.slow == nil {
            p.slow, e = loadSlowPolicy(path)
            if e != nil { return Action{}, e }
        }
        if e = p.slow.condition(o,raw); e != nil { return Action{}, e }
    }
    prior := hPrimitive(raw)''')
    reset = 'func (p *HJEPA) ResetEpisode() {'
    body = substitute(body, reset, reset + '\n p.slow = nil')
    generated = out/'hjepa.go'
    generated.write_text(body + '\n' + (HERE/'temporal.go.inc').read_text())
    subprocess.run(['gofmt', '-w', generated], check=True)
    overlay = out/'overlay.json'
    save(overlay, {'Replace': {str(source): str(generated),
          str(source.with_name('experimental_slow_option_test.go')):str(HERE/'temporal_test.go.inc')}})
    subprocess.run(['go', 'build', '-overlay', overlay, '-o', out/'rocket-client.exe', './cmd/q2coopbot'],
                   cwd=REPO, check=True)


def frozen_encoder(train, trigger):
    import sys
    import torch
    sys.path.insert(0, str(REPO/'scripts'))
    from hjepa_lewm_weapon_damage import load
    encoder = load(V2/'damage.pt').world.encode
    roots = [root_obs(Path(r['source'])/'bot.jsonl', r['frame'])['features'] for r in train]
    with torch.no_grad():
        z = encoder(torch.tensor(roots, dtype=torch.float32)).double().numpy()
    center = z.mean(axis=0)
    _, _, vh = np.linalg.svd(z-center, full_matrices=False)
    axes = vh[:2]
    projection = axes / np.maximum(np.std((z-center) @ axes.T, axis=0)[:, None], 1e-6)
    return dict(Encoder=read(trigger/'near.json')['Encoder'],
                Projection=dict(weight=projection.tolist(), bias=(-projection @ center).tolist()))


def utility(stats):
    return stats['monster_hp'] - stats['self_hp'] - stats['other_hp']


def run(out, workers):
    out.mkdir(parents=True, exist_ok=True)
    trigger = BANK/'lewm-rocket-trigger-v1'
    roots = read(trigger/'plan.json')['roots']
    train = [r for r in roots if r['split'] == 'train']
    heldout = [r for r in roots if r['split'] != 'train'] + read(trigger/'external-plan.json')
    assert len({r['seed'] for r in train+heldout}) == len(train)+len(heldout)
    save(out/'protocol.json', dict(train=[r['seed'] for r in train], heldout=[r['seed'] for r in heldout],
         period=8, reward='first-life live-monster HP removed minus all player HP removed over 40 frames',
         random_seed=1447, directions_per_round=2, round_scales=[.85,.50], initial_weights='zero',
         fixture_forced_weapon=False))
    build_temporal(out)
    base = frozen_encoder(train, trigger)
    center = np.zeros((3,6))
    rng = np.random.default_rng(1447)
    incumbent = 'base'
    history = {}
    for step, scale in enumerate((.85, .50)):
        candidates = {incumbent:center}
        for direction in range(2):
            noise = rng.standard_normal(center.shape)
            for sign in (-1, 1):
                name = f'r{step}-d{direction}-s{sign}'
                w = center + sign*scale*noise
                candidates[name] = w
                save(out/f'{name}.json', dict(base,Controller=dict(weight=w[:,:5].tolist(),bias=w[:,5].tolist())))
        collect(out, train, candidates, workers, fixture_rocket=False, port=49240)
        stats = totals(out,train,candidates)
        history.update(stats)
        incumbent = max(candidates, key=lambda k:utility(stats[k]))
        center = candidates[incumbent]
        save(out/f'round-{step}.json',dict(selected=incumbent,rewards={k:utility(v) for k,v in stats.items()}))
    best = max((k for k in history if k!='base'), key=lambda k:utility(history[k]))
    save(out/'selection.json',dict(incumbent=incumbent,challenger=best,training=history))
    collect(out,heldout,['base',best],workers,fixture_rocket=False,port=49240)
    metrics = totals(out,heldout,['base',best])
    shutil.copyfile(out/f'{best}.json',out/'learned.json')
    save(out/'evaluation.json',dict(candidate=best,train=history,heldout=metrics,
         heldout_reward={k:utility(v) for k,v in metrics.items()},
         learned_sha256=hashlib.sha256((out/'learned.json').read_bytes()).hexdigest()))
    print('heldout',metrics,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--workers',type=int,default=8)
    args = parser.parse_args()
    run(args.output.resolve(),args.workers)

"""Replay the frozen winner with per-frame updates or constant motor biases."""
import argparse
from pathlib import Path
import subprocess

from build import REPO, substitute
from experiment import BANK, collect, read, save, totals


def run(parent):
    out = parent/'ablations'
    out.mkdir(exist_ok=True)
    trigger = BANK/'lewm-rocket-trigger-v1'
    roots = [r for r in read(trigger/'plan.json')['roots'] if r['split'] != 'train']
    roots += read(trigger/'external-plan.json')
    selected = read(parent/'selection.json')['challenger']
    policy = read(parent/f'{selected}.json')
    save(out/'frame.json', policy)
    policy['Controller']['weight'] = [[0.0]*5 for _ in range(3)]
    save(out/'constant.json', policy)
    source = substitute((parent/'hjepa.go').read_text(), 'o.Identity.Frame-p.start >= 8', 'o.Identity.Frame-p.start >= 1')
    generated = out/'hjepa.go'
    generated.write_text(source)
    overlay = out/'overlay.json'
    save(overlay, {'Replace': {str(REPO/'internal/policy/hjepa.go'):str(generated)}})
    subprocess.run(['go','build','-overlay',overlay,'-o',out/'rocket-client.exe','./cmd/q2coopbot'],cwd=REPO,check=True)
    collect(out,roots,['frame','constant'],8,fixture_rocket=False,port=49620)
    save(out/'evaluation.json',totals(out,roots,['frame','constant']))


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('experiment',type=Path)
    run(ap.parse_args().experiment.resolve())

"""Select between two fitted heads on additional TRAIN saves before fresh validation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from experiment import V1, collect, read, root_obs, save, totals
from temporal import utility


def run(out):
    protocol = read(out/'protocol.json')
    worlds = [r for r in read(V1/'plan-verified.json')['roots']
              if r['split']=='train' and r['seed'] not in protocol['train']]
    order = sorted(worlds,key=lambda r:hashlib.sha256(str(r['seed']).encode()).digest())
    counts = Counter()
    selected = []
    for r in order:
        if counts[r['map']] >= 2:
            continue
        counts[r['map']] += 1
        obs = root_obs(Path(r['source'])/'bot.jsonl',r['frame'])
        assert obs['owner']=='provider'
        selected.append(dict(r,feature_hash=hashlib.sha256(json.dumps(obs['features']).encode()).hexdigest()))
    selection = read(out/'selection.json')
    adaptive = sorted((k for k in selection['train_scores'] if k.startswith('adaptive-')),
                      key=selection['train_scores'].get,reverse=True)[:2]
    fixed = list(dict.fromkeys(['fixed8',selection['best_fixed']]))
    save(out/'selection-protocol.json',dict(seeds=[r['seed'] for r in selected],candidates=fixed+adaptive,
         criterion='highest summed native HP return on supplementary training worlds only'))
    collect(out,selected,fixed+adaptive,6,fixture_rocket=False,port=50900,audit_duration=True)
    stats = totals(out,selected,fixed+adaptive)
    winner = max(adaptive,key=lambda k:utility(stats[k]))
    save(out/'supplementary-selection.json',dict(adaptive=winner,stats=stats,
                                                scores={k:utility(v) for k,v in stats.items()}))
    shutil.copyfile(out/f'{winner}.json',out/'selected-duration.json')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path)
    run(ap.parse_args().output.resolve())

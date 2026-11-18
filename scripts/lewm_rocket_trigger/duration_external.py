"""Evaluate a previously selected duration policy on newly captured validation worlds."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median

from duration import paired
from experiment import collect, read, root_obs, save


def run(out, workers, rocket=False):
    selection = read(out/'selection.json')
    candidate = read(out/'supplementary-selection.json')['adaptive']
    folder = out/'rocket-validation' if rocket else out
    seeds = read(folder/'validation-seeds.json')
    protocol = read(out/'protocol.json')
    completed = []
    for seed in read(out/'selection-protocol.json')['seeds']:
        run = read(out/f'{seed}-{candidate}.json')
        completed.extend(int(s.split()[2].split('=')[1]) for s in run['refreshes'] if s.endswith('continuous=true'))
    period = max(1,round(median(completed)))
    matched = f'fixed{period}'
    if not (out/f'{matched}.json').exists():
        policy = read(out/'fixed8.json')
        policy['Termination'] = dict(weight=[[0,0,0,0,0,1]],bias=[-period/8])
        save(out/f'{matched}.json',policy)
    names = list(dict.fromkeys(['fixed8', selection['best_fixed'], matched, candidate]))
    if rocket:
        candidate = read(out/'local-selection.json')['best']
        broad = read(out/'supplementary-selection.json')['adaptive']
        names = list(dict.fromkeys(['fixed8',selection['best_fixed'],broad,candidate]))
        assert not set(seeds).intersection(read(out/'validation-seeds.json'))
    assert not set(seeds).intersection(protocol['train'] + protocol['diagnostic'] + protocol['additional'])
    roots, excluded = [], {}
    for seed in seeds:
        src = folder/f'source-{seed}'
        retry = folder/f'source-retry-{seed}'
        if retry.exists():
            src = retry
        manifest = src/'combat-checkpoint/manifest.json'
        report_path = src/'report.json'
        receipt = read(report_path) if report_path.exists() else {}
        if not manifest.exists() or not (receipt.get('combat_checkpoint') or {}).get('accepted'):
            excluded[seed] = receipt.get('reason','capture failed before receipt')
            continue
        frame = read(manifest)['barrier']['frame']
        row = root_obs(src/'bot.jsonl',frame)
        if row.get('owner')!='provider' or row.get('features') is None:
            excluded[seed] = 'initial frame outside learned controller'
            continue
        if rocket:
            assert row['features'][845+7] == 1
        roots.append(dict(seed=seed, source=str(src), frame=frame, split='new_validation',
             map=read(folder/f'fixture-{seed}.json')['map'], manifest_hash=hashlib.sha256(manifest.read_bytes()).hexdigest(),
             feature_hash=hashlib.sha256(json.dumps(row['features']).encode()).hexdigest()))
    save(folder/'new-roots.json',dict(roots=roots,excluded=excluded,names=names))
    collect(out,roots,names,workers,fixture_rocket=False,port=50810,audit_duration=True)
    save(folder/'new-validation.json',dict(accepted=len(roots),candidate=candidate,matched_train_period=None if rocket else period,
                                        maps=dict(Counter(r['map'] for r in roots)),
                                        excluded=excluded, result=paired(out,roots,names)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--rocket',action='store_true')
    args=parser.parse_args()
    run(args.output.resolve(),args.workers,args.rocket)

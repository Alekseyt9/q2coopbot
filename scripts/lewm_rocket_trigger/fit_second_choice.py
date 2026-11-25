"""Fit a second-decision readout from matched native TRAIN branches."""
import argparse
from itertools import product
from pathlib import Path

import numpy as np

from experiment import BANK, read, save
from mode_fit import MODES, encode_features, fit_layer


def fit(source, output):
    plan = read(source/'plan.json')
    frozen_roots = read(BANK/'lewm-mode-choice-v1/fit-plan.json')['roots']
    if plan['second_frame'] != 21 or sorted(plan['roots'], key=lambda r:r['seed']) != sorted(frozen_roots, key=lambda r:r['seed']):
        raise ValueError('Expected the original training cohort, without diagnostic or test worlds')
    proof = read(source/'result.json')
    rows = proof['paired']
    if set(row['seed'] for row in rows) & set(proof['skipped']):
        raise ValueError('A training seed is both admitted and excluded')
    if sorted([row['seed'] for row in rows] + proof['skipped']) != sorted(row['seed'] for row in frozen_roots):
        raise ValueError('Training receipts do not account for every world')
    maps = np.array([row['map'] for row in rows])
    samples = [read(source/f"{r['seed']}-prior20.json")['branch_inspection'] for r in rows]
    x = encode_features(np.array([sample['features'] for sample in samples]))
    incumbent = np.array([MODES.index(row['incumbent_mode']) for row in rows])
    score, blast, alive = (np.array([[r['outcomes'][mode][key] for mode in MODES] for r in rows])
                           for key in ('score', 'own_blast', 'alive'))
    index = np.arange(len(rows))
    folds = [maps == level for level in sorted(set(maps))]
    base = tuple(int(values[index, incumbent].sum()) for values in (score, blast, alive))
    feature_sets = {'weapon': (np.arange(64,76),1), 'world-and-weapon': (np.arange(76),16)}
    configs = []
    for (label, (columns, latent_penalty)), ridge, risk in product(
            feature_sets.items(), (8,32,128), (0,1,3)):
        target = score - risk*blast
        estimates = np.zeros(score.shape)
        for mask in folds:
            w, b = fit_layer(x[~mask],target[~mask],columns,ridge,latent_penalty)
            estimates[mask] = x[mask] @ w.T + b
        best = estimates.argmax(axis=1)
        predicted_gain = estimates[index,best] - estimates[index,incumbent]
        for margin in (5,15,30):
            picked = np.where(predicted_gain > margin,best,incumbent)
            sums = tuple(int(values[index,picked].sum()) for values in (score,blast,alive))
            configs.append(dict(features=label,ridge=ridge,risk=risk,margin=margin,
                                score=sums[0],blast=sums[1],alive=sums[2],switches=int((picked!=incumbent).sum())))
    eligible = [c for c in configs if c['score'] >= base[0] and c['blast'] <= base[1] and c['alive'] >= base[2]
                and (c['score'] > base[0] or c['blast'] < base[1] or c['alive'] > base[2])]
    output.mkdir(parents=True, exist_ok=False)
    selected = max(eligible,key=lambda c:(c['score'],-c['blast'],c['alive'],-c['switches'])) if eligible else None
    save(output/'fit.json',dict(worlds=len(rows),maps=sorted(set(maps)),baseline=dict(zip(('score','blast','alive'),base)),
                                eligible=len(eligible),candidate=selected,configs=configs))
    if selected is None:
        print('No eligible held-map-out choice',flush=True)
        return
    columns, latent_penalty = feature_sets[selected['features']]
    weights, bias = fit_layer(x,score-selected['risk']*blast,columns,selected['ridge'],latent_penalty)
    model = read(source/'prior20.json')
    model['Scheduler']['SecondReadout'] = dict(weight=weights.tolist(),bias=bias.tolist())
    model['Scheduler']['SecondSwitchMargin'] = selected['margin']
    save(output/'learned.json',model)
    print(selected,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    fit(args.source.resolve(), args.output.resolve())

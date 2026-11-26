"""Run a fitted second-choice controller on unseen native saves."""
import argparse
from pathlib import Path
import shutil

import numpy as np

from experiment import BANK, collect, read, save
from long_eval import initial_objective
from mode_eval import report, roots, stage
from mode_fit import encode_features
from second_choice import inspect_branch
from temporal import build_temporal


def evaluate(out, fit, workers):
    live = out/'live'
    if not live.exists():
        live.mkdir()
        build_temporal(live)
    training = BANK/'lewm-mode-choice-v1/fit-plan.json'
    if not (out/'fit-plan.json').exists():
        shutil.copyfile(training, out/'fit-plan.json')
    elif read(out/'fit-plan.json') != read(training):
        raise ValueError('Unexpected training seed list')
    rows, excluded = roots(out)
    if excluded or not rows:
        raise ValueError(f'Fresh cohort incomplete: {excluded}')
    screened = {r['seed'] for r in read(BANK/'lewm-80-return-v1/test-plan.json')['roots']}
    if any(r['split'] != 'test' or r['seed'] in screened for r in rows):
        raise ValueError('Expected unseen test saves')
    for row in rows:
        row['initial_objective_count'] = initial_objective(row)
        if row['initial_objective_count'] <= 0:
            raise ValueError(f"Test save has no active or pending objective: {row['seed']}")
    plan = dict(roots=rows, horizons=[80,160])
    if (out/'test-plan.json').exists() and read(out/'test-plan.json') != plan:
        raise ValueError('Frozen test cohort changed')
    save(out/'test-plan.json',plan)
    shutil.copyfile(BANK/'lewm-mode-choice-v1/select20.json',out/'prior20.json')
    shutil.copyfile(fit/'learned.json',out/'second20.json')
    names = ('prior20','second20')
    stage(out,names)
    collect(live,rows,names,workers,fixture_rocket=False,port=29920,audit_duration=True,
            followup_frames=80,final_frames=160,score_objective=True,inspect_branch=inspect_branch)
    pairs = []
    for row in rows:
        probes = [read(live/f"{row['seed']}-{name}.json")['branch_inspection'] for name in names]
        before, after = probes
        if before['reached'] != after['reached'] or before.get('prefix_sha256') != after.get('prefix_sha256'):
            raise ValueError(f"Decision prefixes differ for {row['seed']}")
        if before['reached']:
            pairs.append((row['seed'],before,after))
    if not pairs:
        raise ValueError('No confirmed second decision in the test cohort')
    scheduler = read(live/'second20.json')['Scheduler']
    layer = scheduler['SecondReadout']
    x = encode_features(np.array([before['features'] for _,before,_ in pairs]))
    estimate = x @ np.array(layer['weight']).T + layer['bias']
    changed = 0
    for (seed,before,after), scores in zip(pairs,estimate):
        choice = before['second_choice']
        best = int(scores.argmax())
        predicted = best if scores[best]-scores[choice] > scheduler['SecondSwitchMargin'] else choice
        if predicted != after['second_choice']:
            raise ValueError(f"Go and Python disagree at the second choice for {seed}")
        changed += int(predicted != choice)
    results = {str(n):report(out,rows,names,n,baseline='prior20') for n in (80,160)}
    save(out/'test-results.json',dict(second_choice_admitted=len(pairs),second_choice_changed=changed,horizons=results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('fit',type=Path)
    parser.add_argument('--workers',type=int,default=3)
    args = parser.parse_args()
    evaluate(args.output.resolve(),args.fit.resolve(),args.workers)

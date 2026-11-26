"""Measure the effect of changing only the second decision in a native replay."""
import argparse
from collections import Counter
from functools import partial
import hashlib
import json
from pathlib import Path
import re

from experiment import BANK, blast_hp, collect, read, root_obs, save
from mode_fit import MODES
from temporal import build_temporal, utility


POLICIES = ('prior20', *MODES)
FRAME = 21
CHOICE = re.compile(r'lewm_mode frame=(\d+) index=(\d+)')
DAMAGE_FRAME = re.compile(r'\bserver_frame=(\d+)\b')


def inspect_branch(directory, run, decision_frame=FRAME):
    window = run['longer'] if decision_frame > 40 else run
    choices = dict((int(frame), int(index)) for frame, index in
                   (CHOICE.fullmatch(line).groups() for line in window['modes']))
    if decision_frame not in choices or 1 not in choices:
        return dict(reached=False)
    prefix = {}
    with (directory/'bot.jsonl').open(encoding='utf-8') as trace:
        for line in trace:
            row = json.loads(line)
            frame = row['frame']
            if not 1 <= frame < decision_frame:
                continue
            selection = (row.get('combat_policy') or {}).get('selection') or {}
            if frame in prefix or selection.get('owner') != 'provider' or not row['sent_command']:
                return dict(reached=False)
            prefix[frame] = row['sent_command']
    if sorted(prefix) != list(range(1, decision_frame)):
        return dict(reached=False)
    decision = root_obs(directory/'bot.jsonl', decision_frame)
    if decision['owner'] != 'provider' or decision['features'] is None:
        return dict(reached=False)
    # The restored native server assigns a new spawncount on each connection.
    # summarize() already checked each event against that run's generation.
    prior_damage = [re.sub(r'\bspawncount=\d+ ', '', line, count=1) for line in window['events']
                    if int(DAMAGE_FRAME.search(line)[1]) <= decision_frame]
    proof = json.dumps([prefix, decision['features'], prior_damage, choices[1]], sort_keys=True)
    return dict(reached=True, prefix_sha256=hashlib.sha256(proof.encode()).hexdigest(),
                second_choice=choices[decision_frame], features=decision['features'])


def run(out, cases_per_map, workers, resume=False, source_plan=None, decision_index=2, port=29900):
    model = read(BANK/'lewm-mode-choice-v1/select20.json')
    if model['Scheduler']['Frames'] != FRAME - 1:
        raise ValueError('Unexpected incumbent decision period')
    source_plan = source_plan or BANK/'lewm-mode-choice-v1/fit-plan.json'
    original_plan = read(source_plan)['roots']
    cover = Counter()
    cases = []
    for row in sorted(original_plan, key=lambda r: hashlib.sha256(str(r['seed']).encode()).digest()):
        if cover[row['map']] < cases_per_map:
            cases.append(row)
            cover[row['map']] += 1
    decision_frame = 1+(decision_index-1)*model['Scheduler']['Frames']
    horizon = decision_frame+59
    plan = dict(roots=cases, second_frame=decision_frame, horizon=horizon)
    if resume:
        if read(out/'plan.json') != plan or not (out/'rocket-client.exe').is_file():
            raise ValueError('Resume requires the original plan and client')
    else:
        out.mkdir(parents=True, exist_ok=False)
        save(out/'plan.json', plan)
        build_temporal(out)
    for i, name in enumerate(POLICIES):
        if i:
            model['Scheduler']['SecondChoice'] = i - 1
            if decision_index != 2:
                model['Scheduler']['ProbeDecision'] = decision_index
        path = out/f'{name}.json'
        if resume and read(path) != model:
            raise ValueError(f'Existing replay model differs for {name}')
        if not resume:
            save(path, model)
    collect(out, cases, POLICIES, workers, fixture_rocket=False, port=port,
            audit_duration=True, followup_frames=horizon,
            inspect_branch=partial(inspect_branch, decision_frame=decision_frame))
    paired, skipped = [], []
    for row in cases:
        runs = [read(out/f"{row['seed']}-{name}.json") for name in POLICIES]
        probes = [result['branch_inspection'] for result in runs]
        if not all(p['reached'] for p in probes):
            if any(p['reached'] for p in probes):
                raise ValueError(f"Branch reach differs for seed {row['seed']}")
            skipped.append(row['seed'])
            continue
        if len({p['prefix_sha256'] for p in probes}) != 1 or [p['second_choice'] for p in probes[1:]] != list(range(len(MODES))):
            raise ValueError(f"Invalid branch pairing for seed {row['seed']}")
        outcomes = {name: dict(score=utility(result['longer']), own_blast=blast_hp(result['longer']),
                               alive=result['longer']['alive']) for name, result in zip(POLICIES, runs)}
        control = outcomes['prior20']
        if outcomes[MODES[probes[0]['second_choice']]] != control:
            raise ValueError(f"Native same-choice replay diverged for seed {row['seed']}")
        delta = max(outcome['score'] for outcome in outcomes.values()) - control['score']
        safe_delta = max(outcome['score'] for outcome in outcomes.values()
                         if outcome['own_blast'] <= control['own_blast']
                         and (not control['alive'] or outcome['alive'])) - control['score']
        blast_saved = control['own_blast'] - min(outcome['own_blast'] for outcome in outcomes.values()
                                              if outcome['score'] >= control['score']
                                              and (not control['alive'] or outcome['alive']))
        paired.append(dict(seed=row['seed'], map=row['map'], outcomes=outcomes,
                           rocket=row.get('rocket'), incumbent_mode=MODES[probes[0]['second_choice']],
                           best_delta=delta, constrained_delta=safe_delta, blast_saved=blast_saved))
    save(out/'result.json', dict(paired=paired, skipped=skipped,
                                 improved=sum(row['best_delta'] > 0 for row in paired),
                                 total_best_delta=sum(row['best_delta'] for row in paired),
                                 total_constrained_delta=sum(row['constrained_delta'] for row in paired),
                                 total_blast_saved=sum(row['blast_saved'] for row in paired)))
    print(f"Verified {len(paired)} pairs at frame {decision_frame}, {len(skipped)} did not reach it", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cases-per-map', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--source-plan', type=Path)
    parser.add_argument('--decision-index', type=int, choices=range(2,7), default=2)
    parser.add_argument('--port', type=int, default=29900)
    args = parser.parse_args()
    run(args.output.resolve(), args.cases_per_map, args.workers, args.resume, args.source_plan,
        args.decision_index, args.port)

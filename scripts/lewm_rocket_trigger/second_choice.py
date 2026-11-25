"""Measure the effect of changing only the second decision in a native replay."""
import argparse
from collections import Counter
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


def inspect_branch(directory, run):
    choices = dict((int(frame), int(index)) for frame, index in
                   (CHOICE.fullmatch(line).groups() for line in run['modes']))
    if FRAME not in choices or 1 not in choices:
        return dict(reached=False)
    prefix = {}
    with (directory/'bot.jsonl').open(encoding='utf-8') as trace:
        for line in trace:
            row = json.loads(line)
            frame = row['frame']
            if not 1 <= frame < FRAME:
                continue
            selection = (row.get('combat_policy') or {}).get('selection') or {}
            if frame in prefix or selection.get('owner') != 'provider' or not row['sent_command']:
                return dict(reached=False)
            prefix[frame] = row['sent_command']
    if sorted(prefix) != list(range(1, FRAME)):
        return dict(reached=False)
    decision = root_obs(directory/'bot.jsonl', FRAME)
    if decision['owner'] != 'provider' or decision['features'] is None:
        return dict(reached=False)
    # The restored native server assigns a new spawncount on each connection.
    # summarize() already checked each event against that run's generation.
    prior_damage = [re.sub(r'\bspawncount=\d+ ', '', line, count=1) for line in run['events']
                    if int(DAMAGE_FRAME.search(line)[1]) <= FRAME]
    proof = json.dumps([prefix, decision['features'], prior_damage, choices[1]], sort_keys=True)
    return dict(reached=True, prefix_sha256=hashlib.sha256(proof.encode()).hexdigest(),
                second_choice=choices[FRAME], features=decision['features'])


def run(out, cases_per_map, workers, resume=False, source_plan=None):
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
    plan = dict(roots=cases, second_frame=FRAME, horizon=80)
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
        path = out/f'{name}.json'
        if resume and read(path) != model:
            raise ValueError(f'Existing replay model differs for {name}')
        if not resume:
            save(path, model)
    collect(out, cases, POLICIES, workers, fixture_rocket=False, port=29900,
            audit_duration=True, followup_frames=80, inspect_branch=inspect_branch)
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
    print(f"Verified {len(paired)} pairs at frame {FRAME}, {len(skipped)} did not reach it", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cases-per-map', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--source-plan', type=Path)
    args = parser.parse_args()
    run(args.output.resolve(), args.cases_per_map, args.workers, args.resume, args.source_plan)

"""Prepare fresh validation saves before fitting the contextual mode selector."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import subprocess

from build import HERE
from experiment import BANK, read, save


def prepare(out, split='validation', start=36, count=28, per_group=2, exclude_plans=()):
    old = BANK/'lewm-duration-v1'
    trigger = BANK/'lewm-rocket-trigger-v1'
    used = set(read(old/'validation-seeds.json') + read(old/'rocket-validation/validation-seeds.json')
               + read(trigger/'validation-seeds.json'))
    for plan in exclude_plans:
        used.update(row['seed'] for row in read(plan)['roots'])
    descriptors = sorted({r['descriptor'] for r in read(BANK/'fragment-autonomous-train-20261117-052625/selected.json')})
    groups = defaultdict(dict)
    for offset in range(start,start+count):
        plan = out/f'worlds-{offset}.json'
        cmd = [trigger/'q2episode.exe','--episodes',','.join(descriptors),'--split',split,
               '--mode','rules','--count','1','--seed-offset',str(offset),'--out',plan,
               '--artifacts',out/f'fixtures-{offset}']
        proc = subprocess.run(cmd,capture_output=True)
        if proc.returncode:
            continue
        for task in read(plan)['tasks']:
            for fixture in task['instances']:
                seed = fixture['engine_seed']
                if seed not in used and fixture['split']==split:
                    groups[fixture['map'], fixture['inventory']['initial_weapon']=='Rocket Launcher'][seed] = fixture
    cases = []
    for (map_name, rocket), group in sorted(groups.items()):
        for seed in sorted(group,key=lambda s:hashlib.sha256(str(s).encode()).digest())[:per_group]:
            save(out/f'fixture-{seed}.json',group[seed])
            cases.append(dict(seed=seed,map=map_name,rocket=rocket))
    if not cases:
        raise ValueError('No new fixtures in requested seed range')
    save(out/'capture-plan.json',cases)
    return cases


def run(out, retry, split, start, count, per_group, workers, exclude_plans):
    cases = read(out/'capture-plan.json') if retry else prepare(out, split, start, count, per_group, exclude_plans)
    def worker(i):
        for index in range(i,len(cases),workers):
            case=cases[index]
            seed=case['seed']
            receipt=out/f'source-{seed}/report.json'
            if retry and receipt.exists() and (read(receipt).get('combat_checkpoint') or {}).get('accepted'):
                continue
            args=['pwsh','-NoProfile','-File',HERE/'duration_validation.ps1','-OutputRoot',out,'-CaptureSeed',
                  str(seed),'-Port',str((28540 if retry else 28440)+index)]
            if case['rocket']:
                args.append('-RocketCohort')
            if retry:
                args.append('-RetryFailed')
            prefix='capture-retry' if retry else 'capture'
            with (out/f'{prefix}-{seed}.log').open('w') as log:
                subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(worker,range(workers)))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path)
    ap.add_argument('--retry-failed',action='store_true')
    ap.add_argument('--split',choices=['validation','test'],default='validation')
    ap.add_argument('--start',type=int,default=36)
    ap.add_argument('--count',type=int,default=28)
    ap.add_argument('--per-group',type=int,default=2)
    ap.add_argument('--workers',type=int,choices=range(1,9),default=4)
    ap.add_argument('--exclude-plan',type=Path,action='append',default=[])
    args=ap.parse_args()
    run(args.output.resolve(),args.retry_failed,args.split,args.start,args.count,args.per_group,args.workers,args.exclude_plan)

"""Evaluate longer-trained selectors with native encounter completion and HP outcomes."""
import argparse
from pathlib import Path
import re
import shutil

from experiment import collect, read, save
from long_return import PRIOR
from mode_eval import report, roots, stage


def initial_objective(row):
    src=Path(row['source'])
    gen=read(src/'combat-checkpoint/manifest.json')['barrier']['generation']
    pattern=re.compile(rf'^sv_test_objective spawncount={gen} server_frame={row["frame"]} '
                       rf'g_test_objective version=1 map={row["map"]} game_frame=\d+ alive=(\d+) pending=(\d+)$')
    with (src/'server.log').open(errors='replace') as log:
        matches=[m.groups() for line in log if (m:=pattern.fullmatch(line.strip()))]
    if len(matches)!=1:
        raise ValueError(f"Missing or ambiguous source objective for {row['seed']}")
    return sum(map(int,matches[0]))


def run(out, archival, controls, workers):
    live=out/'live'
    live.mkdir(exist_ok=True)
    if not (live/'rocket-client.exe').exists():
        shutil.copyfile(PRIOR/'live/rocket-client.exe',live/'rocket-client.exe')
    if not (out/'prior20.json').exists():
        shutil.copyfile(PRIOR/'select20.json',out/'prior20.json')
    names=['fixed8','fixed12','prior20'] if controls else ['fixed8','fixed12','prior20','long20','safe20']
    stage(out,names)
    if archival:
        cases=read(PRIOR/'evaluation-plan.json')['roots']
        for row in cases:
            for name,original in [('fixed8','fixed8'),('fixed12','fixed12'),('prior20','select20')]:
                receipt=live/f"{row['seed']}-{name}.json"
                if not receipt.exists():
                    shutil.copyfile(PRIOR/'live'/f"{row['seed']}-{original}.json",receipt)
    else:
        cases,excluded=roots(out)
        if not controls and excluded:
            raise ValueError(f'Incomplete test cohort: {excluded}')
        if any(row['split'] != 'test' for row in cases):
            raise ValueError('Unexpected split in the test fixtures')
        already_used={r['seed'] for r in read(PRIOR/'evaluation-plan.json')['roots']}
        if already_used.intersection(r['seed'] for r in cases):
            raise ValueError('Overlap with previous validation worlds')
        for row in cases:
            row['initial_objective_count']=initial_objective(row)
        if not controls:
            save(out/'test-plan.json',dict(roots=cases,excluded=excluded,primary='safe20',horizons=[40,80,160]))
    collect(live,cases,names,workers,fixture_rocket=False,port=29400 if archival else 29500,
            audit_duration=True,followup_frames=80,final_frames=0 if archival else 160,score_objective=not archival)
    if not controls:
        results={str(n):report(out,cases,names,n) for n in ([40,80] if archival else [40,80,160])}
        save(out/('previous-validation.json' if archival else 'test-results.json'),results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--archival',action='store_true')
    parser.add_argument('--controls-only',action='store_true')
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    run(args.output.resolve(),args.archival,args.controls_only,args.workers)

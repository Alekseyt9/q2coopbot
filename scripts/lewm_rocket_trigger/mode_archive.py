"""Compare the frozen readout on archived worlds excluded from its fit."""
import argparse
from pathlib import Path

from duration import cohorts
from experiment import BANK, collect, read, root_obs, save
from mode_eval import report, stage
from mode_fit import MODES


def run(out):
    _,old,extra=cohorts()
    prior=BANK/'lewm-duration-v1'
    local=read(prior/'local-selection.json')['best']
    live=out/'live'
    train={r['seed'] for r in read(out/'fit-plan.json')['roots']}
    assert not train.intersection(r['seed'] for r in old+extra)
    names=[*MODES,'select40','select20','comparison20']
    stage(out,names)
    for r in old+extra:
        r['rocket']=root_obs(Path(r['source'])/'bot.jsonl',r['frame'])['features'][852]==1
        for name,prev in zip(MODES,['fixed8','fixed12',local]):
            old_run=read(prior/f"{r['seed']}-{prev}.json")
            old_run['modes']=[]
            save(live/f"{r['seed']}-{name}.json",old_run)
    collect(live,old+extra,names,7,fixture_rocket=False,port=28940,audit_duration=True)
    save(out/'archive.json',dict(original=report(out,old,names),additional=report(out,extra,names)))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path)
    run(ap.parse_args().output.resolve())

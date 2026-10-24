"""Finalize one prepared model's TRAIN subset from an unchanged shared native pool."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from evaluate_registered_combat_models import read, save, sha
from combat_execution_contract import require_same


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','pool'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--plan-index',type=int,required=True)
    a=ap.parse_args();repo=Path(__file__).resolve().parents[1]
    prepared=read(a.corpus/'prepared.json');assert prepared['state']=='prepared'
    assert prepared['protocol_sha256']==sha(a.corpus/'protocol.json') and not (a.corpus/'complete.json').exists()
    protocol=read(a.corpus/'protocol.json');assert protocol['train_only'] and not protocol['diagnostic_intervention']
    assert len(protocol['assignments'])==1
    assignment=protocol['assignments'][0];plan=read(assignment['plan'])
    assert sha(assignment['plan'])==assignment['plan_sha256']
    pool=read(a.pool/'report.json');digest=sha(a.pool/'report.json')
    assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==len(pool['jobs'])
    assert all(not j['error'] for j in pool['jobs'])
    binding=pool['plans'][a.plan_index]
    assert binding['sha256']==assignment['plan_sha256']
    selected=[j for j in pool['jobs'] if j['plan_index']==a.plan_index]
    assert len(selected)==prepared['native_cases_planned']==24
    assert {j['seed'] for j in selected}==set(assignment['seeds']) and all(j['mode']=='learned' for j in selected)
    # Only an immutable receipt is copied; all actual captures retain their
    # absolute shared-pool paths. No runtime, assets or traces are duplicated.
    mirror=a.corpus/'pool';mirror.mkdir()
    shutil.copy2(a.pool/'report.json',mirror/'report.json')
    target=a.corpus/'processing-train'
    with (a.corpus/'processing-train.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,str(repo/'scripts/process_registered_combat_pool_cuda.py'),
            '--pool',str(a.pool.resolve()),'--plan-index',str(a.plan_index),'--out',str(target.resolve())],
            cwd=repo,stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert result.returncode==0,str(a.corpus/'processing-train.log')
    report=read(target/'report.json');meta=read(target/'merged/report.json')
    assert report['state']=='complete' and report['device']=='cuda' and not report['training_performed']
    assert len(report['cases'])==24 and {c['seed'] for c in report['cases']}==set(assignment['seeds'])
    require_same(protocol['combat_execution_contract'],plan.get('combat_execution_contract'),meta.get('combat_execution_contract'))
    for item in protocol['inputs']:assert sha(item['path'])==item['sha256'],item['path']
    assert sha(a.pool/'report.json')==sha(mirror/'report.json')==digest
    processed=[dict(label='train',cases=24,rows=report['rows'],report=str((target/'report.json').resolve()),sha256=sha(target/'report.json'))]
    save(a.corpus/'complete.json',dict(state='complete',version='combat_execution_contract_corpus_v1',
        protocol_sha256=sha(a.corpus/'protocol.json'),pool_sha256=digest,source_pool_report=str((a.pool/'report.json').resolve()),
        shared_native_cases=len(pool['jobs']),selected_plan_index=a.plan_index,processed=processed,
        native_cases=24,cuda_cases=24,training_performed=False,promotion=False))
    save(a.corpus/'progress.json',dict(stage='complete',processed=processed))
    print(dict(cases=24,rows=report['rows'],shared_pool_cases=len(pool['jobs'])))


if __name__=='__main__':main()

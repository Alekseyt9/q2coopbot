"""Seal all native zero-core proofs; never convert a failed case into quality data."""
import argparse
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or h==expected,str(path);pins[str(path)]=h;return path
    closed=read(pin(a.collection/'complete.json'));assert closed['version']=='combat_native_memory_recovery_zero_v1'
    spec=read(pin(a.collection/'spec.json',closed['spec_sha256']));proofs=[]
    for u in spec['contexts']:
        folder=a.collection/'proofs'/u['unit_id'];seal=read(pin(folder/'complete.json'))
        proof=read(pin(folder/'report.json',seal['report_sha256']));assert proof['context']==u['context']
        proofs.append(dict(context=u['context'],gate_passed=proof['gate_passed'],active_memory_rows=proof['active_memory_rows'],
            report=str((folder/'report.json').resolve()),report_sha256=sha(folder/'report.json')))
    assert len(proofs)==4
    a.out.mkdir(parents=True);report=dict(state='complete',version='combat_memory_recovery_native_zero_report_v1',gate_passed=all(p['gate_passed'] for p in proofs),
        native_cases=8,cuda_cases=8,proofs=proofs,active_memory_rows=sum(p['active_memory_rows'] for p in proofs),inputs=pins,
        training_performed=False,quality_evaluation=False,model_quality_admitted=False,promotion=False)
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(gate_passed=report['gate_passed'],active_rows=report['active_memory_rows']))


if __name__=='__main__':main()

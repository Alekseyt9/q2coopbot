"""Offline-only processing recovery: preserve rejected exports and all native outcomes."""
import argparse
from pathlib import Path
import subprocess
import sys
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--screen',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args();root=a.screen.resolve();out=a.out.resolve();assert not out.exists();out.mkdir(parents=True)
    spec=read(root/'spec.json');pool=read(root/'pool/report.json')
    assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==spec['native_budget']
    jobs={j['plan_index']:j for j in pool['jobs']};cases=[];tasks=[];pins={}
    def pin(p):p=Path(p).resolve();pins[str(p)]=sha(p);return p
    pin(root/'spec.json');pin(root/'pool/report.json');pin(root/'progress.json')
    for v in spec['variants']:
        if v['label']=='rules':continue
        i=v['index'];d=root/f'case-{i:02d}';native=d/'native';capture=Path(jobs[i]['root'])
        report=read(pin(capture/'report.json'));manifest=read(pin(capture/'manifest.json'));r=report['results'][0]
        assert report['capture_complete'] and report['provenance_valid'] and r['capture_valid'] and r['dispatch_valid'] and r['seed_confirmed']
        assert r['generated_fixture']==v['context']['instance'] and sha(v['model'])==v['model_sha256']==manifest['model_weights_sha256'].lower()
        replay=read(pin(native/'replay-0/report.json'));assert replay['command_proof']['accepted']
        pin(native/'replay-0/steps.jsonl');pin(native/'replay-0/server_outcomes.jsonl');pin(v['model'])
        if not (native/'report.json').exists():
            log=pin(root/f'export-{i:02d}.log');reason=log.read_text().strip()
            assert reason.startswith('unfinished control handoff requires explicit nonterminal_v1 objective:')
            assert manifest['reward_config'].get('handoff_bootstrap','')==''
            cases.append(dict(index=i,state='rejected_before_cuda',reason=reason,export_log=str(log),
                native_replay_valid=True,training_eligible=False,raw_native_outcome_retained=True))
            continue
        meta=read(pin(native/'report.json'))
        assert meta['model_sha256']==v['model_sha256'] and meta['version']=='combat_ppo_native_cuda_pending_v1'
        assert not (d/'cuda').exists(), 'Never overwrite a prior CUDA result'
        cases.append(dict(index=i,state='accepted_native_export',native_report_sha256=sha(native/'report.json')))
        tasks.append(dict(model=v['model'],model_sha256=v['model_sha256'],data=str(native),out=str(d/'cuda'),
                          native_report_sha256=sha(native/'report.json')))
    assert len(cases)==spec['cuda_budget'] and tasks
    save(out/'export-admission.json',dict(version='combat_native_export_admission_v1',pool_report_sha256=sha(root/'pool/report.json'),
        cases=cases,inputs=pins,new_native_games=0,original_partial_artifacts_preserved=True,original_objectives_unchanged=True))
    request=out/'cuda-request.json'
    save(request,dict(version='combat_cuda_finalize_batch_v1',compact_finalized_streams=True,tasks=tasks))
    save(out/'progress.json',dict(stage='cuda-verification',accepted=len(tasks),rejected=len(cases)-len(tasks)))
    command=[sys.executable,str(Path(__file__).with_name('finalize_combat_cuda_batch.py')),'--request',str(request),'--receipt-root',str(out/'cuda-batch')]
    with (out/'cuda-batch.log').open('w') as f:
        subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    assert read(out/'cuda-batch/report.json')['state']=='complete'
    assert all(sha(p)==h for p,h in pins.items())
    save(out/'progress.json',dict(stage='complete_with_rejections' if len(tasks)<len(cases) else 'complete',
        accepted_cuda_cases=len(tasks),requested_cuda_cases=len(cases),new_native_games=0))
    print(dict(accepted=len(tasks),rejected=len(cases)-len(tasks),new_native_games=0),flush=True)


if __name__=='__main__':main()

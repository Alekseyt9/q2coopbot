"""Publish a diagnostic-only reference index without rewriting numerical proofs."""
import argparse
import copy
import os
from pathlib import Path
import shutil
from process_combat_architecture_pool import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args();root=args.root.resolve();assert not (root/'analysis-complete.json').exists()
    complete=read(root/'complete.json');summary=read(root/'summary.json');spec=read(root/'spec.json')
    batch=read(root/'cuda-batch/report.json')
    assert complete['state']==summary['state']==batch['state']=='complete'
    assert complete['original_protocol_passed'] is False and len(batch['tasks'])==len(spec['assignments'])==24
    publish=root/'diagnostic-references';assert not publish.exists();publish.mkdir()
    entries=[]
    for a in spec['assignments']:
        raw=root/('replay-'+str(a['plan_index']))/'cuda';meta=read(raw/'report.json')
        assert meta['numerical_verification']=='cuda_verified_v1' and meta['model_sha256']==a['model_sha256']
        item=batch['tasks'][a['plan_index']]
        assert item['report_sha256']==sha(raw/'report.json') and item['cuda_verification_sha256']==sha(raw/'cuda-verification.json')
        target=publish/str(a['plan_index']);target.mkdir()
        # Reports are copied, never hardlinked: scope amendments cannot alter
        # the original sealed report. Large immutable streams share storage.
        for file in raw.iterdir():
            if file.is_file() and file.name!='report.json':os.link(file,target/file.name)
        diagnostic=copy.deepcopy(meta)
        diagnostic.update(version='combat_diagnostic_reference_rollout_v1',diagnostic_only=True,
            ordinary_training_eligible=False,ppo_training_ready=False,
            reference_report=str(raw/'report.json'),reference_report_sha256=sha(raw/'report.json'),
            purpose='reused_train_counterfactual_mechanism_screen',numerical_proof_scope='Unchanged original model/data; only this derived report adds diagnostic corpus purpose. Raw source report and CUDA batch seal are preserved.')
        diagnostic['source_sha256'][str(raw/'report.json')]=sha(raw/'report.json')
        save(target/'report.json',diagnostic)
        assert read(target/'report.json')['diagnostic_only'] and not read(target/'report.json')['ppo_training_ready']
        entries.append(dict(**a,data=str(target),report_sha256=sha(target/'report.json'),diagnostic_only=True,
            raw_verified_data=str(raw),raw_report_sha256=sha(raw/'report.json')))
    save(root/'diagnostic-index.json',dict(state='complete',version='combat_diagnostic_reference_index_v1',entries=entries,
        diagnostic_only=True,ordinary_training_eligible=False,ppo_training_ready=False,training_performed=False,
        scope='All24 cases including unmodified baseline controls are reference-only. Original native/CUDA proofs remain sealed; derived reports block ordinary PPO. No duplicate storage of immutable streams.'))
    paths=[root/name for name in ('protocol.json','comparison.json','complete.json','summary.json','summary.md',
        'branch-effects.json','continuation-first-life.json','diagnostic-index.json','cuda-batch/report.json')]
    for name in ('seal_combat_anchor_screen.py','summarize_combat_anchor_screen.py','report_combat_anchor_continuation.py','recover_combat_anchor_screen.py','test_combat_anchor_domain.py'):
        file=Path(__file__).with_name(name).resolve();snapshot=root/('final-source-'+name)
        shutil.copyfile(file,snapshot);paths.append(snapshot)
    save(root/'analysis-complete.json',dict(state='complete',native_cases=24,cuda_cases=24,unique_train_contexts=8,
        diagnostic_index_sha256=sha(root/'diagnostic-index.json'),summary_sha256=sha(root/'summary.json'),
        evidence={str(p):sha(p) for p in paths},original_protocol_passed=False,posthoc_scope_correction=True,
        original_v1_rejection_sha256=sha(root.parent/'comparison.json'),
        v2_rejection_sha256=sha(root.parent/'analysis-v2/comparison.json'),
        original_v3_reporter_failure_sha256=sha(root/'report_combat_anchor_continuation.log'),
        training_performed=False,promotion=False,quality_confirmation=False,
        scope='Completed corrected diagnostic analysis of ALL original captures; no restart or case deletion. Raw failed attempts preserved. Not original-protocol acceptance or independent learned-policy quality.'))
    print(root/'analysis-complete.json')


if __name__=='__main__':main()

"""Seal the experimental primary learner's native/CUDA pilot receipts."""
import argparse
import json
from pathlib import Path
from audit_coop_paired_export import read, sha, lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--cuda', type=Path, required=True)
    args = parser.parse_args()
    native, cuda = args.native.resolve(), args.cuda.resolve()
    before, after = read(native/'report.json'), read(cuda/'report.json')
    audit = read(cuda/'cuda-verification.json')
    assert before['version'] == 'combat_ppo_native_cuda_pending_v1'
    assert not before['paired_training_ready'] and before['paired_adapter_verified']
    assert after['version'] == 'combat_ppo_rollout_v1' and after['paired_training_ready']
    assert after['numerical_verification'] == 'cuda_verified_v1'
    assert after['paired_adapter_version'] == before['paired_adapter_version'] == 'coop_primary_native_adapter_v1'
    assert after['reward_version'] == before['reward_version'] == 'combat_reward_v11'
    assert audit['state'] == 'passed' and audit['device'] == 'cuda'
    assert audit['log_probability_max_error'] < .003 and audit['value_max_error'] < 1e-4
    assert max(audit['actor_memory_max_error'], audit['critic_memory_max_error']) < 2e-5
    for path, digest in after['source_sha256'].items():
        assert sha(Path(path)) == digest
    assert sha(native/'native-rollout.jsonl') == before['native_rollout_sha256']
    assert sha(cuda/'rollout.jsonl') == after['rollout_sha256']
    assert sha(cuda/'cuda-verification.json') == after['cuda_verification_sha256']
    rows = list(lines(cuda/'rollout.jsonl'))
    assert len(rows) == after['rows'] == before['rows'] == 80
    assert len(list(lines(cuda/'sequence.jsonl'))) == after['sequence_rows'] == 80
    assert all(r['sample']['version'] == after['policy_version'] and r['seed'] == r['sample']['sampling_seed'] for r in rows)
    assert all(len(r['features']) == len(r['next_features']) == 881 for r in rows)
    assert sum(r['terminal'] for r in rows) == 1 and rows[-1]['terminal']
    assert rows[-1]['next_value'] == 0 and rows[-1]['bootstrap_zero']
    result = dict(version='coop_primary_cuda_pilot_audit_v1', rows=len(rows),
                  paired_pilot_training_ready=True, training_performed=False,
                  registry_wide_eligibility=False,
                  source_sha256={str(path): sha(path) for path in
                                 (native/'report.json', cuda/'report.json', cuda/'cuda-verification.json', cuda/'rollout.jsonl', cuda/'sequence.jsonl')})
    (cuda/'paired-ppo-verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}))


if __name__ == '__main__': main()

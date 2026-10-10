"""Continue the sealed v7 A/B after the legacy float-sum replay repair.

Preserves completed captures/control update and failed processing evidence.
Starts fresh processing with a separately identified verifier; no recapture.
"""
import argparse
import pathlib
import sys

from process_combat_architecture_pool import read, save, sha, run
from run_combat_target_refresh import compile_plan
from run_combat_sampling_evaluation import collect_compressed
from train_combat_bc import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=pathlib.Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    repo = pathlib.Path(__file__).resolve().parents[1]
    protocol = read(root/'protocol.json')
    assert torch.cuda.is_available() and protocol['training_device'] == 'cuda'
    source = pathlib.Path(protocol['parent']).parents[1]
    quality = root/'quality'
    repaired = root/'q2ppo-data-repaired.exe'
    assert repaired.is_file()
    for name in ('control', 'quality'):
        pool = read(root/name/'capture/pool/report.json')
        assert pool['state'] == 'complete' and pool['source_unchanged']
        assert len(pool['jobs']) == protocol['training_episodes_per_branch']
        assert not any(j['error'] for j in pool['jobs'])
    failed = quality/'processing-failed-float-sum'
    processing = quality/'processing'
    if not failed.exists():
        assert read(processing/'report.json')['state'] == 'failed'
        assert not read(processing/'report.json')['training']
        # Keep all diagnostic exports and frozen inputs. Only this exact child moves.
        assert processing.parent == quality and failed.parent == quality
        processing.rename(failed)
    for name in ('ppo_recurrent.py', 'ppo_combat.py', 'finalize_combat_cuda_batch.py', 'finalize_combat_cuda_rollout.py'):
        assert sha(failed/'python-sources'/name) == sha(repo/'scripts'/name), 'Trainer changed since capture'
    if not processing.exists() or read(processing/'report.json')['state'] != 'complete':
        save(root/'progress.json', dict(stage='cuda_training', branch='quality', promotion=None))
        command = [sys.executable, repo/'scripts/process_combat_architecture_pool.py',
                   '--capture-root', quality/'capture', '--out', processing,
                   '--config', quality/'config.json', '--exporter', repaired,
                   '--anchor', source/'anchor.json', '--bank', source/'bank.json',
                   '--export-workers', '4', '--cuda-only-export', '--cuda-batch-finalize']
        if processing.exists(): command.append('--resume')
        run(command, quality/'process-repaired.log')
    save(root/'recovery.json', dict(state='processing_complete',
         original_processing_protocol_sha256=sha(failed/'protocol.json'),
         repaired_exporter_sha256=sha(repaired),
         scope='Same captured native exporter, inputs and CUDA trainers; verifier accepts only total-score roundoff with exact components and evidence. No recapture.'))
    evaluation = root/'evaluation'
    assert not evaluation.exists(), 'Evaluation already allocated; inspect its receipt before continuing'
    evaluation.mkdir()
    entries, plans = [], []
    for name in ('control', 'quality'):
        update = root/name/'processing'/name/'update'
        seal = read(update/'complete.json')
        for filename, field in [('weights.json','weights_sha256'), ('checkpoint.pt','checkpoint_sha256'), ('report.json','report_sha256')]:
            assert sha(update/filename) == seal[field]
        assert read(update/'report.json')['device'] == 'cuda'
        run([sys.executable, repo/'scripts/audit_combat_spatial_ppo_checkpoint_cuda.py', '--roots', update], root/name/'checkpoint-audit.log')
        model = read(update/'weights.json')
        model['deterministic'] = not protocol['validation_stochastic']
        weights = evaluation/(name+'-weights.json')
        save(weights, model)
        plan = compile_plan(root/'q2episode.exe', repo/'scripts/scenarios/combat-training/index.json', repo,
                            weights, evaluation/name, 'validation', protocol['families'], protocol['validation_seed_offset'], count=4)
        schedule = read(plan)
        schedule['policy_sampling_seed_offset'] = 20261011 if protocol['validation_stochastic'] else 0
        save(plan, schedule)
        plans.append(plan)
        entries.append(dict(model=name, label='after', root=schedule['output_root'], plan=str(plan),
                            plan_sha256=sha(plan), source_weights_sha256=sha(update/'weights.json'),
                            deterministic_weights_sha256=sha(weights), deterministic=model['deterministic'],
                            policy_sampling_seed_offset=schedule['policy_sampling_seed_offset']))
    count = protocol['validation_episodes_per_branch']
    save(evaluation/'protocol.json', dict(version='combat_action_quality_common_validation_v1', evaluations=entries,
         families=protocol['families'], total_episodes=2*count, episodes_per_model=count, slots=16, timescale=2,
         scope='Common development validation geometry/seeds; no final test or promotion.'))
    save(root/'progress.json', dict(stage='paired_validation', episodes=2*count, promotion=None))
    collect_compressed(repo, plans, evaluation)
    save(root/'progress.json', dict(stage='captures_complete', training_episodes=2*count,
         validation_episodes=2*count, evaluation_root=str(evaluation), scope='Quality report required; no promotion.'))
    run([sys.executable, repo/'scripts/report_combat_action_quality_ab.py', '--root', root], root/'report.log')


if __name__ == '__main__':
    main()

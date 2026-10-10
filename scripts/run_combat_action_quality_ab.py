"""Bounded fresh native reward-v6 A/B; objective forks and training are CUDA only."""
import argparse
import os
import pathlib
import shutil
import sys

from process_combat_architecture_pool import read, save, sha, run
from run_combat_target_refresh import compile_plan
from run_combat_sampling_evaluation import collect_compressed
from train_combat_bc import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=pathlib.Path, required=True)
    ap.add_argument('--seed-offset', type=int, default=320)
    args = ap.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    assert torch.cuda.is_available(), 'CUDA is required before preparing this experiment'
    assert not out.exists() and args.seed_offset >= 320
    assert shutil.disk_usage(out.parent).free > 10 * 1024**3
    out.mkdir()
    os.environ['GOCACHE'] = str(repo/'workspace/build/go-cache')
    os.environ['GOTOOLCHAIN'] = 'auto'
    parent = repo/'workspace/artifacts/parent-ppo-series-v1-20261010/round-2/processing/series/update'
    source = parent.parents[1]
    families = [f'campaign-{m}-site-{site}-{weapon}' for m in ('base1','base2')
                for site in ('02','03') for weapon in ('blaster','machinegun')]
    compiler, exporter = out/'q2episode.exe', out/'q2ppo-data.exe'
    for package, binary in [('q2episode', compiler), ('q2ppo-data', exporter)]:
        run(['go','build','-buildvcs=false','-o',binary,'./cmd/'+package], out/('build-'+package+'.log'))
    save(out/'protocol.json', dict(version='combat_action_quality_ab_v1', parent=str(parent),
        parent_weights_sha256=sha(parent/'weights.json'), families=families,
        train_seed_offset=args.seed_offset, validation_seed_offset=48,
        training_episodes_per_branch=32, validation_episodes_per_branch=32,
        slots=16, timescale=2, training_device='cuda', runtime_action_quality_guard=False,
        scope='Small development A/B, not final test. Same actor/std, critic-output and Adam reset in both branches. '
              'Reward changes only; fresh train split and separate common validation. No promotion.'))
    branches = []
    try:
        rewards = {'control': repo/'scripts/scenarios/combat-reward-recoil-v5.json',
                   'quality': repo/'scripts/scenarios/combat-reward-action-quality-v6.json'}
        for name, reward in rewards.items():
            branch = out/name
            branch.mkdir()
            config = read(source/'config.json')
            config['objective_reward_sha256'] = sha(reward)
            save(branch/'config.json', config)
            fork = branch/'fork'
            run([sys.executable,repo/'scripts/fork_combat_architecture_objective_cuda.py',
                 '--parent',parent,'--config',branch/'config.json','--reward',reward,'--out',fork], branch/'fork.log')
            registry = branch/'registry'
            registry.mkdir()
            files = []
            for family in families:
                filename = family+'.json'
                recipe = read(repo/'scripts/scenarios/combat-training'/filename)
                recipe['recipe']['reward_config'] = reward.relative_to(repo).as_posix()
                save(registry/filename, recipe)
                files.append(filename)
            save(registry/'index.json', dict(version=1, files=files))
            capture = branch/'capture'
            capture.mkdir()
            plan = compile_plan(compiler, registry/'index.json', repo, fork/'weights.json',
                                capture/'train', 'train', families, args.seed_offset, count=4)
            schedule = read(plan)
            assert sum(len(t['seeds']) for t in schedule['tasks']) == 32
            report = read(fork/'report.json')
            save(capture/'models.json', [dict(id=name, architecture=dict(architecture='attention'),
                model=str(fork/'weights.json'), plan=str(plan), capture_root=schedule['output_root'],
                resume_checkpoint=str(fork/'checkpoint.pt'), resume_checkpoint_sha256=sha(fork/'checkpoint.pt'),
                resume_report=str(fork/'report.json'), resume_report_sha256=sha(fork/'report.json'),
                parent_updates_completed=report['updates_completed'])])
            branches.append((name, branch, capture, plan))
        for name, branch, capture, plan in branches:
            save(out/'progress.json', dict(stage='collecting', branch=name, promotion=None))
            collect_compressed(repo, [plan], capture)
            save(out/'progress.json', dict(stage='cuda_training', branch=name, promotion=None))
            processing = branch/'processing'
            run([sys.executable,repo/'scripts/process_combat_architecture_pool.py','--capture-root',capture,
                 '--out',processing,'--config',branch/'config.json','--exporter',exporter,
                 '--anchor',source/'anchor.json','--bank',source/'bank.json','--export-workers','4',
                 '--cuda-only-export','--cuda-batch-finalize'], branch/'process.log')
            assert read(processing/'report.json')['state'] == 'complete'
        evaluation = out/'evaluation'
        evaluation.mkdir()
        entries, plans = [], []
        for name, branch, capture, plan in branches:
            update = branch/'processing'/name/'update'
            run([sys.executable,repo/'scripts/audit_combat_spatial_ppo_checkpoint_cuda.py',
                 '--roots',update],branch/'checkpoint-audit.log')
            model = read(update/'weights.json')
            model['deterministic'] = True
            weights = evaluation/(name+'-weights.json')
            save(weights, model)
            valid = compile_plan(compiler, repo/'scripts/scenarios/combat-training/index.json', repo,
                                 weights,evaluation/name,'validation',families,48,count=4)
            plans.append(valid)
            entries.append(dict(model=name,label='after',root=read(valid)['output_root'],plan=str(valid),
                plan_sha256=sha(valid),source_weights_sha256=sha(update/'weights.json'),
                deterministic_weights_sha256=sha(weights)))
        save(evaluation/'protocol.json',dict(version='combat_action_quality_common_validation_v1',
             evaluations=entries, families=families, total_episodes=64,episodes_per_model=32,
             slots=16,timescale=2,scope='Common validation geometry/seeds and legacy outcome reward; no final test or promotion.'))
        save(out/'progress.json',dict(stage='paired_validation',episodes=64,promotion=None))
        collect_compressed(repo,plans,evaluation)
        save(out/'progress.json',dict(stage='captures_complete',training_episodes=64,validation_episodes=64,
             evaluation_root=str(evaluation),scope='Quality report still required; no promotion.'))
    except Exception as error:
        save(out/'progress.json',dict(stage='failed',error=str(error),promotion=None))
        raise


if __name__ == '__main__':
    main()

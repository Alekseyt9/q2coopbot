"""Resume spatial PPO on fresh native trajectories with wall component clipping."""
import argparse
import os
import pathlib
import shutil
import sys

from process_combat_architecture_pool import read, save, sha, run
from run_combat_target_refresh import compile_plan, pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True, type=pathlib.Path)
    parser.add_argument('--capture', required=True, type=pathlib.Path)
    parser.add_argument('--processing', required=True, type=pathlib.Path)
    parser.add_argument('--seed-offset', type=int, default=112)
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    parent, capture, processing = args.parent.resolve(), args.capture.resolve(), args.processing.resolve()
    assert args.seed_offset >= 112 and not capture.exists() and not processing.exists()
    os.environ['GOCACHE'] = str(repo / 'workspace/build/go-cache')
    os.environ['GOTOOLCHAIN'] = 'auto'
    parent_report = read(parent / 'report.json')
    assert parent_report['state'] == 'complete'
    parent_capture = pathlib.Path(parent_report['protocol']['capture_root'])
    parent_bindings = {b['id']: b for b in read(parent_capture / 'models.json')}
    assert set(parent_bindings) == {'instant', 'postmove'}
    capture.mkdir()
    try:
        compiler = repo / 'workspace/build/q2episode-movement-v1.exe'
        exporter = repo / 'workspace/build/q2ppo-data-cuda-v1.exe'
        run(['go', 'build', '-buildvcs=false', '-o', compiler, './cmd/q2episode'], capture / 'build-compiler.log')
        run(['go', 'build', '-buildvcs=false', '-o', exporter, './cmd/q2ppo-data'], capture / 'build-exporter.log')
        shutil.copy2(parent / 'config.json', capture / 'config.json')
        bindings, plans = [], []
        for name in ('instant', 'postmove'):
            update = parent / name / 'update'
            seal, report = read(update / 'complete.json'), read(update / 'report.json')
            for filename, field in [('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'), ('report.json', 'report_sha256')]:
                assert sha(update / filename) == seal[field]
            assert report['device'] == 'cuda' and report['updates_completed'] >= 1
            assert not read(update / 'weights.json')['deterministic']
            audit = read(update / 'checkpoint-cuda-audit.json')
            assert audit['device'] == 'cuda' and audit['actor_value_std_exact'] and audit['optimizer_state_exact']
            assert audit['weights_sha256'] == sha(update / 'weights.json') and audit['checkpoint_sha256'] == sha(update / 'checkpoint.pt')
            frozen = capture / (name + '-weights.json')
            shutil.copy2(update / 'weights.json', frozen)
            old_plan = read(parent_bindings[name]['plan'])
            families = [t['episode']['id'] for t in old_plan['tasks']]
            assert len(families) == 20
            plan = compile_plan(compiler, old_plan['registry_path'], repo, frozen, capture / name,
                                'train', families, args.seed_offset)
            fresh = read(plan)
            old_seeds = {seed for t in old_plan['tasks'] for seed in t['seeds']}
            fresh_seeds = {seed for t in fresh['tasks'] for seed in t['seeds']}
            assert len(fresh_seeds) == 80 and not fresh_seeds & old_seeds
            bindings.append(dict(id=name, architecture=parent_bindings[name]['architecture'],
                model=str(frozen), plan=str(plan), capture_root=fresh['output_root'],
                parent_sha256=sha(frozen), parent_report_sha256=sha(update / 'report.json'),
                resume_checkpoint=str(update / 'checkpoint.pt'), resume_checkpoint_sha256=sha(update / 'checkpoint.pt'),
                resume_report=str(update / 'report.json'), resume_report_sha256=sha(update / 'report.json'),
                parent_updates_completed=report['updates_completed']))
            plans.append(plan)
        save(capture / 'models.json', bindings)
        save(capture / 'plans.json', [str(p) for p in plans])
        save(capture / 'preparation.json', dict(state='prepared', episodes=160, episodes_per_model=80,
             slots=16, timescale=2, training_device='cuda', numerical_verification='cuda_verified_v1',
             models_sha256=sha(capture / 'models.json'), config_sha256=sha(capture / 'config.json'),
             parent_report_sha256=sha(parent / 'report.json'), train_seed_offset=args.seed_offset,
             scope='Fresh stochastic own-policy trajectories with wall component clipping. Continue actor/value/std and Adam from sealed PPO checkpoints. No reward or training config changes; no old rollouts reused. Final test deferred.'))
        save(capture / 'execution.json', dict(stage='collecting_own_policy', episodes=160))
        pool(repo, repo / 'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe', plans,
             capture / 'pool', capture / 'collect.log')
        save(capture / 'execution.json', dict(stage='cuda_ppo_processing', episodes=160))
        legacy = repo / 'workspace/artifacts/aproc-v1-20261007'
        run([sys.executable, repo / 'scripts/process_combat_architecture_pool.py', '--capture-root', capture,
             '--out', processing, '--config', capture / 'config.json', '--exporter', exporter,
             '--anchor', legacy / 'anchor.json', '--bank', legacy / 'bank.json', '--export-workers', 4,
             '--cuda-only-export'], capture / 'process.log')
        result = read(processing / 'report.json')
        assert result['state'] == 'complete' and len(result['training']) == 2
        assert all(r['device'] == 'cuda' and r['updates_completed'] == bindings[i]['parent_updates_completed'] + 1
                   for i, r in enumerate(result['training']))
        run([sys.executable, repo / 'scripts/audit_combat_spatial_ppo_checkpoint_cuda.py', '--roots',
             processing / 'instant/update', processing / 'postmove/update'], capture / 'checkpoint-audit.log')
        save(capture / 'execution.json', dict(stage='complete', episodes=160,
             processing_report_sha256=sha(processing / 'report.json'),
             promotion='None; paired validation with the same movement guard is required'))
    except Exception as error:
        save(capture / 'execution.json', dict(stage='failed', error=str(error), promotion='None; artifacts preserved'))
        raise


if __name__ == '__main__':
    main()

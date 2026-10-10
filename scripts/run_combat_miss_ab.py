"""Matched fresh native capture and CUDA PPO for control/miss-reward forks."""
import argparse, os, pathlib, shutil, sys
from process_combat_architecture_pool import read, save, sha, run
from run_combat_spatial_ppo import wait_process
from run_combat_target_refresh import compile_plan, pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', type=pathlib.Path, required=True)
    ap.add_argument('--wait-pid', type=int)
    ap.add_argument('--preparation', type=pathlib.Path, required=True)
    ap.add_argument('--capture', type=pathlib.Path, required=True)
    ap.add_argument('--processing', type=pathlib.Path, required=True)
    ap.add_argument('--seed-offset', type=int, default=120)
    a = ap.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    capture, processing, preparation, smoke = [p.resolve() for p in (a.capture,a.processing,a.preparation,a.smoke)]
    assert not capture.exists() and not processing.exists() and a.seed_offset >= 120
    capture.mkdir()
    try:
        if a.wait_pid:
            wait_process(a.wait_pid,capture/'execution.json',stage='waiting_for_native_miss_smoke')
        receipt = read(smoke/'progress.json')
        assert receipt['stage'] == 'complete', 'Miss reward native smoke is not complete'
        assert receipt['quality_report_sha256'] == sha(smoke/'quality-report.json')
        proof = read(smoke/'recovery/verified-members.json')
        assert proof['state'] == 'complete' and len(proof['members']) == 16
        assert proof['protocol_sha256'] == sha(smoke/'protocol.json')
        os.environ['GOCACHE'] = str(repo/'workspace/build/go-cache')
        os.environ['GOTOOLCHAIN'] = 'auto'
        compiler, exporter = [repo/'workspace/build'/f'{name}-miss-ab-v1.exe' for name in ('q2episode','q2ppo-data')]
        run(['go','build','-buildvcs=false','-o',compiler,'./cmd/q2episode'],capture/'build-compiler.log')
        run(['go','build','-buildvcs=false','-o',exporter,'./cmd/q2ppo-data'],capture/'build-exporter.log')
        template = read(repo/'workspace/artifacts/movement-ppo-eval-v2-20261010/spatial-postmove-after/plan.json')
        families = [t['episode']['id'] for t in template['tasks']]
        assert len(families) == 20 and len(set(families)) == 20
        bindings, plans = [], []
        for name in ('control','miss'):
            parent = preparation/name
            fork = read(parent/'report.json')
            assert fork['device'] == 'cuda' and fork['actor_std_exact'] and fork['critic_output_zero'] and fork['adam_reset']
            assert sha(parent/'weights.json') == fork['weights_sha256'] and sha(parent/'checkpoint.pt') == fork['checkpoint_sha256']
            frozen = capture/(name+'-weights.json')
            shutil.copy2(parent/'weights.json',frozen)
            config = capture/(name+'-config.json')
            shutil.copy2(preparation/(name+'-config.json'),config)
            assert sha(config) == fork['config_sha256']
            registry = pathlib.Path(template['registry_path'])
            if name == 'miss':
                directory = capture/'miss-registry'
                directory.mkdir()
                files = []
                for task in template['tasks']:
                    episode = task['episode']
                    episode['recipe']['reward_config'] = 'scripts/scenarios/combat-reward-blaster-miss-v1.json'
                    if episode.get('generator'):
                        episode['generator'].setdefault('seed_revision',episode['revision'])
                    episode['revision'] += 1
                    filename = episode['id']+'.json'
                    files.append(filename)
                    save(directory/filename,episode)
                registry = directory/'index.json'
                save(registry,dict(version=1,files=files))
            plan = compile_plan(compiler,registry,repo,frozen,capture/name,'train',families,a.seed_offset)
            schedule = read(plan)
            seeds = {seed for task in schedule['tasks'] for seed in task['seeds']}
            assert len(seeds) == 80
            assert not seeds & {seed for task in template['tasks'] for seed in task['seeds']}
            binding = dict(id=name,architecture=dict(id='postmove-attention',architecture='attention'),
                           model=str(frozen),plan=str(plan),capture_root=schedule['output_root'],
                           parent_sha256=sha(frozen),parent_report_sha256=sha(parent/'report.json'),
                           resume_checkpoint=str(parent/'checkpoint.pt'),resume_checkpoint_sha256=sha(parent/'checkpoint.pt'),
                           resume_report=str(parent/'report.json'),resume_report_sha256=sha(parent/'report.json'),
                           parent_updates_completed=fork['updates_completed'])
            assert read(frozen).get('attention') and not read(frozen)['deterministic']
            bindings.append(binding)
            plans.append(plan)
        control, miss = [read(p) for p in plans]
        assert control['model_sha256'] == miss['model_sha256'], 'A/B actors/critics must start identical'
        for left,right in zip(control['tasks'],miss['tasks']):
            assert left['seeds'] == right['seeds'] and left['instances'] == right['instances']
            assert left['episode']['recipe']['loadout'] == right['episode']['recipe']['loadout']
        save(capture/'models.json',bindings)
        save(capture/'plans.json',[str(p) for p in plans])
        save(capture/'preparation.json',dict(state='prepared',episodes=160,episodes_per_model=80,
             slots=16,timescale=2,training_device='cuda',train_seed_offset=a.seed_offset,
             models_sha256=sha(capture/'models.json'),configs={name:sha(capture/(name+'-config.json')) for name in ('control','miss')},
             scope='Equal fresh own-policy budgets, identical actor/std and reset critic/Adam; reward only differs. Shared refill queue, separately pinned CUDA processing objectives. No consumed rollouts reused.'))
        save(capture/'execution.json',dict(stage='collecting_matched_own_policy',episodes=160))
        try:
            pool(repo,repo/'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe',plans,capture/'pool',capture/'collect.log')
        except RuntimeError:
            # The existing processor retries only invalid native members and
            # keeps their failed receipts. A complete gameplay loss is valid.
            terminal=read(capture/'pool/report.json')
            assert terminal['state']=='failed' and terminal['source_unchanged']
            assert len(terminal['jobs'])==160
        processing.mkdir()
        legacy = repo/'workspace/artifacts/aproc-v1-20261007'
        reports = []
        for name in ('control','miss'):
            destination = processing/name
            save(capture/'execution.json',dict(stage='cuda_processing',binding=name,episodes=160))
            run([sys.executable,repo/'scripts/process_combat_architecture_pool.py','--capture-root',capture,
                 '--out',destination,'--model-id',name,'--config',capture/(name+'-config.json'),
                 '--exporter',exporter,'--anchor',legacy/'anchor.json','--bank',legacy/'bank.json',
                 '--export-workers',4,'--cuda-only-export'],capture/(name+'-process.log'))
            result = read(destination/'report.json')
            assert result['state'] == 'complete' and len(result['training']) == 1
            assert result['training'][0]['device'] == 'cuda'
            run([sys.executable,repo/'scripts/audit_combat_spatial_ppo_checkpoint_cuda.py','--roots',
                 destination/name/'update'],capture/(name+'-checkpoint-audit.log'))
            reports.append(dict(binding=name,report=str(destination/'report.json'),report_sha256=sha(destination/'report.json')))
        save(processing/'report.json',dict(state='complete',training_device='cuda',branches=reports,
             preparation_sha256=sha(capture/'preparation.json'),scope='Both matched objective updates complete; native quality comparison still required.'))
        save(capture/'execution.json',dict(stage='complete',episodes=160,processing_report_sha256=sha(processing/'report.json')))
    except Exception as error:
        save(capture/'execution.json',dict(stage='failed',error=str(error),promotion='None; evidence retained'))
        raise


if __name__ == '__main__':
    main()

"""Fresh first-life native collection, batched CUDA finalization and resumed PPO."""
import argparse, os, pathlib, shutil, sys
from process_combat_architecture_pool import read, save, sha, run
from run_combat_target_refresh import compile_plan, pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', type=pathlib.Path, required=True)
    ap.add_argument('--parent', type=pathlib.Path, required=True)
    ap.add_argument('--template-plan', type=pathlib.Path, required=True)
    ap.add_argument('--contract', type=pathlib.Path, required=True)
    ap.add_argument('--capture', type=pathlib.Path, required=True)
    ap.add_argument('--processing', type=pathlib.Path, required=True)
    ap.add_argument('--seed-offset', type=int, default=128)
    a = ap.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    smoke, parent, template_path, contract, capture, processing = [p.resolve() for p in
        (a.smoke, a.parent, a.template_plan, a.contract, a.capture, a.processing)]
    assert not capture.exists() and not processing.exists() and a.seed_offset >= 128
    acceptance = read(smoke/'death-stop-acceptance.json')
    proof = read(smoke/'recovery/verified-members.json')
    assert read(smoke/'progress.json')['stage'] == 'complete'
    assert acceptance['state'] == 'passed' and acceptance['deaths'] and acceptance['episodes'] == 16
    assert acceptance['quality_report_sha256'] == sha(smoke/'quality-report.json')
    assert proof['state'] == 'complete' and len(proof['members']) == 16
    assert proof['protocol_sha256'] == sha(smoke/'protocol.json')
    seal, parent_report = read(parent/'complete.json'), read(parent/'report.json')
    for name, key in [('weights.json','weights_sha256'), ('checkpoint.pt','checkpoint_sha256'), ('report.json','report_sha256')]:
        assert sha(parent/name) == seal[key]
    assert parent_report['device'] == 'cuda' and not read(parent/'weights.json')['deterministic']
    for name, key in [('config.json','config_sha256'), ('anchor.json','anchor_sha256'), ('bank.json','bank_sha256')]:
        assert sha(contract/name) == parent_report[key], 'Continuation objective changed'
    template = read(template_path)
    families = [t['episode']['id'] for t in template['tasks']]
    assert len(families) == len(set(families)) == 20
    capture.mkdir()
    try:
        os.environ['GOCACHE'] = str(repo/'workspace/build/go-cache')
        os.environ['GOTOOLCHAIN'] = 'auto'
        compiler = capture/'q2episode.exe'
        exporter = capture/'q2ppo-data.exe'
        run(['go','build','-buildvcs=false','-o',compiler,'./cmd/q2episode'],capture/'build-compiler.log')
        run(['go','build','-buildvcs=false','-o',exporter,'./cmd/q2ppo-data'],capture/'build-exporter.log')
        frozen = capture/'control-weights.json'
        shutil.copy2(parent/'weights.json',frozen)
        plan = compile_plan(compiler,template['registry_path'],repo,frozen,capture/'control','train',families,a.seed_offset)
        schedule = read(plan)
        seeds = {s for t in schedule['tasks'] for s in t['seeds']}
        assert len(seeds) == 80
        assert not seeds & {s for t in template['tasks'] for s in t['seeds']}
        assert not seeds & {s for t in read(smoke/'control-death-stop-baseline/plan.json')['tasks'] for s in t['seeds']}
        binding = dict(id='control',architecture=dict(id='postmove-attention',architecture='attention'),
            model=str(frozen),plan=str(plan),capture_root=schedule['output_root'],
            parent_sha256=sha(frozen),parent_report_sha256=sha(parent/'report.json'),
            resume_checkpoint=str(parent/'checkpoint.pt'),resume_checkpoint_sha256=sha(parent/'checkpoint.pt'),
            resume_report=str(parent/'report.json'),resume_report_sha256=sha(parent/'report.json'),
            parent_updates_completed=parent_report['updates_completed'])
        save(capture/'models.json',[binding])
        save(capture/'preparation.json',dict(state='prepared',episodes=80,slots=16,timescale=2,
            training_device='cuda',train_seed_offset=a.seed_offset,parent_updates_completed=parent_report['updates_completed'],
            death_stop_acceptance_sha256=sha(smoke/'death-stop-acceptance.json'),
            models_sha256=sha(capture/'models.json'),cuda_finalization='independent_cases_single_interpreter_v1',
            scope='Fresh own-policy seeds, unchanged control reward, first-life stop with ordered native death proof. No quality promotion.'))
        save(capture/'execution.json',dict(stage='collecting_own_policy',episodes=80))
        try:
            pool(repo,repo/'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe',[plan],capture/'pool',capture/'collect.log')
        except RuntimeError:
            terminal = read(capture/'pool/report.json')
            assert terminal['state'] == 'failed' and terminal['source_unchanged'] and len(terminal['jobs']) == 80
        save(capture/'execution.json',dict(stage='cuda_processing',episodes=80))
        run([sys.executable,repo/'scripts/process_combat_architecture_pool.py','--capture-root',capture,
            '--out',processing,'--config',contract/'config.json','--exporter',exporter,
            '--anchor',contract/'anchor.json','--bank',contract/'bank.json','--export-workers',4,
            '--cuda-only-export','--cuda-batch-finalize'],capture/'process.log')
        result = read(processing/'report.json')
        assert result['state'] == 'complete' and len(result['training']) == 1
        assert result['training'][0]['updates_completed'] == parent_report['updates_completed']+1
        assert result['protocol']['cuda_finalization'] == 'independent_cases_single_interpreter_v1'
        run([sys.executable,repo/'scripts/audit_combat_spatial_ppo_checkpoint_cuda.py','--roots',processing/'control/update'],capture/'checkpoint-audit.log')
        save(capture/'execution.json',dict(stage='complete',episodes=80,
            processing_report_sha256=sha(processing/'report.json'),scope='CUDA update complete; paired native quality assessment still required.'))
    except Exception as error:
        save(capture/'execution.json',dict(stage='failed',error=str(error),promotion='None; evidence retained'))
        raise


if __name__ == '__main__':
    main()

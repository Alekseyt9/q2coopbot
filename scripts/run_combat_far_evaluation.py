"""Queue paired long-range native evaluation after the current pool closes."""
import argparse
import os
import pathlib
import shutil
import sys

from process_combat_architecture_pool import read, run, save, sha
from run_combat_spatial_ppo import wait_process
from run_combat_target_refresh import compile_plan, pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=pathlib.Path, required=True)
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--draft', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    previous, draft, out = args.previous.resolve(), args.draft.resolve(), args.out.resolve()
    assert not out.exists()
    out.mkdir()
    try:
        preparation = read(draft / 'report.json')
        assert preparation['state'] == 'static_preflight_passed_native_pending'
        registry = pathlib.Path(preparation['registry']).resolve()
        assert sha(registry) == preparation['registry_sha256']
        families = []
        for record in preparation['recipes']:
            path = registry.parent / (record['id'] + '.json')
            assert sha(path) == record['recipe_sha256']
            families.append(record['id'])
        assert len(families) == len(set(families)) == 4
        parent_protocol = read(previous / 'protocol.json')
        assert len(parent_protocol['evaluations']) == 6
        compiler = repo / 'workspace/build/q2episode-precision-v1.exe'
        os.environ['GOCACHE'] = str(repo / 'workspace/build/go-cache')
        os.environ['GOTOOLCHAIN'] = 'auto'
        entries, plans = [], []
        for entry in parent_protocol['evaluations']:
            name, label = entry['model'], entry['label']
            weight = None
            if entry['deterministic_weights_sha256']:
                source_plan = read(entry['plan'])
                source = pathlib.Path(source_plan['model_path'])
                assert sha(source) == entry['deterministic_weights_sha256']
                weight = out / (name + '-' + label + '-weights.json')
                shutil.copy2(source, weight)
                assert sha(weight) == entry['deterministic_weights_sha256']
            branch = out / (name + '-' + label)
            plan = compile_plan(compiler, registry, repo, weight, branch, 'validation', families, 0)
            plans.append(plan)
            entries.append(dict(model=name, label=label, root=str(branch / 'capture'),
                plan=str(plan), plan_sha256=sha(plan), source_weights_sha256=entry['source_weights_sha256'],
                deterministic_weights_sha256=sha(weight) if weight else None))
        save(out / 'protocol.json', dict(version='combat_far_spatial_validation_v1', evaluations=entries,
            families=families, total_episodes=96, episodes_per_model=16, slots=16, timescale=2,
            validation_seed_offset=0, comparison_reference='firebc-baseline',
            comparison_stage='far_geometry_transfer_after_spatial_ppo',
            previous_protocol_sha256=sha(previous / 'protocol.json'),
            draft_report_sha256=sha(draft / 'report.json'), registry_sha256=sha(registry),
            driver_sha256=sha(__file__), compiler_sha256=sha(compiler),
            scope='First native acceptance of four draft far recipes: two campaign sites with Blaster/Machinegun, four validation seeds each. Same six frozen variants as prior spatial PPO comparison, each seed a separate native run. No training or promotion; validation development data, final test deferred. Parent wall rays are parent-location metadata only.'))
        wait_process(args.wait_pid, out / 'progress.json', stage='waiting_for_previous_evaluation')
        terminal = read(previous / 'progress.json')
        assert terminal['stage'] == 'complete', 'Previous evaluation is not sealed complete'
        assert terminal['quality_report_sha256'] == sha(previous / 'quality-report.json')
        prior_proof = read(previous / 'recovery/verified-members.json')
        assert prior_proof['state'] == 'complete' and prior_proof['protocol_sha256'] == sha(previous / 'protocol.json')
        # Report completed prior trajectories by distance before starting the next pool.
        run([sys.executable, repo / 'scripts/report_combat_aim_distance.py', '--root', previous], out / 'previous-distance.log')
        save(out / 'progress.json', dict(stage='paired_native_evaluation', episodes=96))
        pool(repo, repo / 'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe', plans,
             out / 'pool', out / 'evaluation.log')
        manifests = sorted(pathlib.Path(entries[0]['root']).glob('case-*/s-*/manifest.json'))
        assert len(manifests) == 16
        manifest = read(manifests[0])
        run([sys.executable, repo / 'scripts/verify_combat_evaluation_members.py', '--root', out,
             '--source-fingerprint', manifest['source_fingerprint'],
             '--native-fingerprint', manifest['native_source_fingerprint']], out / 'verify.log')
        run([sys.executable, repo / 'scripts/report_combat_architecture_evaluation.py', '--root', out,
             '--member-proof', out / 'recovery/verified-members.json'], out / 'report.log')
        for tool in ('evaluation_strata', 'selected_target_aim', 'blaster_hits', 'aim_modes', 'aim_distance'):
            run([sys.executable, repo / ('scripts/report_combat_' + tool + '.py'), '--root', out], out / (tool + '.log'))
        save(out / 'progress.json', dict(stage='complete', episodes=96,
            quality_report_sha256=sha(out / 'quality-report.json'),
            promotion='None; draft acceptance and quality require review'))
    except Exception as error:
        save(out / 'progress.json', dict(stage='failed', error=str(error),
            promotion='None; partial artifacts preserved'))
        raise


if __name__ == '__main__':
    main()

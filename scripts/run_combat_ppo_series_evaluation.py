"""Common development evaluation of initial and sequentially continued CUDA actors."""
import argparse
import pathlib
import shutil

from combat_native_evaluation_closure import finalize
from process_combat_architecture_pool import read, save, sha, run
from run_combat_ppo_series import sealed_update
from run_combat_sampling_evaluation import collect_compressed
from run_combat_spatial_ppo import wait_process
from run_combat_stochastic_heldout import verify_sources
from run_combat_target_refresh import compile_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--series', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--validation-offset', type=int)
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    series, out = args.series.resolve(), args.out.resolve()
    assert not out.exists()
    protocol_sha = sha(series / 'protocol.json')
    spec = read(series / 'protocol.json')
    validation_offset = spec['planned_validation_offset'] if args.validation_offset is None else args.validation_offset
    assert validation_offset >= 0 and validation_offset % 4 == 0
    out.mkdir()
    save(out / 'queue.json', dict(series=str(series), series_protocol_sha256=protocol_sha,
        observed_process_pid=args.wait_pid, episodes=400, validation_offset=validation_offset,
        training_planned_validation_offset=spec['planned_validation_offset'],
        slots=16, timescale=2, reserve_gib=3,
        scope='Parent and final actor x two policy RNG arms plus rules on80 common validation conditions. No independent test/promotion.'))
    try:
        if args.wait_pid:
            wait_process(args.wait_pid, out / 'progress.json', 'waiting_for_cuda_series')
        assert sha(series / 'protocol.json') == protocol_sha
        execution = read(series / 'progress.json')
        assert execution['stage'] == 'complete' and execution['parent_optimizer_preserved']
        assert execution['completed_rounds'] == spec['rounds']
        initial, latest = pathlib.Path(spec['parent_update']), pathlib.Path(execution['final_update'])
        before, after = sealed_update(initial), sealed_update(latest)
        assert after['updates_completed'] == before['updates_completed'] + spec['rounds']
        assert sha(latest / 'checkpoint.pt') == execution['final_checkpoint_sha256']
        assert sha(latest / 'weights.json') == execution['final_weights_sha256']
        assert sha(initial / 'checkpoint.pt') == spec['parent_checkpoint_sha256']
        assert sha(initial / 'weights.json') == spec['parent_weights_sha256']
        reference = pathlib.Path(spec['reference'])
        assert sha(reference / 'protocol.json') == spec['reference_protocol_sha256']
        source_binding = verify_sources(repo, reference)
        # Recent560-member native capture peak fit6GiB;400 cases, shared immutable
        # binaries and terminal-member LZX use a3GiB dispatch reserve. Check it now.
        assert shutil.disk_usage(out).free > 3 * 1024**3, 'Reserve3GiB for400 native comparison; compact closed archives before dispatch'
        capture = series / 'round-1/capture'
        template = read(read(capture / 'models.json')[0]['plan'])
        plans, entries, conditions = [], [], None
        for name, update in [('parent', initial), ('after', latest), ('rules', None)]:
            weights = None
            if update:
                weights = out / (name + '-weights.json')
                shutil.copyfile(update / 'weights.json', weights)
                assert not read(weights)['deterministic']
            modes = [('stochastic-a', 20261011), ('stochastic-b', 20261012)] if weights else [('baseline', 0)]
            for label, offset in modes:
                path = compile_plan(repo / 'workspace/build/q2episode-sampling-v1.exe', template['registry_path'],
                    repo, weights, out / (name + '-' + label), 'validation', spec['families'], validation_offset)
                plan = read(path)
                plan['policy_sampling_seed_offset'] = offset
                save(path, plan)
                instance_set = {t['episode']['id']: (t['seeds'], t['instances']) for t in plan['tasks']}
                if conditions is None:
                    conditions = instance_set
                assert instance_set == conditions
                run([repo / 'workspace/build/q2episode-sampling-v1.exe', '--verify-plan', path, '--root', repo], path.parent / 'verify.log')
                plans.append(path)
                entries.append(dict(model=name, label=label, root=plan['output_root'], plan=str(path), plan_sha256=sha(path),
                    deterministic_weights_sha256=sha(weights) if weights else None, source_weights_sha256=sha(weights) if weights else None,
                    deterministic=False if weights else True, policy_sampling_seed_offset=offset))
        save(out / 'protocol.json', dict(version='combat_cuda_series_common_development_v1', evaluations=entries,
            families=spec['families'], episodes_per_model=80, total_episodes=400, slots=16, timescale=2,
            comparison_reference='rules-baseline', series_protocol_sha256=protocol_sha,
            validation_offset=validation_offset, training_planned_validation_offset=spec['planned_validation_offset'],
            validation_offset_override_reason='Explicit cohort-bound correction: original planned offset exceeds registered validation count.' if validation_offset != spec['planned_validation_offset'] else None,
            scope=f'Two CUDA updates versus unchanged parent, exact common validation{validation_offset}..{validation_offset + 3} conditions and two declared RNG arms plus rules. '
                  'Development, no globally unseen-seed claim. Rules score cannot be compared to68/80 on earlier validation24..27 as the conditions differ. '
                  'No independent test, training or automatic promotion.'))
        assert verify_sources(repo, reference) == source_binding
        save(out / 'progress.json', dict(stage='evaluating', episodes=400, promotion=None))
        result = collect_compressed(repo, plans, out)
        finalize(repo, out, result)
    except Exception as error:
        save(out / 'progress.json', dict(stage='failed', error=str(error), promotion=None))
        raise


if __name__ == '__main__':
    main()

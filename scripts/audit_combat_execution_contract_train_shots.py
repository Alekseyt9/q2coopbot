"""Describe native MG shots in the accepted TRAIN rows of a closed contract experiment."""
import argparse
import collections
import json
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha
from combat_execution_contract import LEGACY, CHECKED_GROUND, require_same
from report_combat_machinegun_hits import scan
from report_combat_machinegun_aim import measure
from report_registered_machinegun_outcomes import outcome


def summarize(items):
    counts = collections.Counter(outcome(shot) for shot, _ in items)
    measured = [value for _, value in items if value['status'] == 'measured']
    return dict(shots=len(items), outcomes=dict(counts), measured_shots=len(measured),
                mean_pre_recoil_degrees=sum(v['pre_recoil_degrees'] for v in measured)/len(measured) if measured else None,
                mean_post_recoil_degrees=sum(v['post_recoil_degrees'] for v in measured)/len(measured) if measured else None,
                post_recoil_above_10_degrees=sum(v['post_recoil_degrees'] > 10 for v in measured),
                recoil_worsened_above_1_degree=sum(v['post_recoil_degrees'] > v['pre_recoil_degrees']+1 for v in measured),
                muzzle_blocked=sum(int(shot['fields']['muzzle_blocked']) for shot, _ in items))


def buckets(items):
    groups = collections.defaultdict(list)
    for shot, value in items:
        item = (shot, value)
        groups['all'].append(item)
        groups['status:'+value['status']].append(item)
        if value['status'] == 'measured':
            groups['distance:'+('below128' if value['distance'] < 128 else '128to256' if value['distance'] < 256 else '256plus')].append(item)
            groups['speed:'+('below100' if value['native_horizontal_speed'] < 100 else '100plus')].append(item)
            groups['recoil:'+('above6' if abs(value['native_recoil_pitch']) > 6 else 'atmost6')].append(item)
    return {name: summarize(group) for name, group in sorted(groups.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), 'Fresh output required'
    root = args.workflow.resolve()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = pins.get(str(path)) or sha(path)
        assert expected is None or expected == digest, str(path)
        pins[str(path)] = digest
        return path

    for name in ('audit_combat_execution_contract_train_shots.py', 'report_combat_machinegun_hits.py',
                 'report_combat_machinegun_aim.py', 'report_registered_machinegun_outcomes.py',
                 'report_combat_selected_target_aim.py', 'native_projectile_window.py'):
        pin(Path(__file__).with_name(name))
    closed = read(pin(root/'complete.json'))
    assert closed['state'] == 'complete' and closed['source_unchanged']
    assert closed['native_train_cases'] == closed['cuda_train_cases'] == 192
    cases, rounds, variants, seeds = [], [], {}, set()
    total_rows = 0
    for arm, contract in (('legacy', LEGACY), ('checked', CHECKED_GROUND)):
        arm_shots = []
        for number in range(1, 5):
            folder = root/f'round-{number:02d}'/arm
            seal = read(pin(folder/'update/complete.json'))
            update = read(pin(folder/'update/report.json', seal['report_sha256']))
            assert update['device'] == 'cuda' and update['scope_updates'] == number
            require_same(contract, update['combat_execution_contract'])
            corpus = folder/'corpus'
            receipt = read(pin(corpus/'complete.json'))
            protocol = read(pin(corpus/'protocol.json', receipt['protocol_sha256']))
            assert protocol['train_only'] and all(a['split'] == 'train' for a in protocol['assignments'])
            require_same(contract, protocol['combat_execution_contract'])
            pool = read(pin(receipt['source_pool_report'], receipt['pool_sha256']))
            assert pool['state'] == 'complete' and pool['source_unchanged']
            processing = corpus/'processing-train'
            processed = read(pin(processing/'report.json', receipt['processed'][0]['sha256']))
            assert processed['state'] == 'complete' and len(processed['cases']) == 24
            jobs = {job['job']: job for job in pool['jobs']}
            round_shots, round_rows = [], 0
            for case in processed['cases']:
                job = jobs[case['job']]
                assert case['split'] == 'train' and not job['error'] and job['plan_index'] == receipt['selected_plan_index']
                assert case['seed'] == job['seed'] and case['seed'] not in seeds
                seeds.add(case['seed'])
                case_root = processing/f"case-{case['job']:04d}"
                meta_path = case_root/'cuda/report.json'
                meta = read(pin(meta_path, update['inputs'][str(meta_path.resolve())]))
                require_same(contract, meta['combat_execution_contract'])
                assert meta['numerical_verification'] == 'cuda_verified_v1' and meta['model_sha256'] == update['behavior_sha256']
                rows_path = pin(case_root/'cuda/rollout.jsonl', meta['rollout_sha256'])
                with rows_path.open(encoding='utf-8-sig') as stream:
                    accepted = {(row['index'], row['frame']) for row in map(json.loads, stream)}
                assert len(accepted) == meta['rows']
                steps_path = case_root/'native/replay-0/steps.jsonl'
                pin(steps_path, meta['source_sha256'][str(steps_path.resolve())])
                log_path = next(Path(path) for path in meta['source_sha256'] if path.endswith('server.log'))
                pin(log_path, meta['source_sha256'][str(log_path)])
                windows = {}
                with steps_path.open(encoding='utf-8-sig') as stream:
                    for step in map(json.loads, stream):
                        observation = step['observation']
                        if (step['index'], observation['identity']['frame']) not in accepted:
                            continue
                        execution, native = step['server_execution'], step['native_step']
                        assert step['owner'] == 'provider' and observation['identity']['life'] == 1 and observation['health'] > 0
                        assert execution['matched'] and execution['window_exclusive'] and execution['recovery_commands'] == 0
                        key = tuple(native[k] for k in ('spawncount', 'begin_frame', 'actor', 'sequence'))
                        assert key not in windows
                        windows[key] = step
                assert len(windows) == len(accepted)
                with log_path.open(encoding='utf-8-sig') as stream:
                    shots = [(shot, measure(shot, windows[shot['window']])) for shot in scan(stream, set(windows)) if shot['eligible']]
                cases.append(dict(arm=arm, round=number, seed=case['seed'], rows=len(accepted), groups=buckets(shots)))
                round_shots.extend(shots)
                round_rows += len(accepted)
            assert round_rows == update['rows']
            total_rows += round_rows
            rounds.append(dict(arm=arm, round=number, rows=round_rows, groups=buckets(round_shots)))
            arm_shots.extend(round_shots)
            print(dict(arm=arm, round=number, rows=round_rows, shots=len(round_shots)), flush=True)
        variants[arm] = buckets(arm_shots)
    assert len(cases) == len(seeds) == 192 and total_rows == 21201
    assert all(sha(path) == digest for path, digest in pins.items())
    args.out.mkdir(parents=True)
    save(args.out/'report.json', dict(state='complete', version='combat_execution_contract_train_shots_v1',
        train_cases=192, rows=total_rows, variants=variants, rounds=rounds, cases=cases, inputs=pins,
        neural_execution=False, new_games=False, training_performed=False, promotion=False,
        scope='Accepted own-policy TRAIN rows only; no validation trajectories. Native MG outcomes and known selected-target snapshot geometry. '
              'Statuses partition shots; distance/speed/recoil groups overlap and cover measured targets only. '
              'Angular errors are descriptive, not avoidability or hit probability; motion, spread and snapshot age remain. '
              'Native labels never become policy inputs; this audit neither fits a model nor changes reward.'))
    save(args.out/'complete.json', dict(state='complete', report_sha256=sha(args.out/'report.json')))


if __name__ == '__main__':
    main()

"""Observed-eye versus actual muzzle geometry; no hidden target or NN replay."""
import argparse
import collections
import json
import math
import pathlib

from process_combat_architecture_pool import read, save, sha
from report_combat_machinegun_aim import angular, measure
from report_combat_machinegun_hits import scan, vector
from run_combat_spatial_ppo import wait_process


def parallax(shot, step):
    baseline = measure(shot, step)
    if baseline['status'] != 'measured':
        return baseline
    observation, action = step['observation'], step['action']
    target = next(e for e in observation['enemies'] if e['id'] == action['target_entity'] and
                  e.get('observed_track', 0) == action.get('target_track', 0))
    solid = target['observed_solid']
    bottom, top = -((solid >> 5) & 31) * 8, ((solid >> 10) & 63) * 8 - 32
    height = (top + bottom) / 2 if top - bottom < 16 else min(max(22, bottom + 8), top - 8)
    point = [observation['position'][i] + target['relative'][i] for i in range(3)]
    point[2] += height
    eye = list(observation['position'])
    eye[2] += -2 if observation['ducked'] else 22
    muzzle = vector(shot['fields']['start'])
    eye_ray = [point[i] - eye[i] for i in range(3)]
    muzzle_ray = [point[i] - muzzle[i] for i in range(3)]
    if sum(v * v for v in eye_ray) < 1e-12:
        return dict(status='degenerate_eye_target')
    actual_aim = vector(shot['fields']['aim'])
    eye_error, muzzle_error = angular(actual_aim, eye_ray), angular(actual_aim, muzzle_ray)
    assert abs(muzzle_error - baseline['post_recoil_degrees']) < 1e-9
    return dict(status='measured', distance=baseline['distance'],
        snapshot_eye_to_native_muzzle=math.dist(eye, muzzle), parallax_degrees=angular(eye_ray, muzzle_ray),
        native_aim_vs_snapshot_eye_degrees=eye_error, native_aim_vs_muzzle_degrees=muzzle_error,
        muzzle_minus_eye_error_degrees=muzzle_error - eye_error,
        native_speed=shot['metrics']['speed'], selected_target_damage=baseline['selected_target_damage'])


def report(root):
    protocol, proof = read(root / 'protocol.json'), read(root / 'recovery/verified-members.json')
    assert proof['state'] == 'complete' and proof['protocol_sha256'] == sha(root / 'protocol.json')
    groups, sources = {}, []
    for entry in protocol['evaluations']:
        counts, rows = collections.Counter(), []
        receipts = sorted(pathlib.Path(entry['root']).glob('case-*/s-*/report.json'))
        assert len(receipts) == protocol['episodes_per_model']
        for receipt in receipts:
            assert sha(receipt) == proof['members'][str(receipt.parent)]['report_sha256']
            capture = read(receipt)
            assert capture['capture_complete'] and capture['provenance_valid'] and len(capture['results']) == 1
            result = capture['results'][0]
            assert result['dataset']['command_proof']['accepted']
            folder = pathlib.Path(result['root'])
            steps, server = folder / 'dataset/steps.jsonl', folder / 'server.log'
            steps_hash, server_hash = sha(steps), sha(server)
            by_window = {}
            with steps.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    step = json.loads(line)
                    obs, native, execution = step['observation'], step.get('native_step'), step.get('server_execution', {})
                    if (native and obs['identity']['life'] == 1 and obs['health'] > 0 and obs['identity']['frame'] > 100
                        and execution.get('matched') and execution.get('window_exclusive')):
                        key = (native['spawncount'], native['begin_frame'], native['actor'], native['sequence'])
                        assert key not in by_window
                        by_window[key] = step
            with server.open(encoding='utf-8-sig') as stream:
                shots = scan(stream, set(by_window))
            for shot in shots:
                if not shot['eligible']:
                    continue
                item = parallax(shot, by_window[shot['window']])
                counts[item['status']] += 1
                if item['status'] == 'measured':
                    rows.append(item)
            assert sha(steps) == steps_hash and sha(server) == server_hash
            sources.append(dict(report=str(receipt), report_sha256=sha(receipt), steps_sha256=steps_hash, server_sha256=server_hash))
        def summary(values):
            keys = ('snapshot_eye_to_native_muzzle', 'parallax_degrees', 'native_aim_vs_snapshot_eye_degrees',
                    'native_aim_vs_muzzle_degrees', 'muzzle_minus_eye_error_degrees')
            return dict(shots=len(values), **{key: sum(v[key] for v in values) / len(values) if values else None for key in keys},
                        parallax_above2=sum(v['parallax_degrees'] > 2 for v in values),
                        selected_target_damage=sum(v['selected_target_damage'] for v in values))
        strata = {}
        for distance in ('below128', '128to256', '256plus'):
            for motion in ('moving', 'stationary'):
                values = [v for v in rows if ('below128' if v['distance'] < 128 else '128to256' if v['distance'] < 256 else '256plus') == distance
                          and ('moving' if v['native_speed'] >= 10 else 'stationary') == motion]
                strata[distance + ' / ' + motion] = summary(values)
        groups[entry['model'] + '-' + entry['label']] = dict(status_counts=dict(counts), all=summary(rows), strata=strata)
    save(root / 'muzzle-parallax.json', dict(state='complete', protocol_sha256=sha(root / 'protocol.json'),
        member_proof_sha256=sha(root / 'recovery/verified-members.json'), groups=groups, sources=sources,
        scope='Exclusive native machinegun windows, first life after frame100, explicit provider-selected observed bbox. '
              'Compare snapshot-eye and native-muzzle rays to the SAME observed target point. No hidden/server target coordinates. '
              'Snapshot-to-muzzle offset combines own movement/timing, stance and weapon offset; it does not isolate movement causality. '
              'No moving-target extrapolation, spread correction or counterfactual action replay. Descriptive geometric difference, not hit probability. '
              'Legacy controls without declared target remain unmeasured. No NN inference or training.'))
    print({name: group['all'] for name, group in groups.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.wait_pid:
        wait_process(args.wait_pid, root / 'muzzle-parallax-progress.json', 'waiting_for_native_evaluation')
    report(root)
    save(root / 'muzzle-parallax-progress.json', dict(state='complete', report_sha256=sha(root / 'muzzle-parallax.json')))

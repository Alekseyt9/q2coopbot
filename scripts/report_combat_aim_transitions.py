"""Observed aim-mode/target transitions on exclusive native provider commands."""
import argparse
import collections
import json
import math
import pathlib

from process_combat_architecture_pool import read, save, sha
from report_combat_selected_target_aim import measure
from run_combat_spatial_ppo import wait_process


def report(root):
    protocol = read(root / 'protocol.json')
    proof = read(root / 'recovery/verified-members.json')
    assert proof['state'] == 'complete' and proof['protocol_sha256'] == sha(root / 'protocol.json')
    groups, sources = {}, []
    for entry in protocol['evaluations']:
        plan = read(entry['plan'])
        assert sha(entry['plan']) == entry['plan_sha256']
        model = read(plan['model_path']) if plan.get('model_path') else {}
        name = entry['model'] + '-' + entry['label']
        if not model.get('aim_mode_head'):
            groups[name] = dict(status='mode_not_declared')
            continue
        counters, buckets = collections.Counter(), {}
        receipts = sorted(pathlib.Path(entry['root']).glob('case-*/s-*/report.json'))
        assert len(receipts) == protocol['episodes_per_model']
        for receipt in receipts:
            assert sha(receipt) == proof['members'][str(receipt.parent)]['report_sha256']
            capture = read(receipt)
            assert capture['capture_complete'] and capture['provenance_valid']
            assert len(capture['results']) == 1
            result = capture['results'][0]
            assert result['dataset']['command_proof']['accepted']
            path = pathlib.Path(result['root']) / 'dataset/steps.jsonl'
            digest = sha(path)
            previous = None
            with path.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    row = json.loads(line)
                    obs, action, applied = row['observation'], row['action'], row['applied_action']
                    identity = obs['identity']
                    execution = row.get('server_execution', {})
                    if (row['owner'] != 'provider' or identity['life'] != 1 or identity['frame'] <= 100
                        or obs['health'] <= 0 or not row.get('native_step') or not execution.get('matched')
                        or not execution.get('window_exclusive')):
                        previous = None
                        continue
                    counters['exclusive_native_provider_commands'] += 1
                    mode = action.get('aim_mode', 0)
                    assert mode in (0, 1)
                    target = (action.get('target_entity', 0), action.get('target_track', 0))
                    chain = tuple(identity[k] for k in ('connection', 'map', 'spawncount', 'actor', 'life'))
                    current = dict(frame=identity['frame'], chain=chain, target=target, mode=mode, weapon=obs['weapon'])
                    adjacent = (previous and current['chain'] == previous['chain'] and
                                current['frame'] == previous['frame'] + 1 and current['weapon'] == previous['weapon'])
                    transition = 'no_adjacent_context'
                    if adjacent:
                        counters['adjacent_native_commands'] += 1
                        target_changed = target != previous['target']
                        mode_changed = mode != previous['mode']
                        counters['mode_switches'] += mode_changed
                        counters['target_switches'] += target_changed
                        transition = 'target_changed' if target_changed else 'mode_changed_same_target' if mode_changed else 'stable_mode_same_target'
                    previous = current
                    item = measure(dict(observation=obs, selection=dict(owner='provider', candidate=action), applied=applied))
                    assert item is not None
                    counters[item['status']] += 1
                    if item['status'] != 'measured':
                        continue
                    enemy = next(e for e in obs['enemies'] if e['id'] == target[0] and (e.get('observed_track') or 0) == target[1])
                    distance = 'near' if enemy['distance'] <= 128 else 'medium' if enemy['distance'] <= 512 else 'far'
                    motion = 'stationary' if math.hypot(*obs['velocity'][:2]) < 5 else 'moving'
                    command_angle = math.hypot(applied['yaw_delta_degrees'], applied['pitch_delta_degrees'])
                    for kind, value in [('all', 'all'), ('transition', transition), ('range', distance),
                                        ('mode', 'fine' if mode else 'coarse'), ('motion', motion),
                                        ('weapon_range_transition', obs['weapon'] + ' / ' + distance + ' / ' + transition)]:
                        bucket = buckets.setdefault(kind + ': ' + value, collections.Counter())
                        bucket['frames'] += 1
                        bucket['angular_sum'] += item['angular']
                        bucket['above10'] += item['angular'] > 10
                        bucket['command_angle_sum'] += command_angle
                        bucket['attack_frames'] += bool(applied['attack'])
                        if applied['attack']:
                            bucket['firing_angular_sum'] += item['angular']
                            bucket['firing_above10'] += item['angular'] > 10
            assert sha(path) == digest, 'Sealed dataset changed during report'
            sources.append(dict(variant=name, report=str(receipt), report_sha256=sha(receipt),
                steps=str(path), steps_sha256=digest))
        summaries = {key: dict(frames=b['frames'], mean_angular_degrees=b['angular_sum'] / b['frames'],
            above10_fraction=b['above10'] / b['frames'], mean_command_angle_degrees=b['command_angle_sum'] / b['frames'],
            firing_frames=b['attack_frames'], firing_mean_angular_degrees=b['firing_angular_sum'] / b['attack_frames'] if b['attack_frames'] else None,
            firing_above10_fraction=b['firing_above10'] / b['attack_frames'] if b['attack_frames'] else None)
            for key, b in buckets.items()}
        groups[name] = dict(status='measured', episodes=len(receipts), counts=dict(counters), buckets=summaries)
    save(root / 'aim-transitions.json', dict(state='complete', protocol_sha256=sha(root / 'protocol.json'),
        member_proof_sha256=sha(root / 'recovery/verified-members.json'), groups=groups, sources=sources,
        scope='First-life exclusive matched native provider commands after frame100. Adjacent context requires same identity/weapon and next frame. '
              'Explicit selected target+track only, observed clear bbox. Applied view-ray error to observed bbox excludes lead/recoil/muzzle parallax; not shot accuracy. '
              'Mode/target transitions and range/motion strata are descriptive associations, not causality or independent samples. No neural replay.'))
    print({name: group.get('counts', {}) for name, group in groups.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.wait_pid:
        wait_process(args.wait_pid, root / 'aim-transitions-progress.json', 'waiting_for_native_evaluation')
    report(root)
    save(root / 'aim-transitions-progress.json', dict(state='complete', report_sha256=sha(root / 'aim-transitions.json')))

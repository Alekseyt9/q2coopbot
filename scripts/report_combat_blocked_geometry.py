"""Describe observed hull probes on cancelled provider movement; no collision replay."""
import argparse
import collections
import json
import math
import pathlib

from process_combat_architecture_pool import read, save, sha


def report(root, variants, output):
    protocol_path, proof_path = root / 'protocol.json', root / 'recovery/verified-members.json'
    protocol, proof = read(protocol_path), read(proof_path)
    assert proof['state'] == 'complete' and proof['protocol_sha256'] == sha(protocol_path)
    entries = {e['model'] + '-' + e['label']: e for e in protocol['evaluations']}
    assert len(entries) == len(protocol['evaluations'])
    groups, sources = {}, []
    for variant in variants:
        entry = entries[variant]
        counts, reasons, examples = collections.Counter(), collections.Counter(), []
        paths = sorted(pathlib.Path(entry['root']).glob('case-*/s-*/report.json'))
        assert len(paths) == protocol['episodes_per_model']
        for path in paths:
            assert proof['members'][str(path.parent)]['report_sha256'] == sha(path)
            capture = read(path)
            assert capture['capture_complete'] and capture['provenance_valid'] and len(capture['results']) == 1
            result = capture['results'][0]
            assert result['dataset']['command_proof']['accepted']
            steps = pathlib.Path(result['root']) / 'dataset/steps.jsonl'
            digest = sha(steps)
            with steps.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    row = json.loads(line)
                    obs, action, cmd = row['observation'], row['action'], row['sent_command']
                    execution = row.get('server_execution', {})
                    if (row['owner'] != 'provider' or obs['identity']['life'] != 1 or obs['health'] <= 0
                        or not row.get('native_step') or not execution.get('matched')
                        or not execution.get('window_exclusive') or math.hypot(action['forward'], action['side']) <= .05):
                        continue
                    counts['requested_native_movement'] += 1
                    if cmd['Forward'] != 0 or cmd['Side'] != 0:
                        continue
                    counts['fully_cancelled'] += 1
                    reasons.update(e['reason'] for e in row.get('interventions', []) if e['component'] == 'movement')
                    geometry = obs.get('local_geometry')
                    if not geometry or geometry['frame'] != obs['identity']['frame']:
                        counts['geometry_missing_or_stale'] += 1
                        continue
                    # Positive Side is right (negative mathematical yaw). Applied yaw
                    # retains actual aiming, including target modes and protocol rounding.
                    direction = math.degrees(math.atan2(-action['side'], action['forward'])) + row['applied_action']['yaw_delta_degrees']
                    probes = geometry['probes']
                    if not probes:
                        counts['geometry_missing_or_stale'] += 1
                        continue
                    error = lambda probe: abs((probe['yaw_offset_degrees'] - direction + 180) % 360 - 180)
                    probe = min(probes, key=error)
                    clearance = probe['standing_hull_clearance']
                    if clearance is None:
                        counts['nearest_hull_unknown'] += 1
                        continue
                    counts['nearest_hull_known'] += 1
                    bucket = 'under8' if clearance < 8 else '8to32' if clearance < 32 else '32plus'
                    counts['nearest_clearance_' + bucket] += 1
                    alternatives = [p for p in probes if p['standing_hull_clearance'] is not None and p['standing_hull_clearance'] >= 32]
                    counts['some_probe_clearance32plus'] += bool(alternatives)
                    counts['observed_speed100plus'] += math.hypot(*obs['velocity'][:2]) >= 100
                    if len(examples) < 12 and bucket == 'under8':
                        examples.append(dict(worker=str(steps.parent.parent), identity=obs['identity'],
                            action=action, applied_action=row['applied_action'], velocity=obs['velocity'],
                            requested_direction_offset=direction, nearest_probe=probe, angle_error_degrees=error(probe),
                            alternate_probe_offsets=[p['yaw_offset_degrees'] for p in alternatives],
                            interventions=row.get('interventions', [])))
            assert digest == sha(steps), 'Native steps changed'
            sources.append(dict(path=str(steps), sha256=digest, report_sha256=sha(path)))
        groups[variant] = dict(episodes=len(paths), counts=dict(counts), movement_reasons=dict(reasons), examples=examples)
    save(output, dict(state='complete', version='combat_blocked_geometry_v1', groups=groups, sources=sources,
        protocol_sha256=sha(protocol_path), proof_sha256=sha(proof_path),
        scope='Existing client/BSP observations only, exclusive matched first-life provider commands. '
        'Nearest of eight angular probes approximates command direction, not inertia-displaced trajectory. '
        '32-unit clearance in another direction does not establish a safe path or feasible action. '
        'No engine collision or neural replay; example selection is ordered, not random.'))
    return {name: group['counts'] for name, group in groups.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--variants', nargs='+', required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    args = parser.parse_args()
    print(report(args.root.resolve(), args.variants, args.out.resolve()), flush=True)

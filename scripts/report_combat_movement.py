"""First-life movement and held-attack diagnostics from sealed native steps."""
import argparse
import collections
import json
import math
import pathlib

from process_combat_architecture_pool import read, save, sha
from run_combat_spatial_ppo import wait_process


def metrics(path):
    counts = collections.Counter()
    reasons, owners = collections.Counter(), collections.Counter()
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            row = json.loads(line)
            observation = row['observation']
            if observation['identity']['life'] != 1 or observation['health'] <= 0:
                continue
            if not row.get('native_step') or not row.get('server_execution', {}).get('matched'):
                continue
            counts['alive_native_commands'] += 1
            owners[row['owner']] += 1
            cmd = row['sent_command']
            stationary = math.hypot(*observation['velocity'][:2]) < 5
            attack = bool(cmd['Buttons'] & 1)
            counts['stationary'] += stationary
            counts['held_attack'] += attack
            counts['held_attack_stationary'] += attack and stationary
            counts['applied_horizontal_command'] += cmd['Forward'] != 0 or cmd['Side'] != 0
            if row['owner'] == 'provider':
                requested = math.hypot(row['action']['forward'], row['action']['side']) > .05
                counts['provider_requested_movement'] += requested
                counts['provider_movement_stopped'] += requested and cmd['Forward'] == 0 and cmd['Side'] == 0
            for event in row.get('interventions', []):
                if event['component'] == 'movement':
                    reasons[event['reason']] += 1
    assert counts['alive_native_commands'] > 0
    return dict(counts=counts, movement_reasons=reasons, owners=owners)


def report(root):
    protocol, proof, quality = read(root / 'protocol.json'), read(root / 'recovery/verified-members.json'), read(root / 'quality-report.json')
    assert quality['state'] == proof['state'] == 'complete'
    assert quality['protocol_sha256'] == proof['protocol_sha256'] == sha(root / 'protocol.json')
    if (root / 'progress.json').exists():
        progress = read(root / 'progress.json')
        assert progress['stage'] == 'complete' and progress['quality_report_sha256'] == sha(root / 'quality-report.json')
    groups, sources = {}, []
    for entry in protocol['evaluations']:
        rows = []
        for report_path in sorted(pathlib.Path(entry['root']).glob('case-*/s-*/report.json')):
            capture = read(report_path)
            assert proof['members'][str(report_path.parent)]['report_sha256'] == sha(report_path)
            assert capture['capture_complete'] and capture['provenance_valid'] and len(capture['results']) == 1
            result = capture['results'][0]
            assert result['capture_valid'] and result['dispatch_valid'] and result['dataset']['command_proof']['accepted']
            steps = pathlib.Path(result['root']) / 'dataset/steps.jsonl'
            digest = sha(steps)
            item = metrics(steps)
            assert digest == sha(steps), 'Steps changed during measurement'
            item.update(seed=result['seed'], case=report_path.parent.parent.name,
                        win=bool(result.get('goal_stop') and result['goal_stop']['reason'] == 'combat_goal_complete'),
                        death=result['first_life']['end_reason'] == 'first_observed_death')
            rows.append(item)
            sources.append(dict(report=str(report_path), report_sha256=sha(report_path), steps=str(steps), steps_sha256=digest))
        assert len(rows) == protocol['episodes_per_model']
        counts, reasons, owners = collections.Counter(), collections.Counter(), collections.Counter()
        for item in rows:
            counts.update(item['counts']); reasons.update(item['movement_reasons']); owners.update(item['owners'])
        fraction = lambda numerator, denominator: counts[numerator] / counts[denominator] if counts[denominator] else None
        groups[entry['model'] + '-' + entry['label']] = dict(episodes=len(rows), counts=counts,
            pooled_stationary_fraction=fraction('stationary', 'alive_native_commands'),
            held_attack_stationary_fraction=fraction('held_attack_stationary', 'held_attack'),
            provider_requested_movement_stopped_fraction=fraction('provider_movement_stopped', 'provider_requested_movement'),
            mean_episode_stationary_fraction=sum(r['counts']['stationary'] / r['counts']['alive_native_commands'] for r in rows) / len(rows),
            movement_reasons=reasons, owners=owners, episodes_detail=rows)
    output = dict(version='combat_first_life_movement_v1', state='complete', groups=groups, sources=sources,
                  protocol_sha256=sha(root / 'protocol.json'), proof_sha256=sha(root / 'recovery/verified-members.json'),
                  quality_sha256=sha(root / 'quality-report.json'),
                  scope='First-life alive observations with matched native command intervals only; excludes preparation outside the exported steps and later lives. Stationary means observed planar speed below5 world units/s. Held attack is the sent attack button, not a verified shot. Physical speed can include knockback. Provider movement cancellation is unavailable for rules with guards embedded before capture. Episode and pooled fractions differ when trajectories have different lengths. No neural execution or claim of successful evasion.')
    save(root / 'first-life-movement.json', output)
    percent = lambda value: 'n/a' if value is None else f'{100 * value:.2f}%'
    lines = ['# First-life native movement', '', '| Variant | Stationary, pooled | Stationary, mean episode | Held attack while stationary | Provider movement fully stopped |', '| --- | ---: | ---: | ---: | ---: |']
    for name, group in groups.items():
        lines.append(f"| {name} | {percent(group['pooled_stationary_fraction'])} | {percent(group['mean_episode_stationary_fraction'])} | {percent(group['held_attack_stationary_fraction'])} | {percent(group['provider_requested_movement_stopped_fraction'])} |")
    lines.extend(['', output['scope'], ''])
    (root / 'first-life-movement.md').write_text('\n'.join(lines), encoding='utf-8')
    return dict(state='complete', episodes=sum(g['episodes'] for g in groups.values()), report=str(root / 'first-life-movement.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.wait_pid:
        wait_process(args.wait_pid, root / 'movement-progress.json', stage='waiting_for_evaluation')
    result = report(root)
    save(root / 'movement-progress.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

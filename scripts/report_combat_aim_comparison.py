"""Compact sealed comparison of native quality and descriptive aim diagnostics."""
import argparse
import pathlib

from process_combat_architecture_pool import read, save, sha
from run_combat_spatial_ppo import wait_process


def report(root):
    status, protocol, quality = [read(root / p) for p in ('progress.json', 'protocol.json', 'quality-report.json')]
    assert status['stage'] == 'complete' and status['diagnostics_complete']
    assert sha(root / 'quality-report.json') == status['quality_report_sha256']
    assert quality['protocol_sha256'] == sha(root / 'protocol.json')
    for file, field in [('diagnostics-acceptance.json', 'diagnostics_acceptance_sha256'),
                        ('ownership-acceptance.json', 'ownership_acceptance_sha256'),
                        ('storage-final.json', 'physical_storage_sha256')]:
        assert sha(root / file) == status[field]
    diagnostics = read(root / 'diagnostics-acceptance.json')
    assert diagnostics['state'] == 'complete' and diagnostics['quality_report_sha256'] == sha(root / 'quality-report.json')
    reports = {}
    for file in ('machinegun-hits.json', 'machinegun-selected-target-aim.json', 'selected-target-aim.json', 'aim-modes.json'):
        assert diagnostics['reports'][file] == sha(root / file)
        reports[file] = read(root / file)
        assert reports[file]['protocol_sha256'] == sha(root / 'protocol.json')
    ownership = read(root / 'ownership-acceptance.json')
    assert ownership['clean'], 'Visible combat fallback prevents accepted learned comparison'
    configs = read(root / 'sampling-config-audit.json')
    assert configs['state'] == 'complete' and not configs['repeated_stochastic_execution_configs']
    assert configs['protocol_sha256'] == sha(root / 'protocol.json')
    rows = []
    for variant in quality['variants']:
        name = variant['variant']
        hits = reports['machinegun-hits.json']['groups'][name]
        aim = reports['machinegun-selected-target-aim.json']['groups'][name]
        frame_aim = reports['selected-target-aim.json']['groups'][name]
        modes = reports['aim-modes.json']['groups'][name]
        shots = aim.get('all', {})
        rows.append(dict(variant=name, episodes=variant['episodes'], wins=variant['wins'], deaths=variant['deaths'],
            mean_received_damage=variant['mean_received_damage'], native_mg_shots=hits.get('counts', {}).get('shots', 0),
            native_mg_live_damage_fraction=hits.get('live_monster_damage_fraction'),
            selected_mg_shots=shots.get('shots', 0), selected_mg_post_recoil_error=shots.get('post_recoil_degrees'),
            native_mg_range=aim.get('descriptive_strata', {}),
            native_mg_motion={key: value for key, value in hits.get('descriptive_strata', {}).items() if key in ('moving', 'stationary')},
            selected_firing_view_error=frame_aim.get('firing', {}).get('angular_mean_degrees'),
            mode_switches=modes.get('mode_switches'), mode_adjacent_frames=modes.get('adjacent_frames')))
    scopes = ('Native MG damage fraction measures shots with live-monster damage, not isolated aim accuracy or selected-target success. '
              'Selected MG angle is muzzle/recoil ray versus pre-command observed explicit bbox; it differs from applied view-ray frame error. '
              'Range/motion strata and mode switches are descriptive. Mode switches are declared provider-frame counts, not exclusively native commands. '
              'Different actions alter the states and shot opportunities. '
              'Two RNG arms share generated conditions and are not independent sample doubling. Development only, no training/test/promotion or neural replay.')
    save(root / 'aim-comparison.json', dict(state='complete', protocol_sha256=sha(root / 'protocol.json'),
        quality_sha256=sha(root / 'quality-report.json'), diagnostics_sha256=sha(root / 'diagnostics-acceptance.json'),
        sampling_configs_sha256=sha(root / 'sampling-config-audit.json'),
        sources={file: sha(root / file) for file in reports}, rows=rows, scope=scopes))
    def number(value):
        return 'unknown' if value is None else f'{value:.2f}'
    lines = ['# Sealed combat aim comparison', '', '| Variant | Wins / episodes | Deaths | Received damage | Native MG shots / damage fraction | Selected MG shots / post-recoil error | Firing view error | Mode switches / adjacent frames |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        fraction = row['native_mg_live_damage_fraction']
        lines.append(f"| {row['variant']} | {row['wins']}/{row['episodes']} | {row['deaths']} | {number(row['mean_received_damage'])} | "
            f"{row['native_mg_shots']} / {number(100 * fraction if fraction is not None else None)}% | "
            f"{row['selected_mg_shots']} / {number(row['selected_mg_post_recoil_error'])}° | {number(row['selected_firing_view_error'])}° | "
            f"{row['mode_switches'] if row['mode_switches'] is not None else 'unknown'} / {row['mode_adjacent_frames'] if row['mode_adjacent_frames'] is not None else 'unknown'} |")
    lines += ['', '| Variant | Native MG distance stratum | Selected shots | Post-recoil error | Selected target damage shots |', '|---|---|---:|---:|---:|']
    for row in rows:
        for label, group in row['native_mg_range'].items():
            if not label.startswith('distance_'):
                continue
            lines.append(f"| {row['variant']} | {label} | {group['shots']} | {number(group['post_recoil_degrees'])}° | {group['selected_target_damage']} |")
    lines += ['', '| Variant | Native MG motion stratum | Shots | Live-monster damage fraction |', '|---|---|---:|---:|']
    for row in rows:
        for label, group in row['native_mg_motion'].items():
            value = group.get('live_monster_damage_fraction')
            lines.append(f"| {row['variant']} | {label} | {group['counts'].get('shots', 0)} | {number(100 * value if value is not None else None)}% |")
    lines += ['', scopes, '']
    (root / 'aim-comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    print([(r['variant'], r['wins'], r['deaths']) for r in rows], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.wait_pid:
            wait_process(args.wait_pid, root / 'aim-comparison-progress.json', 'waiting_for_sealed_evaluation')
        report(root)
        save(root / 'aim-comparison-progress.json', dict(state='complete', report_sha256=sha(root / 'aim-comparison.json')))
    except Exception as error:
        save(root / 'aim-comparison-progress.json', dict(state='failed', error=str(error)))
        raise

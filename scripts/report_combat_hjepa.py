"""Summarize the paired H-JEPA smoke screen without promoting the candidate."""
import argparse
import hashlib
import json
import pathlib


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=pathlib.Path, required=True)
    args = ap.parse_args()
    root = args.root
    result = dict(version='hjepa_pilot_v1', promotion=False, modes={},
                  scope='Fixed Machinegun, one Parasite on base1, skill 1; 4 paired fresh seeds. '
                        'Connectivity and narrow gameplay screen only. Not campaign, transfer or superiority evidence.')
    cohorts = []
    for mode in ('hjepa', 'rules'):
        path = root / ('eval-' + mode) / 'report.json'
        report = json.loads(path.read_text())
        if not report['provenance_valid'] or not report['capture_complete']:
            raise ValueError('incomplete/unverified native capture')
        rows = []
        for r in report['results']:
            if not r['capture_valid'] or not r['dispatch_valid'] or not r['dataset']['command_proof']['accepted']:
                raise ValueError('unverified native command/dispatch')
            damage = r['first_life']['damage']
            rows.append(dict(seed=r['seed'], win=bool(r['goal_stop']), death=bool(r['death_stop']),
                             received_hp=damage['received_health_damage'],
                             self_hp=damage['self_health_damage'],
                             provider_frames=r['provider_controlled_frames'],
                             guard_interventions=r['guard_interventions'],
                             p95_us=r['selection_p95_us'], p99_us=r['selection_p99_us']))
        cohorts.append(sorted(r['seed'] for r in rows))
        result['modes'][mode] = dict(episodes=rows, wins=sum(r['win'] for r in rows),
                                    deaths=sum(r['death'] for r in rows),
                                    received_hp=sum(r['received_hp'] for r in rows),
                                    provider_frames=sum(r['provider_frames'] for r in rows),
                                    guard_interventions=sum(r['guard_interventions'] for r in rows))
    if cohorts[0] != cohorts[1] or len(cohorts[0]) != len(set(cohorts[0])):
        raise ValueError('unpaired or duplicate seed cohorts')
    weights = root / 'model/weights.json'
    result['weights_sha256'] = hashlib.sha256(weights.read_bytes()).hexdigest()
    result['training_report'] = str(root / 'model/report.json')
    result['limitation'] = ('Observable utility uses health minus visible-enemy count: occlusion can change it. '
                            'VICReg-style regularization, structured features and BC proposals differ from the visual H-JEPA paper. '
                            'No weapon selection. Experimental synchronous fixed-weapon harness only.')
    out = root / 'summary.json'
    out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

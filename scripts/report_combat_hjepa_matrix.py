"""Aggregate verified native matrices and low-level action diagnostics."""
import argparse
import collections
import json
import math
import pathlib


def aim_error(o, a):
    enemies = [e for e in o['enemies'] if e.get('clear_shot') and e.get('observed_solid')]
    if not enemies:
        return None
    e = min(enemies, key=lambda e: e['distance'])
    x, y, z = e['relative']
    solid = e['observed_solid']
    bottom, top = -((solid >> 5) & 31) * 8, ((solid >> 10) & 63) * 8 - 32
    z += (bottom + top) / 2 + (2 if o['ducked'] else -22)
    yaw = math.degrees(math.atan2(y, x))
    pitch = -math.degrees(math.atan2(z, math.hypot(x, y)))
    current_pitch, current_yaw = (v * 360 / 65536 for v in o['view_angles'][:2])
    wrap = lambda d: (d + 180) % 360 - 180
    return math.hypot(wrap(yaw-current_yaw-a['yaw_delta_degrees']),
                      wrap(pitch-current_pitch-a['pitch_delta_degrees']))


def summarize(paths):
    episodes, errors, attack_errors, owners = [], [], [], collections.Counter()
    for path in paths:
        report = json.loads(path.read_text())
        if not report['capture_complete'] or not report['provenance_valid']:
            raise ValueError('invalid capture ' + str(path))
        for r in report['results']:
            if not r['capture_valid'] or not r['dataset']['command_proof']['accepted']:
                raise ValueError('invalid native commands')
            damage = r['first_life']['damage']
            episodes.append(dict(seed=r['seed'], map=r['first_life']['map'],
                                 win=bool(r['goal_stop']), death=bool(r['death_stop']),
                                 timeout=not bool(r['goal_stop']) and not bool(r['death_stop']),
                                 received_hp=damage['received_health_damage'],
                                 guard_interventions=r['guard_interventions'],
                                 provider_frames=r['provider_controlled_frames'],
                                 p95_us=r['selection_p95_us']))
        for steps in path.parent.glob('worker-*/dataset/steps.jsonl'):
            for line in steps.read_text().splitlines():
                s = json.loads(line)
                o = s['observation']
                if not s.get('native_step') or not o['enemies']:
                    continue
                owners[s['owner']] += 1
                err = aim_error(o, s['applied_action'])
                if err is not None:
                    errors.append(err)
                    if s['applied_action']['attack']:
                        attack_errors.append(err)
    if len({r['seed'] for r in episodes}) != len(episodes):
        raise ValueError('duplicate seed')
    return dict(episodes=episodes, wins=sum(r['win'] for r in episodes),
                deaths=sum(r['death'] for r in episodes), received_hp=sum(r['received_hp'] for r in episodes),
                timeouts=sum(r['timeout'] for r in episodes),
                guard_interventions=sum(r['guard_interventions'] for r in episodes),
                owner_frames=dict(owners),
                mean_post_action_aim_error_deg=sum(errors)/len(errors) if errors else None,
                mean_attack_aim_error_deg=sum(attack_errors)/len(attack_errors) if attack_errors else None,
                attack_observations=len(attack_errors))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--matrix', type=pathlib.Path, action='append', required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    a = ap.parse_args()
    groups = collections.defaultdict(list)
    for root in a.matrix:
        for path in root.glob('*/case-*/report.json'):
            groups[path.parent.parent.name].append(path)
    results = {name: summarize(paths) for name, paths in groups.items()}
    cohorts = {tuple(sorted(r['seed'] for r in result['episodes'])) for result in results.values()}
    if len(cohorts) != 1:
        raise ValueError('incomplete or unpaired matrix')
    count = len(next(iter(cohorts)))
    maps = sorted({r['map'] for result in results.values() for r in result['episodes']})
    report = dict(version='hjepa_native_matrix_v1', variants=results, promotion=False,
                  scope=f'Frozen paired {count}-seed screen on {", ".join(maps)}; site and split bindings in source plans. '
                        'Tiny screen, not superiority proof. '
                        'H-JEPA legacy execution; PPO stock_weapons_fire_v1 plus required checked-wall guard: '
                        'system-level reference comparison, not pure architecture isolation. '
                        'Aim diagnostic approximates static observed bbox alignment without recoil/motion, not shot accuracy.')
    a.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: {j: r[j] for j in ('wins','deaths','timeouts','received_hp','guard_interventions','mean_attack_aim_error_deg')}
                      for k, r in results.items()}, indent=2))


if __name__ == '__main__':
    main()

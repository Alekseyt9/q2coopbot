"""Report completed reward A/B using native evidence and observed commands only."""
import argparse
import json
import math
import pathlib
import sys

from process_combat_architecture_pool import read, save, sha, run
from report_combat_selected_target_aim import measure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=pathlib.Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    assert read(root/'progress.json')['stage'] == 'captures_complete'
    repo = pathlib.Path(__file__).resolve().parents[1]
    evaluation = root/'evaluation'
    pool = read(evaluation/'pool/report.json')
    assert pool['state'] == 'complete' and pool['source_unchanged'] and not any(j['error'] for j in pool['jobs'])
    manifest = read(pathlib.Path(pool['jobs'][0]['root'])/'manifest.json')
    run([sys.executable,repo/'scripts/verify_combat_evaluation_members.py','--root',evaluation,
         '--source-fingerprint',manifest['source_fingerprint'],
         '--native-fingerprint',manifest['native_source_fingerprint']], root/'verify-evaluation.log')
    run([sys.executable,repo/'scripts/report_combat_architecture_evaluation.py','--root',evaluation,
         '--member-proof',evaluation/'recovery/verified-members.json'],root/'quality.log')
    run([sys.executable,repo/'scripts/report_combat_selected_target_aim.py','--root',evaluation],root/'selected-target.log')
    for script,log in [('report_combat_machinegun_hits.py','machinegun-hits.log'),
                       ('report_combat_blaster_hits.py','blaster-hits.log')]:
        run([sys.executable,repo/'scripts'/script,'--root',evaluation],root/log)
    quality = read(evaluation/'quality-report.json')
    episodes = read(evaluation/'quality-episodes.json')
    cycle_audit = None
    reward = repo/'scripts/scenarios/combat-reward-target-sequence-v7.json'
    if (root/'protocol.json').exists() and read(root/'protocol.json').get('quality_reward_sha256')==sha(reward):
        directory = root/'target-cycle-audit'
        run([sys.executable,repo/'scripts/audit_combat_sequence_reward.py','--evaluation',evaluation,
             '--exporter',pathlib.Path(pool['jobs'][0]['root'])/'q2combat-export.exe',
             '--reward',reward,'--out',directory,'--compress'],root/'target-cycle-audit.log')
        cycle_audit = read(directory/'report.json')
    groups = {}
    for entry in read(evaluation/'protocol.json')['evaluations']:
        count = attack = standing_attack = measured_attack = far_attack = big_turn = upward_attack = 0
        total_yaw = angular = max_yaw = 0.
        for path in pathlib.Path(entry['root']).glob('case-*/s-*/report.json'):
            member = read(path)
            trace = pathlib.Path(member['results'][0]['root'])/'bot.jsonl'
            seen = set()
            with trace.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    c = json.loads(line).get('combat_policy')
                    if not c or c.get('selection',{}).get('owner') != 'provider':
                        continue
                    o = c['observation']
                    if o['identity']['life'] != 1 or o['identity']['frame'] <= 100 or o['health'] <= 0:
                        continue
                    identity = tuple(o['identity'].get(k) for k in ('connection','spawncount','frame'))
                    if identity in seen:
                        continue
                    seen.add(identity)
                    a = c['applied']
                    count += 1
                    total_yaw += abs(a['yaw_delta_degrees'])
                    max_yaw = max(max_yaw,abs(a['yaw_delta_degrees']))
                    big_turn += abs(a['yaw_delta_degrees']) >= 45
                    if a['attack']:
                        attack += 1
                        standing_attack += math.hypot(*o['velocity'][:2]) < 10
                        pitch = o['view_angles'][0]*360/65536+a['pitch_delta_degrees']
                        pitch = (pitch+180)%360-180
                        upward_attack += pitch < -35
                        aimed = measure(c)
                        if aimed and aimed['status'] == 'measured':
                            measured_attack += 1
                            angular += aimed['angular']
                            far_attack += aimed['angular'] > 30
        label = entry['model']+'-'+entry['label']
        native = [x for x in episodes if x['variant'] == label]
        groups[label] = dict(provider_frames=count, attack_commands=attack,
            mean_abs_yaw_per_frame=total_yaw/count if count else None,
            max_abs_yaw_delta=max_yaw, turns_at_least_45_degrees=big_turn,
            upward_attack_commands=upward_attack,
            standing_attack_fraction=standing_attack/attack if attack else None,
            selected_target_measured_attack_commands=measured_attack,
            nominal_mean_attack_angular_error=angular/measured_attack if measured_attack else None,
            nominal_far_attack_fraction=far_attack/measured_attack if measured_attack else None,
            mean_game_frames=sum(x['frames'] for x in native)/len(native))
    pairs = {}
    labels = {e['label'] for e in read(evaluation/'protocol.json')['evaluations']}
    for label in sorted(labels):
        baseline = {(x['episode'],x['seed']):x for x in episodes if x['variant']=='control-'+label}
        candidate = {(x['episode'],x['seed']):x for x in episodes if x['variant']=='quality-'+label}
        assert baseline and baseline.keys() == candidate.keys()
        pairs[label] = dict(gained_wins=sum(candidate[k]['win'] and not baseline[k]['win'] for k in baseline),
                           lost_wins=sum(baseline[k]['win'] and not candidate[k]['win'] for k in baseline))
    result = dict(state='complete',quality_report_sha256=sha(evaluation/'quality-report.json'),
        variants=quality['variants'], command_metrics=groups, paired=pairs,
        native_machinegun=read(evaluation/'machinegun-hits.json')['groups'],
        native_blaster=read(evaluation/'blaster-projectile-hits.json')['groups'],
        native_reports_sha256={f:sha(evaluation/f) for f in ('machinegun-hits.json','blaster-projectile-hits.json')},
        scope='Common development cohort, not final-test superiority. Commands are not native shot counts; '
              'standing is observed horizontal speed, not proof of uselessness. Nominal selected-target angles '
              'exclude recoil/projectile lead/obstruction; far angle is not an exact native miss. No promotion.')
    if cycle_audit is not None:
        cycle_groups = {}
        for entry in read(evaluation/'protocol.json')['evaluations']:
            label = entry['model']+'-'+entry['label']
            rows = [r for r in cycle_audit['members'] if r['label']==label]
            cycles = sum(len(r['cycles']) for r in rows)
            steps = sum(r['steps'] for r in rows)
            cycle_groups[label] = dict(episodes=len(rows),available_steps=steps,cycles=cycles,
                cycles_per1000_available_steps=1000*cycles/steps if steps else None)
        result['target_cycles'] = cycle_groups
        result['target_cycle_audit_sha256'] = sha(root/'target-cycle-audit/report.json')
    save(root/'result.json',result)
    lines = ['# Парная проверка боевых моделей','',result['scope'],'',
             '| Ветка | Победы | Смерти | Полученный урон | Поворот, градусов/кадр | Атака стоя |',
             '|---|---:|---:|---:|---:|---:|']
    for v in result['variants']:
        m = groups[v['variant']]
        lines.append(f"| {v['variant']} | {v['wins']}/{v['episodes']} | {v['deaths']} | "
                     f"{v['mean_received_damage']:.1f} | {m['mean_abs_yaw_per_frame']:.2f} | "
                     f"{m['standing_attack_fraction']:.1%} |")
    for label, paired in pairs.items():
        lines += ['',f"{label}: новых побед {paired['gained_wins']}, потерянных {paired['lost_wins']}."]
    if cycle_audit is not None:
        lines += ['','| Ветка | A→B→A | На1000 подтвержденных reward-переходов |','|---|---:|---:|']
        for label,m in result['target_cycles'].items():
            lines.append(f"| {label} | {m['cycles']} | {m['cycles_per1000_available_steps']:.2f} |")
    (root/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(root/'progress.json',dict(stage='complete',result_sha256=sha(root/'result.json'),promotion=None))
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()

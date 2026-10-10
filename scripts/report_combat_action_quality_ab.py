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
                       ('report_combat_blaster_hits.py','blaster-hits.log'),
                       ('report_combat_first_shot_latency.py','first-shot-latency.log')]:
        run([sys.executable,repo/'scripts'/script,'--root',evaluation],root/log)
    quality = read(evaluation/'quality-report.json')
    episodes = read(evaluation/'quality-episodes.json')
    cycle_audit = None
    reward = repo/'scripts/scenarios/combat-reward-target-sequence-v7.json'
    selected_reward = repo/'scripts/scenarios/combat-reward-selected-aim-v8.json'
    quality_reward_sha = read(root/'protocol.json').get('quality_reward_sha256') if (root/'protocol.json').exists() else None
    if quality_reward_sha in (sha(reward),sha(selected_reward)):
        directory = root/'target-cycle-audit'
        run([sys.executable,repo/'scripts/audit_combat_sequence_reward.py','--evaluation',evaluation,
             '--exporter',pathlib.Path(pool['jobs'][0]['root'])/'q2combat-export.exe',
             '--reward',reward,'--out',directory,'--compress'],root/'target-cycle-audit.log')
        cycle_audit = read(directory/'report.json')
    selected_audit = None
    if quality_reward_sha==sha(selected_reward):
        directory = root/'selected-aim-audit'
        run([sys.executable,repo/'scripts/audit_combat_sequence_reward.py','--evaluation',evaluation,
             '--exporter',pathlib.Path(pool['jobs'][0]['root'])/'q2combat-export.exe',
             '--reward',selected_reward,'--out',directory,'--compress'],root/'selected-aim-audit.log')
        selected_audit = read(directory/'report.json')
    groups = {}
    for entry in read(evaluation/'protocol.json')['evaluations']:
        count = attack = standing_attack = measured_attack = far_attack = big_turn = upward_attack = 0
        total_yaw = angular = max_yaw = 0.
        longest_standing_attack = longest_visible_no_attack = 0
        standing_attack_with_move = 0
        for path in pathlib.Path(entry['root']).glob('case-*/s-*/report.json'):
            member = read(path)
            trace = pathlib.Path(member['results'][0]['root'])/'bot.jsonl'
            seen = set()
            previous_identity = None
            standing_run = waiting_run = 0
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
                    contiguous = (previous_identity is not None and
                                  identity[:2] == previous_identity[:2] and
                                  identity[2] == previous_identity[2]+1)
                    if not contiguous:
                        standing_run = waiting_run = 0
                    previous_identity = identity
                    standing = math.hypot(*o['velocity'][:2]) < 10
                    standing_run = standing_run+1 if standing and a['attack'] else 0
                    waiting_run = (waiting_run+1 if not a['attack'] and
                                   any(e.get('clear_shot') is True for e in o['enemies']) else 0)
                    longest_standing_attack = max(longest_standing_attack,standing_run)
                    longest_visible_no_attack = max(longest_visible_no_attack,waiting_run)
                    standing_attack_with_move += (standing and a['attack'] and
                                               (abs(a['forward']) > 0 or abs(a['side']) > 0))
                    count += 1
                    total_yaw += abs(a['yaw_delta_degrees'])
                    max_yaw = max(max_yaw,abs(a['yaw_delta_degrees']))
                    big_turn += abs(a['yaw_delta_degrees']) >= 45
                    if a['attack']:
                        attack += 1
                        standing_attack += standing
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
            standing_attack_with_movement_commands=standing_attack_with_move,
            longest_standing_attack_game_seconds=longest_standing_attack*.1,
            longest_visible_no_attack_game_seconds=longest_visible_no_attack*.1,
            selected_target_measured_attack_commands=measured_attack,
            nominal_mean_attack_angular_error=angular/measured_attack if measured_attack else None,
            nominal_far_attack_fraction=far_attack/measured_attack if measured_attack else None,
            mean_game_frames=sum(x['frames'] for x in native)/len(native))
    pairs = {}
    family_pairs = []
    labels = {e['label'] for e in read(evaluation/'protocol.json')['evaluations']}
    for label in sorted(labels):
        baseline = {(x['episode'],x['seed']):x for x in episodes if x['variant']=='control-'+label}
        candidate = {(x['episode'],x['seed']):x for x in episodes if x['variant']=='quality-'+label}
        assert baseline and baseline.keys() == candidate.keys()
        pairs[label] = dict(gained_wins=sum(candidate[k]['win'] and not baseline[k]['win'] for k in baseline),
                           lost_wins=sum(baseline[k]['win'] and not candidate[k]['win'] for k in baseline))
        for family in sorted({k[0] for k in baseline}):
            keys = [k for k in baseline if k[0] == family]
            family_pairs.append(dict(label=label,episode=family,episodes=len(keys),
                control_wins=sum(baseline[k]['win'] for k in keys),
                quality_wins=sum(candidate[k]['win'] for k in keys),
                control_deaths=sum(baseline[k]['death'] for k in keys),
                quality_deaths=sum(candidate[k]['death'] for k in keys),
                gained_wins=sum(candidate[k]['win'] and not baseline[k]['win'] for k in keys),
                lost_wins=sum(baseline[k]['win'] and not candidate[k]['win'] for k in keys)))
    result = dict(state='complete',quality_report_sha256=sha(evaluation/'quality-report.json'),
        variants=quality['variants'], command_metrics=groups, paired=pairs,paired_families=family_pairs,
        native_machinegun=read(evaluation/'machinegun-hits.json')['groups'],
        native_blaster=read(evaluation/'blaster-projectile-hits.json')['groups'],
        first_shot_latency=read(evaluation/'first-shot-latency.json')['groups'],
        native_reports_sha256={f:sha(evaluation/f) for f in ('machinegun-hits.json','blaster-projectile-hits.json','first-shot-latency.json')},
        scope='Common development cohort, not final-test superiority. Commands are not native shot counts; '
              'standing is observed horizontal speed, not proof of uselessness. '
              'Standing attack and visible no-attack spans require consecutive observed provider frames; '
              'movement commands at low speed do not prove a collision. '
              'Nominal selected-target angles exclude recoil/projectile lead/obstruction; '
              'far angle is not an exact native miss. No promotion.')
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
    if selected_audit is not None:
        references = {}
        for entry in read(evaluation/'protocol.json')['evaluations']:
            label = entry['model']+'-'+entry['label']
            rows = [r for r in selected_audit['members'] if r['label']==label]
            reasons = {}
            for row in rows:
                for reason,count in row['aim_references'].items(): reasons[reason] = reasons.get(reason,0)+count
            references[label] = dict(available_steps=sum(r['steps'] for r in rows),reasons=reasons,
                positive_progress=sum(r['aim_positive'] for r in rows),negative_progress=sum(r['aim_negative'] for r in rows))
        result['selected_aim_references'] = references
        result['selected_aim_audit_sha256'] = sha(root/'selected-aim-audit/report.json')
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
    lines += ['','| Семейство | Боев на ветку | Победы control / quality | Смерти control / quality |',
              '|---|---:|---:|---:|']
    for row in family_pairs:
        lines.append(f"| {row['episode']} ({row['label']}) | {row['episodes']} | "
                     f"{row['control_wins']} / {row['quality_wins']} | "
                     f"{row['control_deaths']} / {row['quality_deaths']} |")
    if cycle_audit is not None:
        lines += ['','| Ветка | A→B→A | На1000 подтвержденных reward-переходов |','|---|---:|---:|']
        for label,m in result['target_cycles'].items():
            lines.append(f"| {label} | {m['cycles']} | {m['cycles_per1000_available_steps']:.2f} |")
    (root/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(root/'progress.json',dict(stage='complete',result_sha256=sha(root/'result.json'),promotion=None))
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()

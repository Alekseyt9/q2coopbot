"""Read a bounded, byte-pinned prefix of a live companion trace; no inference."""
import argparse
import collections
import hashlib
import json
import math
import pathlib


def review(path, idle_stop):
    limit = path.stat().st_size
    digest = hashlib.sha256()
    counts = collections.Counter()
    guards = collections.Counter()
    maps = collections.Counter()
    windows = []
    window = None
    last = None
    consumed = idle = malformed = 0
    stopped = False
    def close():
        nonlocal window
        if window:
            windows.append(window)
            window = None
    with path.open('rb') as stream:
        while consumed < limit:
            line = stream.readline(limit - consumed)
            if not line:
                break
            digest.update(line)
            consumed += len(line)
            if not line.endswith(b'\n'):
                malformed += 1
                break
            row = json.loads(line)
            key = tuple(row.get(k) for k in ('connection','spawncount','map','frame'))
            if key == last:
                counts['duplicate_snapshot_rows_skipped'] += 1
                continue
            last = key
            counts['snapshots'] += 1
            maps[row.get('map','unknown')] += 1
            policy = row.get('combat_policy', {})
            selection = policy.get('selection', {})
            enemies = policy.get('observation', {}).get('enemies', [])
            idle = idle + 1 if row.get('goal') == 'wait_for_teammate' and row.get('teammate') is None and not enemies else 0
            if idle >= idle_stop:
                stopped = True
                break
            counts['observed_enemy_snapshots'] += bool(enemies)
            sent = row.get('sent_command', {})
            counts['sent_attack_snapshots'] += bool(sent.get('Buttons',0) & 1)
            counts['sent_movement_snapshots'] += bool(sent.get('Forward') or sent.get('Side') or sent.get('Up'))
            if selection.get('owner') != 'provider':
                counts['noncombat_or_fallback_snapshots'] += 1
                close()
                continue
            counts['provider_snapshots'] += 1
            candidate = selection.get('candidate', {})
            clear = any(e.get('clear_shot') for e in enemies)
            attack = bool(candidate.get('attack'))
            counts['provider_clear_target_snapshots'] += clear
            counts['provider_attack_snapshots'] += attack
            counts['clear_target_without_candidate_attack'] += clear and not attack
            counts['provider_sent_attack_snapshots'] += bool(sent.get('Buttons',0) & 1)
            speed = math.hypot(*row.get('self_velocity',[0,0,0])[:2])
            counts['provider_speed_below_10'] += speed < 10
            for intervention in selection.get('interventions', []):
                guards[intervention.get('reason',str(intervention))] += 1
            identity = key[:3] + (policy.get('observation',{}).get('identity',{}).get('life'),)
            if window is None or window['identity'] != list(identity):
                close()
                window = dict(identity=list(identity),first_frame=row['frame'],last_frame=row['frame'],
                              snapshots=0,first_candidate_attack_frame=None,first_sent_attack_frame=None,
                              speed_below_10=0,clear_target_without_attack=0,
                              interventions={},pre_first_sent_attack_proposals=0,
                              pre_first_sent_attack_interventions={})
            window['last_frame'] = row['frame']
            window['snapshots'] += 1
            window['speed_below_10'] += speed < 10
            window['clear_target_without_attack'] += clear and not attack
            for intervention in selection.get('interventions', []):
                reason=intervention.get('reason',str(intervention))
                window['interventions'][reason]=window['interventions'].get(reason,0)+1
                if window['first_sent_attack_frame'] is None:
                    window['pre_first_sent_attack_interventions'][reason]=window['pre_first_sent_attack_interventions'].get(reason,0)+1
            if window['first_sent_attack_frame'] is None and attack:
                window['pre_first_sent_attack_proposals'] += 1
            if attack and window['first_candidate_attack_frame'] is None:
                window['first_candidate_attack_frame'] = row['frame']
            if sent.get('Buttons',0) & 1 and window['first_sent_attack_frame'] is None:
                window['first_sent_attack_frame'] = row['frame']
    close()
    return dict(trace=str(path.resolve()),snapshot_limit_bytes=limit,consumed_bytes=consumed,
                consumed_prefix_sha256=digest.hexdigest(),stopped_after_absent_teammate_snapshots=idle if stopped else None,
                incomplete_tail_lines=malformed,counts=dict(counts),maps=dict(maps),interventions=dict(guards),
                provider_windows=windows,scope='One first row per consecutive snapshot key. Commands and observed velocity, not native shots/hits. Windows begin at provider ownership and end on any ownership gap or life/map change. Stops after prolonged absent teammate; later reconnects excluded. Different traces are not paired causal comparisons.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--trace',type=pathlib.Path,action='append',required=True)
    parser.add_argument('--out',type=pathlib.Path,required=True)
    parser.add_argument('--idle-stop',type=int,default=200)
    args=parser.parse_args()
    assert args.idle_stop > 0
    result=dict(version='live_companion_behavior_review_v1',traces=[review(p,args.idle_stop) for p in args.trace])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for trace in result['traces']:
        print(pathlib.Path(trace['trace']).name,trace['counts'],trace['interventions'],flush=True)


if __name__=='__main__':
    main()

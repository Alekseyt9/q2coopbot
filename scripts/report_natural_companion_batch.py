"""Summarize closed native companion traces without model inference."""
import argparse
import collections
import hashlib
import json
import math
import pathlib


def trace_summary(path, expected_map):
    before = path.stat()
    digest = hashlib.sha256()
    counts = collections.Counter()
    goals = collections.Counter()
    guards = collections.Counter()
    windows = []
    last = None
    window = None
    with path.open('rb') as source:
        for line in source:
            digest.update(line)
            row = json.loads(line)
            if row.get('map') != expected_map:
                continue
            key = tuple(row.get(k) for k in ('connection', 'spawncount', 'map', 'frame'))
            if key == last:
                counts['duplicate_rows'] += 1
                continue
            last = key
            counts['snapshots'] += 1
            goals[row.get('goal', '')] += 1
            alive = row.get('health', 0) > 0
            mate = row.get('teammate') is not None
            counts['alive_frames'] += int(alive)
            counts['alive_without_visible_teammate'] += int(alive and not mate)
            selection = (row.get('combat_policy') or {}).get('selection') or {}
            owner = selection.get('owner') == 'provider'
            counts['provider_frames'] += int(owner)
            attack = bool((selection.get('candidate') or {}).get('attack'))
            sent = bool((row.get('sent_command') or {}).get('Buttons', 0) & 1)
            counts['provider_attack_proposals'] += int(owner and attack)
            counts['provider_sent_attack_frames'] += int(owner and sent)
            counts['provider_attack_suppressed_frames'] += int(owner and attack and not sent)
            reason = (row.get('arbitration') or {}).get('limit_reason')
            if owner and attack and not sent:
                guards[reason or 'unattributed'] += 1
            velocity = row.get('self_velocity') or [0, 0, 0]
            static = alive and math.hypot(velocity[0], velocity[1]) < 10
            counts['provider_static_sent_attack_frames'] += int(owner and sent and static)
            visible = any(e.get('clear_shot') is True for e in ((row.get('combat_policy') or {}).get('observation') or {}).get('enemies', []))
            counts['provider_visible_enemy_frames'] += int(owner and visible)
            waiting = alive and row.get('goal') == 'wait_for_teammate' and not mate
            identity = key[:-1]
            if waiting:
                if window and window['identity'] == identity and row['frame'] == window['last_frame'] + 1:
                    window['last_frame'] = row['frame']
                    window['frames'] += 1
                else:
                    if window:
                        windows.append(window)
                    window = dict(identity=identity, first_frame=row['frame'], last_frame=row['frame'], frames=1, self=row.get('self'))
            elif window:
                windows.append(window)
                window = None
    if window:
        windows.append(window)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'trace changed while reading: {path}')
    return dict(path=str(path), bytes=after.st_size, sha256=digest.hexdigest(), counts=dict(counts), goals=dict(goals), suppressed_attack_guards=dict(guards), wait_windows=[w for w in windows if w['frames'] >= 50])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    raw = (root / 'report.json').read_bytes()
    report = json.loads(raw)
    if not report.get('source_unchanged') or any(not r.get('infrastructure_ok') for r in report['results']):
        raise RuntimeError('a closed, source-stable batch without infrastructure failures is required')
    cases = []
    for result in report['results']:
        case_root = root / f"{result['map']}-{result['mode']}-{result['seed']}"
        companion = trace_summary(case_root / 'bot.jsonl', result['map'])
        leader = trace_summary(case_root / 'teammate.jsonl', result['map'])
        if companion['counts'].get('provider_frames', 0) != result['provider_frames']:
            raise RuntimeError(f'provider count differs from native harness report: {case_root}')
        cases.append(dict(map=result['map'], mode=result['mode'], seed=result['seed'], result=result, companion=companion, leader=leader))
    output = dict(report_sha256=hashlib.sha256(raw).hexdigest(), scope='Snapshot commands are not native shots. Static means XY speed below10. Missing teammate means not observed, not proven dead. Long wait means at least50 consecutive game frames.', cases=cases)
    (root / 'behavior-summary.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    for c in cases:
        print(json.dumps(dict(map=c['map'], mode=c['mode'], seed=c['seed'], counts=c['companion']['counts'], guards=c['companion']['suppressed_attack_guards'], wait_windows=c['companion']['wait_windows'])))


if __name__ == '__main__':
    main()

"""Reconcile monitor aim counters against the admitted native-control interval.

Preparation telemetry can start before the first server-dispatched policy step.
Only its independently reconstructed contribution is subtracted; all terminal
outcomes and all admitted-frame counters remain exact comparisons.
"""
import json
from collections import Counter
from pathlib import Path

from recover_combat_anchor_screen import terminal_receipt


def admitted_terminal_receipt(result, first_admitted_frame):
    receipt = terminal_receipt(result)
    life = receipt['first_life']
    assert type(first_admitted_frame) is int
    assert life['start_frame'] <= first_admitted_frame <= life['end_frame']
    path = Path(result['root'])/'bot.jsonl'
    with path.open(encoding='utf-8-sig') as stream:
        rows = [json.loads(line) for line in stream]
    rows = [r for r in rows if life['start_frame'] <= r['frame'] <= life['end_frame']]
    assert [r['frame'] for r in rows] == list(range(life['start_frame'], life['end_frame']+1))
    assert all(r['map'] == life['map'] and r['spawncount'] == life['spawncount']
               and r['self_entity'] == life['bot_entity'] for r in rows)
    full, warmup, admitted = Counter(), Counter(), Counter()
    for row in rows:
        entity = (row.get('arbitration') or {}).get('aim_entity')
        if not entity:
            continue
        targets = [e for e in row.get('enemies', []) if e['id'] == entity]
        if len(targets) != 1:
            continue
        cls = targets[0]['class']
        attack = bool(row['sent_command']['Buttons'] & 1)
        for counts in (full, warmup if row['frame'] < first_admitted_frame else admitted):
            counts[(cls, 'aim_frames')] += 1
            counts[(cls, 'attack_command_frames')] += int(attack)

    def summarize(counts):
        return [dict({'class': cls}, **{key: counts[(cls, key)] for key in ('aim_frames', 'attack_command_frames')})
                for cls in sorted({cls for cls, key in counts})]

    # Never silently discard an unexplained aggregate mismatch.
    assert sorted(life['aim_by_class'], key=lambda x: x['class']) == summarize(full), 'Monitor aim summary cannot be reconstructed'
    assert full == warmup+admitted
    life['aim_by_class'] = summarize(admitted)
    proof = dict(first_admitted_frame=first_admitted_frame,
                 full_monitor_counts=summarize(full), preparation_counts=summarize(warmup),
                 admitted_counts=summarize(admitted), monitor_path=str(path))
    return receipt, proof

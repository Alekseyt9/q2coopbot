"""Reconstruct movement telemetry before removing preparation-only counters."""
import json
import math
from collections import Counter
from pathlib import Path

from combat_teacher_native_receipt import admitted_terminal_receipt


def admitted_terminal_receipt_v2(result, first_admitted_frame):
    receipt, proof = admitted_terminal_receipt(result, first_admitted_frame)
    life = receipt['first_life']
    path = Path(result['root'])/'bot.jsonl'
    with path.open(encoding='utf-8-sig') as stream:
        rows = [json.loads(line) for line in stream]
    rows = [r for r in rows if life['start_frame'] <= r['frame'] <= life['end_frame']]
    full, warmup, admitted = Counter(), Counter(), Counter()
    for i, row in enumerate(rows):
        source = (row.get('arbitration') or {}).get('move_source') or ''
        assert isinstance(source, str)
        moved = 0
        if i+1 < len(rows):
            other = rows[i+1]
            assert other['frame'] == row['frame']+1
            dx, dy = (other['self'][j]-row['self'][j] for j in (0,1))
            assert math.isfinite(dx) and math.isfinite(dy)
            moved = int(math.sqrt(dx*dx+dy*dy) > 2)
        for counts in (full, warmup if row['frame'] < first_admitted_frame else admitted):
            counts[(source,'command_frames')] += 1
            counts[(source,'actual_next_frame_steps')] += moved

    def summarize(counts):
        return [dict(source=source, **{key: counts[(source,key)] for key in ('command_frames','actual_next_frame_steps')})
                for source in sorted({source for source,key in counts})]

    assert sorted(life['movement'],key=lambda x:x['source']) == summarize(full), 'Monitor movement summary cannot be reconstructed'
    assert full == warmup+admitted
    life['movement'] = summarize(admitted)
    proof.update(version='combat_teacher_admitted_terminal_v2',
                 full_monitor_movement=summarize(full), preparation_movement=summarize(warmup),
                 admitted_movement=summarize(admitted))
    return receipt, proof

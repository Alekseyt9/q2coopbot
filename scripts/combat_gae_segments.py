"""Explicit credit-assignment cuts over immutable, independently verified rows.

These cuts stop GAE recursion, not temporal observation context or native play.
They never turn a horizon into a terminal state or discard its next-state value.
"""
import math

VERSION = 'combat_gae_segments_v1'


def consecutive(a, b):
    return (a['seed'] == b['seed'] and b['index'] == a['index'] + 1
            and b['frame'] == a['next_frame'])


def segment_rows(rows, max_steps):
    assert type(max_steps) is int and 1 <= max_steps <= 500
    assert rows, 'Cannot segment an empty training corpus'
    view = list(rows)
    seen = set()
    segments = []
    start = 0
    added = 0
    for i, row in enumerate(rows):
        key = (row['seed'], row['index'])
        assert key not in seen, 'Duplicate transition identity'
        seen.add(key)
        assert type(row['terminal']) is bool and type(row['truncated']) is bool
        assert row['next_frame'] > row['frame'], 'Invalid transition interval'
        assert all(math.isfinite(row[k]) for k in ('reward', 'next_value'))
        assert math.isfinite(row['sample']['value'])
        last = i + 1 == len(rows)
        reason = ('native_terminal' if row['terminal'] else
                  'native_truncation' if row['truncated'] else
                  'dataset_end' if last else
                  'discontinuity' if not consecutive(row, rows[i + 1]) else None)
        if reason is None and i - start + 1 == max_steps:
            assert not row.get('bootstrap_zero'), 'Unexplained zero-bootstrap segment boundary'
            view[i] = dict(row, truncated=True)
            added += 1
            reason = 'scheduled_cut'
        if reason is not None:
            segments.append(dict(seed=row['seed'], start_index=rows[start]['index'],
                                 end_index=row['index'], start_frame=rows[start]['frame'],
                                 next_frame=row['next_frame'], rows=i - start + 1,
                                 reason=reason, next_value=row['next_value'],
                                 bootstrap_value=0. if row['terminal'] else row['next_value']))
            start = i + 1
    assert start == len(rows) and all(s['rows'] <= max_steps for s in segments)
    assert sum(s['rows'] for s in segments) == len(rows)
    return view, dict(version=VERSION, max_steps=max_steps, rows=len(rows),
                      added_cuts=added, segment_count=len(segments), segments=segments,
                      context_reset=False, raw_rows_modified=False,
                      scope='GAE recursion only; exact observed next-state value retained, native terminal/truncation semantics preserved.')

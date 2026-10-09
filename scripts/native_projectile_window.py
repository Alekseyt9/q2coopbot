"""Bind native events by log order to exact command windows, not frame lookup."""
import re

STEP=re.compile(r'^sv_test_step version=1 phase=(begin|end) spawncount=(-?\d+) frame=(\d+) seq=(\d+) actor=(\d+)$')
EVENT=re.compile(r'^sv_test_(?:projectile|damage) spawncount=(-?\d+) server_frame=(\d+) ')

def records(stream):
    pending=None;previous=None
    for raw in stream:
        line=raw.strip();match=STEP.match(line)
        if line.startswith('sv_test_step ') and not match:raise ValueError('Malformed native step')
        if match:
            phase,generation,frame,sequence,actor=match.groups();generation,frame,sequence,actor=map(int,(generation,frame,sequence,actor))
            if phase=='begin':
                assert pending is None and sequence>0 and actor>0
                if previous:assert generation==previous[0] and frame==previous[1]+1 and actor==previous[2] and sequence>previous[3]
                pending=(generation,frame,actor,sequence)
            else:
                assert pending is not None and pending==(generation,frame-1,actor,sequence)
                previous=pending;pending=None
        event=EVENT.match(line)
        if event and pending:
            generation,frame=map(int,event.groups());assert generation==pending[0] and pending[1]<=frame<=pending[1]+1
        yield pending,line
    assert pending is None,'Incomplete native step'

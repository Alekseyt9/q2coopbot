"""Metadata-only map splits for paired-teacher learnability diagnostics."""
import hashlib
from collections import Counter


def map_folds(records, count=4):
    assert type(count) is int and count >= 2
    maps = sorted({r['map'] for r in records},
                  key=lambda name: hashlib.sha256(('teacher-map-cv-v1:'+name).encode()).hexdigest())
    assert len(maps) >= count
    assignment = {name: i % count for i, name in enumerate(maps)}
    return [assignment[r['map']] for r in records]


def world_weights(records):
    counts = Counter((r['seed'], r['source_model_sha256']) for r in records)
    return [1.0/counts[(r['seed'], r['source_model_sha256'])] for r in records]

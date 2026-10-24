"""Verify new optional body observations against exact live client trace records."""
import argparse
import collections
import json
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    root, pins = args.corpus.resolve(), {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = pins.get(str(path)) or sha(path)
        assert expected is None or expected == digest, str(path)
        pins[str(path)] = digest
        return path

    pin(__file__)
    seal = read(pin(root/'complete.json'))
    assert seal['state'] == 'complete' and not seal['training_performed']
    pool = read(pin(root/'pool/report.json', seal['pool_sha256']))
    assert pool['source_unchanged'] and pool['usable_captures'] == 4
    counts, cases = collections.Counter(), []
    for binding in seal['processed']:
        assert binding['label'] == 'train'
        report_path = pin(binding['report'], binding['sha256'])
        report = read(report_path)
        assert report['device'] == 'cuda' and not report['training_performed']
        for case in report['cases']:
            assert case['split'] == 'train'
            directory = report_path.parent/f"case-{case['job']:04d}"
            meta = read(pin(directory/'cuda/report.json'))
            assert meta['numerical_verification'] == 'cuda_verified_v1' and meta['feature_version'] == 'combat_features_v10'
            replay = pin(directory/'cuda/rollout.jsonl', meta['rollout_sha256'])
            with replay.open(encoding='utf-8-sig') as stream:
                accepted = {(row['index'], row['frame']) for row in map(json.loads, stream) if len(row['features']) == 1147}
            assert len(accepted) == meta['rows']
            steps = directory/'native/replay-0/steps.jsonl'
            pin(steps, meta['source_sha256'][str(steps)])
            trace = next(Path(path) for path in meta['source_sha256'] if path.endswith('bot.jsonl'))
            pin(trace, meta['source_sha256'][str(trace)])
            raw = {}
            with trace.open(encoding='utf-8-sig') as stream:
                for row in map(json.loads, stream):
                    key = tuple(row.get(name) for name in ('spawncount', 'frame', 'self_entity', 'client_sequence'))
                    assert key not in raw
                    raw[key] = row
            counters = collections.Counter(cases=1)
            with steps.open(encoding='utf-8-sig') as stream:
                for step in map(json.loads, stream):
                    observation = step['observation']
                    if (step['index'], observation['identity']['frame']) not in accepted:
                        continue
                    counters['accepted_rows'] += 1
                    native = step['native_step']
                    key = tuple(native[name] for name in ('spawncount', 'begin_frame', 'actor', 'sequence'))
                    row = raw[key]
                    assert row['combat_policy']['observation']['identity'] == observation['identity']
                    bodies = observation.get('observed_dynamic_bodies')
                    if bodies is None:
                        counters['unavailable_frames'] += 1
                        continue
                    assert row['combat_policy']['observation']['observed_dynamic_bodies'] == bodies
                    assert bodies['version'] == 'observed_dynamic_bodies_v1' and bodies['frame'] == observation['identity']['frame']
                    assert bodies['range'] == 512 and bodies['capacity'] == 16 and len(bodies['bodies']) <= 16
                    assert bodies['bodies'] == sorted(bodies['bodies'], key=lambda body: (body['distance'], body['id']))
                    assert len({body['id'] for body in bodies['bodies']}) == len(bodies['bodies'])
                    client = {item['id']: item for item in row.get('collision_bodies', [])}
                    dead = {item['id']: item for item in row.get('defeated', [])}
                    counters['available_frames'] += 1
                    counters['frames_with_bodies'] += bool(bodies['bodies'])
                    for body in bodies['bodies']:
                        entity = body['id']
                        assert entity in client or entity in dead
                        assert entity != observation['identity']['actor'] and body['distance'] <= 512
                        assert body['observed_death_animation'] == (entity in dead)
                        source = client.get(entity, dead.get(entity))
                        assert body['protocol_solid'] == source.get('solid', 0)
                        if body['protocol_bounds_known']:
                            assert body['protocol_solid'] not in (0, 31)
                            assert 'protocol_mins' in body and 'protocol_maxs' in body
                        else:
                            assert 'protocol_mins' not in body and 'protocol_maxs' not in body
                        assert not any(key in body for key in ('health', 'health_before', 'native_bbox', 'server_frame'))
                        counters['body_records'] += 1
                        counters['death_animation_records'] += body['observed_death_animation']
                        counters['unknown_protocol_bounds_records'] += not body['protocol_bounds_known']
                        counters['unknown_death_bounds_records'] += body['observed_death_animation'] and not body['protocol_bounds_known']
            assert counters['accepted_rows'] == meta['rows']
            counts.update(counters)
            cases.append(dict(seed=case['seed'], counts=dict(counters)))
    assert counts['accepted_rows'] == 501 and counts['cases'] == 4
    assert counts['death_animation_records'] > 0 and counts['unknown_death_bounds_records'] > 0
    assert all(sha(path) == digest for path, digest in pins.items())
    args.out.mkdir(parents=True)
    save(args.out/'report.json', dict(state='complete', version='combat_dynamic_body_observation_live_v1',
        counts=dict(counts), cases=cases, inputs=pins, neural_execution=False, new_games=False,
        training_performed=False, promotion=False,
        scope='Four live registry scenes on base3/city1/jail1/ware2 and CUDA-verified legacy v10 rows. '
              'Only metadata provenance/unknown masks are audited here. Body fields are currently ignored by v10 '
              'features; this does not prove a body-aware neural policy or better combat.'))
    save(args.out/'complete.json', dict(state='complete', report_sha256=sha(args.out/'report.json')))


if __name__ == '__main__':
    main()

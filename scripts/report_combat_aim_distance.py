"""Distance strata of observed selected-target ray geometry; no model execution."""
import argparse
import collections
import json
import math
import pathlib

from process_combat_architecture_pool import read, save, sha
from report_combat_selected_target_aim import measure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    args = parser.parse_args()
    protocol_path = args.root / 'protocol.json'
    protocol = read(protocol_path)
    proof_path = args.root / 'recovery/verified-members.json'
    proof = read(proof_path)
    quality = read(args.root / 'quality-report.json')
    assert proof['state'] == quality['state'] == 'complete'
    assert proof['protocol_sha256'] == quality['protocol_sha256'] == sha(protocol_path)
    groups, sources = {}, []
    for entry in protocol['evaluations']:
        buckets = collections.defaultdict(collections.Counter)
        statuses = collections.Counter()
        for path in sorted(pathlib.Path(entry['root']).glob('case-*/s-*/report.json')):
            report = read(path)
            assert proof['members'][str(path.parent)]['report_sha256'] == sha(path)
            assert report['capture_complete'] and report['provenance_valid']
            assert len(report['results']) == 1
            result = report['results'][0]
            assert result['capture_valid'] and result['dispatch_valid']
            trace = pathlib.Path(result['root']) / 'bot.jsonl'
            with trace.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    capture = json.loads(line).get('combat_policy')
                    item = measure(capture) if capture else None
                    if item is None:
                        continue
                    statuses[item['status']] += 1
                    if item['status'] != 'measured':
                        continue
                    enemy = next(e for e in capture['observation']['enemies']
                                 if e['id'] == item['entity'] and
                                 (e.get('observed_track') or 0) == item['track'])
                    distance = enemy['distance']
                    assert math.isfinite(distance) and distance >= 0
                    band = 'near' if distance <= 128 else 'medium' if distance <= 512 else 'far'
                    for phase in ('all', 'firing') if item['firing'] else ('all',):
                        bucket = buckets[band + '-' + phase]
                        bucket['frames'] += 1
                        bucket['above_10_degrees'] += item['angular'] > 10
                        for key in ('yaw', 'pitch', 'angular'):
                            bucket[key + '_sum'] += item[key]
            sources.append(dict(trace=str(trace), trace_sha256=sha(trace), report_sha256=sha(path)))
        strata = {}
        for band in ('near', 'medium', 'far'):
            strata[band] = {}
            for phase in ('all', 'firing'):
                bucket = buckets[band + '-' + phase]
                count = bucket['frames']
                strata[band][phase] = dict(frames=count,
                    above_10_fraction=bucket['above_10_degrees'] / count if count else None,
                    **{key + '_mean_degrees': bucket[key + '_sum'] / count if count else None
                       for key in ('yaw', 'pitch', 'angular')})
        groups[entry['model'] + '-' + entry['label']] = dict(statuses=statuses, strata=strata)
    report = dict(version='combat_selected_target_distance_v1',
        protocol_sha256=sha(protocol_path), proof_sha256=sha(proof_path), groups=groups,
        distance_definition='Observed enemy-origin distance in world units; near [0,128], medium (128,512], far (512,infinity).',
        sources=sources,
        scope='First-life alive provider frames after frame100 with explicit observed selected target and known bbox. Applied ray geometry only, without lead/recoil/muzzle correction. Frames are not shots; angular proximity is not hit rate. Controls without declared target remain unmeasured. No model execution or training.')
    save(args.root / 'selected-target-distance.json', report)
    print(json.dumps(groups), flush=True)


if __name__ == '__main__':
    main()

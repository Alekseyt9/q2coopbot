"""Frame overlap of observed standing and blocked intents; no neural execution."""
import argparse
import collections
import json
import math
import pathlib

from audit_registered_execution_actions import read, sha


def classify(c):
    o, p, a = c['observation'], c['proposed'], c['applied']
    velocity = o['velocity']
    assert len(velocity) == 3 and all(math.isfinite(x) for x in velocity)
    assert isinstance(o['on_ground'], bool)
    proposed = max(abs(p['forward']), abs(p['side'])) >= .05 or p['vertical'] != 'release'
    suppressed = proposed and a['forward'] == a['side'] == 0 and a['vertical'] == 'release'
    clear = any(e.get('clear_shot') for e in o.get('enemies') or [])
    standing = math.hypot(*velocity[:2]) < 10
    labels = {i['reason'] for i in c['selection'].get('interventions', [])
              if i['component'] in ('movement', 'vertical')}
    result = collections.Counter(provider_frames=1, suppressed_frames=int(suppressed),
        standing_clear_frames=int(standing and clear),
        standing_clear_suppressed_frames=int(standing and clear and suppressed),
        standing_clear_no_movement_intent_frames=int(standing and clear and not proposed),
        standing_clear_applied_movement_frames=int(standing and clear and proposed and not suppressed),
        moving_suppressed_frames=int(not standing and suppressed))
    if suppressed and 'unsupported_motion_guard' in labels:
        result['unsupported_frames'] += 1
        result['unsupported_airborne_frames' if not o['on_ground'] else
               'unsupported_grounded_vertical_velocity_frames' if abs(velocity[2]) > 1 else
               'unsupported_grounded_other_frames'] += 1
    reasons = collections.Counter({label: 1 for label in labels} if standing and clear and suppressed else {})
    return result, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evaluation', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    root = args.evaluation.resolve()
    seal_path = root/'complete.json'
    seal = read(seal_path)
    assert seal['state'] == 'complete' and not seal['training_performed']
    pool_path = root/'pool/report.json'
    assert sha(pool_path) == seal['pool_sha256']
    jobs = {j['job']: j for j in read(pool_path)['jobs']}
    pins = {str(seal_path): sha(seal_path), str(pool_path): sha(pool_path)}
    variants = {}
    # The directory also contains the aggregated execution-limits report.
    # Select cases from the sealed variant specification, never a filename glob.
    spec = read(seal['spec']) if 'spec' in seal else None
    if spec is None:
        candidates = [p for p in root.glob('execution-*.json') if p.name != 'execution-limits.json']
    else:
        candidates = [root/('execution-'+v['label']+'.json') for v in spec['variants'] if v['mode'] == 'learned']
    for audit_path in sorted(candidates):
        source = read(audit_path)
        assert source['state'] == 'complete' and source['pool_sha256'] == seal['pool_sha256']
        pins[str(audit_path)] = sha(audit_path)
        cases = []
        for case in source['cases']:
            job = jobs[case['job']]
            capture_path = pathlib.Path(job['root'])/'report.json'
            assert sha(capture_path) == case['capture_sha256']
            trace = pathlib.Path(read(capture_path)['results'][0]['root'])/'bot.jsonl'
            assert sha(trace) == case['trace_sha256']
            counts, reasons, seen = collections.Counter(), collections.Counter(), set()
            with trace.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    c = json.loads(line).get('combat_policy') or {}
                    if (c.get('selection') or {}).get('owner') != 'provider':
                        continue
                    o = c['observation']; identity = o['identity']
                    if identity['life'] != 1 or o['health'] <= 0 or identity['frame'] < case['network_release_frame']:
                        continue
                    assert identity['spawncount'] == case['release_generation']
                    key = tuple(identity[k] for k in ('map','spawncount','connection','actor','life','frame'))
                    assert key not in seen; seen.add(key)
                    for action in (c['proposed'], c['applied']):
                        assert tuple(action['identity'][k] for k in ('map','spawncount','connection','actor','life','frame')) == key
                    n, r = classify(c); counts.update(n); reasons.update(r)
            assert counts['provider_frames'] == case['counts']['provider_frames']
            assert counts['suppressed_frames'] == case['counts']['movement_command_fully_suppressed_frames']
            assert sha(trace) == case['trace_sha256']
            cases.append(dict(job=case['job'], episode=case['episode'], seed=case['seed'],
                trace_sha256=case['trace_sha256'], counts=dict(counts), standing_suppression_reasons=dict(reasons)))
        groups = {}
        for group in ('all', 'ordinary', 'recovery'):
            selected = [c for c in cases if group == 'all' or c['episode'].endswith('-recovery') == (group == 'recovery')]
            groups[group] = dict(cases=len(selected), counts=dict(sum((collections.Counter(c['counts']) for c in selected), collections.Counter())),
                standing_suppression_reasons=dict(sum((collections.Counter(c['standing_suppression_reasons']) for c in selected), collections.Counter())))
        label = audit_path.stem.removeprefix('execution-')
        variants[label] = dict(cases=cases, groups=groups)
        print(label, groups['all'], flush=True)
    assert len(variants) == 12
    for path, digest in pins.items(): assert sha(path) == digest, path
    args.out.write_text(json.dumps(dict(state='complete', version='movement_overlap_v1', variants=variants,
        inputs=pins, script_sha256=sha(__file__), training_performed=False, neural_execution=False,
        scope='Observed pre-command XY speed below10; exact frame overlap with suppressed intents. Overlap is not causal proof: blocked acceleration retains momentum, applied movement can collide. Unsupported grounded-other requires further command/physics diagnosis.'), indent=2), encoding='utf-8')


if __name__ == '__main__': main()

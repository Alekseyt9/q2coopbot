"""Observed stationary Shotgun geometry labels; no neural execution or hidden state."""
import argparse
import math
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha
from audit_combat_contact_loss_probe import load_steps


def label(step):
    o, n, a = step['observation'], step['next_observation'], step['action']
    sample = step.get('sample', {})
    if '/v_shotg/' not in o['weapon'] and o['weapon'] != 'Shotgun':
        return None
    if sample.get('aim_mode', 0) != 1 or not sample.get('target', 0):
        return None
    if not o['on_ground'] or not n['on_ground'] or o['ducked'] != n['ducked']:
        return None
    if math.sqrt(sum(v*v for v in o['velocity'])) > 10:
        return None
    if max(abs(v-w) for v, w in zip(o['position'], n['position'])) > .125:
        return None
    target = next((e for e in o['enemies'] if e['id'] == a.get('target_entity') and e.get('clear_shot')), None)
    if target is None or target.get('observed_solid', 0) in (0, 31):
        return None
    solid = target['observed_solid']
    radius, bottom, top = (solid & 31)*8, -((solid >> 5) & 31)*8, ((solid >> 10) & 63)*8-32
    if radius <= 0 or top <= bottom:
        return None
    dx, dy, dz = target['relative']
    # Native CENTER_HANDED Shotgun: no lateral offset, viewheight minus eight;
    # unlike Machinegun its firing ray does not add visual kick angles.
    dz += (bottom+top)/2 - ((-2 if o['ducked'] else 22)-8)
    yaw, pitch = math.atan2(dy, dx)*180/math.pi, -math.atan2(dz, math.hypot(dx, dy))*180/math.pi
    current = [v*360/65536 for v in o['view_angles'][:2]]
    wrap = lambda v: (v+180) % 360-180
    delta = [wrap(yaw-current[1]), wrap(pitch-current[0])]
    return dict(frame=o['identity']['frame'], target_entity=target['id'],
                delta_degrees=[max(-13.5, min(13.5, v)) for v in delta],
                unclipped_delta_degrees=delta, relative_height=target['relative'][2],
                horizontal_distance=math.hypot(dx, dy), ducked=o['ducked'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--request', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    assert not a.out.exists()
    request = read(a.request)
    assert request['version'] == 'combat_stationary_shotgun_aim_request_v1'
    entries, pins, worlds = [], {str(a.request.resolve()): sha(a.request)}, {'train': set(), 'diagnostic': set()}
    for entry in request['episodes']:
        root = Path(entry['screen'])
        spec, pool = read(root/'spec.json'), read(root/'pool/report.json')
        assert pool['state'] == 'complete' and pool['source_unchanged']
        v = next(v for v in spec['variants'] if v['index'] == entry['index'])
        assert v['mode'] == 'learned'
        split = entry['split']
        assert split in worlds
        key = (v['context']['map'], v['context']['seed'])
        assert key not in worlds[split], 'Duplicate world in split'
        worlds[split].add(key)
        case = root/f"case-{entry['index']:02d}"
        meta, proof = read(case/'cuda/report.json'), read(case/'cuda/cuda-verification.json')
        assert proof['state'] == 'passed' and proof['device'] == 'cuda'
        assert meta['model_sha256'] == v['model_sha256'] == sha(v['model'])
        assert sha(case/'cuda/cuda-verification.json') == meta['cuda_verification_sha256']
        native = next((case/'native').glob('replay-*/steps.jsonl'))
        steps = load_steps(native)
        rollout = load_steps(case/'cuda/rollout.jsonl')
        by_frame = {s['observation']['identity']['frame']: s for s in steps}
        assert len(by_frame) == len(steps) and len(rollout) == meta['rows']
        labels = []
        for i, row in enumerate(rollout):
            s = by_frame[row['frame']]
            assert s['owner'] == 'provider' and s['sample'] == row['sample']
            if (y := label(s)) is not None:
                labels.append(dict(index=i, **y))
        entries.append(dict(split=split, map=key[0], seed=key[1], source_cuda=str((case/'cuda').resolve()),
                            native=str(native.resolve()), native_sha256=sha(native), labels=labels,
                            source_model_sha256=v['model_sha256'], rows=len(rollout)))
        for p in [root/'spec.json', root/'pool/report.json', case/'cuda/report.json', case/'cuda/cuda-verification.json', native]:
            pins[str(p.resolve())] = sha(p)
    assert worlds['train'] and worlds['diagnostic'] and not worlds['train'] & worlds['diagnostic']
    counts = {s: dict(worlds=len(worlds[s]), labels=sum(len(e['labels']) for e in entries if e['split'] == s),
                      worlds_with_labels=sum(bool(e['labels']) for e in entries if e['split'] == s)) for s in worlds}
    assert counts['train']['labels'] > 0
    a.out.mkdir(parents=True)
    save(a.out/'report.json', dict(version='combat_stationary_shotgun_aim_data_v1', state='passed',
        episodes=entries, counts=counts, inputs=pins, neural_execution=False, ppo_eligible=False,
        executed_demonstration=False, scope='Snapshot bbox-center geometry, stationary actual pose; no pellet/spread/optimal action claim. Future pose used only to filter unchanged stationary executions, never a policy input.'))
    print(counts, flush=True)


if __name__ == '__main__':
    main()

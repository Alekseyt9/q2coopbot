"""Explicit post-hoc verification; preserve the rejected original experiment."""
import argparse
import copy
import json
from pathlib import Path

from combat_teacher_native_receipt import admitted_terminal_receipt
from combat_teacher_movement_receipt import admitted_terminal_receipt_v2
from finish_combat_anchor_probe import first_diff
from process_combat_architecture_pool import read, save, sha
from recover_combat_anchor_screen import first_life, terminal_receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--terminal-version', type=int, choices=(1,2), default=1)
    a = ap.parse_args()
    source, out = a.source.resolve(), a.out.resolve()
    repo = Path(__file__).resolve().parents[1]
    assert not out.exists()
    assert read(source/'progress.json')['stage'] == 'failed'
    allowed_errors = {'Matching clamp altered terminal'}
    if a.terminal_version == 2:
        allowed_errors.add('Matching clamp altered admitted terminal')
    assert read(source/'progress.json')['error'] in allowed_errors
    receipt_function = admitted_terminal_receipt if a.terminal_version == 1 else admitted_terminal_receipt_v2
    assert not (source/'complete.json').exists()
    protocol = read(source/'protocol.json')
    pool = read(source/'pool/report.json')
    request = read(source/'cuda-request.json')
    assert pool['state'] == 'complete' and pool['source_unchanged']
    assert pool['usable_captures'] == len(pool['jobs']) == protocol['native_cases']
    assert read(source/'cuda-batch/report.json')['state'] == 'complete'
    pins = copy.deepcopy(protocol['inputs'])
    for path, digest in pins.items():
        assert sha(path) == digest, path
    for path in (Path(__file__), repo/'scripts/combat_teacher_native_receipt.py', repo/'scripts/combat_teacher_movement_receipt.py',
                 source/'progress.json', source/'protocol.json', source/'pool/report.json',
                 source/'cuda-request.json', source/'cuda-batch/report.json'):
        pins[str(path.resolve())] = sha(path)
    out.mkdir(parents=True)
    corrected = copy.deepcopy(protocol)
    corrected.update(original_protocol_passed=False, posthoc_scope_correction=True,
        original_rejected_root=str(source),
        terminal_receipt_version='combat_teacher_admitted_terminal_v'+str(a.terminal_version),
        terminal_scope='All native terminal fields remain exact. Monitor aim_by_class (v1/v2) and movement (v2) are reconciled by independently reconstructing FULL bot.jsonl counts and subtracting frames before the first server-dispatched dataset step. Admitted counters and actual observations/commands are never dropped or reordered.',
        inputs=pins)
    save(out/'protocol.json', corrected)
    save(out/'progress.json', dict(stage='verifying'))
    pairs, exports = {}, []
    try:
        for i, job in enumerate(sorted(pool['jobs'], key=lambda j: j['plan_index'])):
            assert not job['error']
            assignment = protocol['assignments'][job['plan_index']]
            assert assignment['seed'] == job['seed'] and assignment['plan_index'] == job['plan_index']
            report = Path(job['root'])/'report.json'
            capture = read(report)
            assert capture['capture_complete'] and capture['provenance_valid']
            cap = capture['results'][0]
            assert cap['capture_valid'] and cap['dispatch_valid'] and cap['seed_confirmed']
            assert cap['generated_fixture']['split'] == 'train'
            path = Path(cap['root'])/'dataset/steps.jsonl'
            with path.open(encoding='utf-8-sig') as stream:
                rows = first_life(cap, [json.loads(line) for line in stream])
            pairs.setdefault(job['seed'], {})[assignment['mode']] = (cap, rows)
            directory = source/f'export-{i:02d}-cuda'
            task = request['tasks'][i]
            assert Path(task['out']).resolve() == directory
            assert task['model_sha256'] == assignment['model_sha256']
            meta, proof = read(directory/'report.json'), read(directory/'cuda-verification.json')
            assert proof['state'] == 'passed' and proof['device'] == 'cuda'
            assert proof['model_sha256'] == assignment['model_sha256']
            assert sha(directory/'rollout.jsonl') == meta['rollout_sha256']
            assert sha(directory/'cuda-verification.json') == meta['cuda_verification_sha256']
            if assignment['mode'] != 'baseline':
                assert proof['wide_gate_clamp_and_final_fire_cdf_verified']
                assert meta['diagnostic_only'] and not meta['ordinary_training_eligible']
            for path in (report, path, Path(cap['root'])/'bot.jsonl', directory/'report.json',
                         directory/'cuda-verification.json', directory/'rollout.jsonl'):
                pins[str(path)] = sha(path)
            exports.append(dict(**assignment, data=str(directory)))

        results = []
        for ctx in protocol['selected']:
            seed, frame = ctx['seed'], ctx['frame']
            modes = pairs[seed]
            bc, bb = modes['baseline']
            projected = lambda rows: [dict(observation=s['observation'], sent_command=s['sent_command'], owner=s['owner']) for s in rows]
            prefix = lambda rows: [s for s in projected(rows) if s['observation']['identity']['frame'] < frame]
            baseline_root = next(s for s in bb if s['observation']['identity']['frame'] == frame)
            choice = baseline_root['sample']['wide_aim']['choice']
            outcomes, reconciliation = {}, {}
            original_diff = None
            for mode, (cap, rows) in modes.items():
                assert cap['generated_fixture'] == bc['generated_fixture']
                assert first_diff(prefix(bb), prefix(rows)) is None, (seed, mode, 'prefix')
                current = next(s for s in rows if s['observation']['identity']['frame'] == frame)
                assert first_diff(current['observation'], baseline_root['observation']) is None
                if mode != 'baseline':
                    assert current['sample']['wide_aim']['choice'] == int(mode[-1])
                    if int(mode[-1]) == choice:
                        assert first_diff(projected(bb), projected(rows)) is None
                        first_frame = bb[0]['observation']['identity']['frame']
                        assert rows[0]['observation']['identity']['frame'] == first_frame
                        left, left_proof = receipt_function(bc, first_frame)
                        right, right_proof = receipt_function(cap, first_frame)
                        assert first_diff(left, right) is None, (seed, 'admitted terminal')
                        original_diff = first_diff(terminal_receipt(bc), terminal_receipt(cap))
                        reconciliation = dict(baseline=left_proof, matching_clamp=right_proof)
                damage = cap['first_life']['damage']
                end = cap['goal_stop']['kill_frame'] if cap['goal_stop'] else cap['first_life']['end_frame']
                outcomes[mode] = dict(goal=bool(cap['goal_stop']),
                    death=bool(cap['death_stop'] or cap['first_life']['minimum_health'] <= 0),
                    kills=sum(v['monster_kills'] for v in damage['by_mod']),
                    dealt=sum(v['monster_health_damage'] for v in damage['by_mod']),
                    received=damage['received_health_damage'],
                    rules_combat_rows=sum(s['owner'] == 'rules' for s in rows if s['observation']['identity']['frame'] <= end),
                    root_choice=current['sample']['wide_aim']['choice'], root_attack=current['sample']['attack'],
                    root_sent=current['sent_command'])
            results.append(dict(seed=seed, map=bc['first_life']['map'], frame=frame,
                prefix_exact=True, current_observation_exact=True, matching_clamp_first_life_exact=True,
                matching_clamp_admitted_terminal_exact=True, original_terminal_difference=original_diff,
                monitor_aim_reconciliation=reconciliation, outcomes=outcomes))
        for path, digest in pins.items():
            assert sha(path) == digest, path
        corrected['inputs'] = pins
        save(out/'protocol.json', corrected)
        save(out/'exports.json', dict(entries=exports, diagnostic_only=True, ordinary_training_eligible=False,
                                      scope='All original branches retained, post-hoc monitor warmup scope correction, no policy fitting.'))
        save(out/'results.json', dict(state='complete', cases=results, diagnostic_only=True,
            original_protocol_passed=False, posthoc_scope_correction=True, training_performed=False, promotion=False))
        save(out/'complete.json', dict(state='complete', native_cases=len(exports), cuda_cases=len(exports),
            results_sha256=sha(out/'results.json'), exports_sha256=sha(out/'exports.json'),
            original_protocol_passed=False, posthoc_scope_correction=True, diagnostic_only=True,
            training_performed=False, promotion=False))
        save(out/'progress.json', dict(stage='complete'))
        print(out/'complete.json')
    except BaseException as e:
        save(out/'progress.json', dict(stage='failed', error=str(e)))
        raise


if __name__ == '__main__':
    main()

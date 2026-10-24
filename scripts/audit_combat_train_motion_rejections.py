"""Describe rejected motion by grounded/vertical/pose state on sealed TRAIN only; no safety or causal claims."""
import argparse
import collections
import json
import math
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    pins, cases, groups = {}, [], collections.defaultdict(collections.Counter)

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        assert str(path) not in pins or pins[str(path)] == digest
        pins[str(path)] = digest
        return path

    pin(__file__)
    for root in args.corpus:
        root = root.resolve()
        closed = read(pin(root / 'complete.json'))
        protocol = read(pin(root / 'protocol.json', closed['protocol_sha256']))
        assert protocol['train_only'] and len(protocol['assignments']) == 1
        assignment = protocol['assignments'][0]
        assert assignment['split'] == 'train'
        plan = read(pin(assignment['plan'], assignment['plan_sha256']))
        assert all(task['split'] == 'train' for task in plan['tasks'])
        recipes = {seed: task['episode']['id'] for task in plan['tasks'] for seed in task['seeds']}
        processed = closed['processed'][0]
        metadata = read(pin(processed['report'], processed['sha256']))
        assert metadata['device'] == 'cuda' and not metadata['training_performed']
        merged = read(pin(root / 'processing-train/merged/report.json'))
        corpus_rows = 0
        for member in merged['members']:
            directory = Path(member['directory'])
            meta = read(pin(directory / 'report.json'))
            rollout = pin(directory / 'rollout.jsonl', member['rollout_sha256'])
            selected = {}
            with rollout.open(encoding='utf-8-sig') as stream:
                for row in map(json.loads, stream):
                    assert row['index'] not in selected
                    selected[row['index']] = row['seed']
            assert len(selected) == member['rows'] == meta['rows']
            step_path = next(directory.parent.glob('native/replay-*/steps.jsonl'))
            pin(step_path, meta['source_sha256'][str(step_path.resolve())])
            counts, zero_reasons, reasons = collections.Counter(), collections.Counter(), collections.Counter()
            examples, found, profiles = [], set(), collections.Counter()
            with step_path.open(encoding='utf-8-sig') as stream:
                for step in map(json.loads, stream):
                    if step['index'] not in selected:
                        continue
                    assert step['index'] not in found
                    found.add(step['index'])
                    obs, nxt = step['observation'], step.get('next_observation')
                    assert step['owner'] == 'provider' and step['sample']
                    action, applied = step['action'], step['applied_action']
                    rn = math.hypot(action['forward'], action['side'])
                    an = math.hypot(applied['forward'], applied['side'])
                    reduced = rn >= .25 and an < .1
                    interventions = sorted({v['reason'] for v in step.get('interventions') or []
                                            if v['component'] == 'movement'})
                    reasons.update(interventions)
                    if reduced:
                        zero_reasons.update(interventions or ['unexplained'])
                    if reduced:
                        posture = 'grounded' if obs['on_ground'] else 'airborne'
                        vertical_state = 'vertical_gt1' if abs(obs['velocity'][2]) > 1 else 'flat_vertical'
                        pose_state = 'ducked' if obs['ducked'] else 'standing'
                        for reason in interventions or ['unexplained']:
                            profiles['/'.join((reason,posture,vertical_state,pose_state))] += 1
                    counts['grounded_rows'] += bool(obs['on_ground'])
                    counts['grounded_vertical_gt1_rows'] += bool(obs['on_ground'] and abs(obs['velocity'][2]) > 1)
                    counts['unsupported_grounded_rows'] += bool(reduced and obs['on_ground'] and 'unsupported_motion_guard' in interventions)
                    counts['unsupported_grounded_vertical_gt1_rows'] += bool(reduced and obs['on_ground'] and abs(obs['velocity'][2]) > 1 and 'unsupported_motion_guard' in interventions)
                    counts['rows'] += 1
                    counts['large_request'] += rn >= .25
                    counts['reduced_near_zero'] += reduced
                    counts['reduced_on_ground'] += reduced and obs['on_ground']
                    counts['reduced_airborne'] += reduced and not obs['on_ground']
                    counts['large_applied'] += an >= .25
                    counts['jump_requested'] += action['vertical'] == 'jump'
                    counts['jump_applied'] += applied['vertical'] == 'jump'
                    if nxt:
                        identity, following = obs['identity'], nxt['identity']
                        adjacent = following['frame'] == identity['frame'] + 1 and all(
                            identity[k] == following[k] for k in ('map', 'connection', 'spawncount', 'actor', 'life'))
                        if adjacent:
                            counts['adjacent_rows'] += 1
                            travel = math.dist(obs['position'][:2], nxt['position'][:2])
                            counts['large_applied_low_travel'] += an >= .25 and travel < 1
                    no_target_attack = applied['attack'] and not action.get('target_entity')
                    counts['attack_without_target'] += no_target_attack
                    counts['attack_without_target_visible'] += no_target_attack and bool(obs['enemies'])
                    counts['attack_without_target_no_visible'] += no_target_attack and not obs['enemies']
                    counts['attack_without_target_empty_ammo'] += no_target_attack and obs['ammo'] <= 0
                    if reduced and len(examples) < 12:
                        examples.append(dict(index=step['index'], frame=obs['identity']['frame'],
                            on_ground=obs['on_ground'], reasons=interventions, position=obs['position'], velocity=obs['velocity'],
                            ducked=obs['ducked'], requested_vertical=action['vertical'], applied_vertical=applied['vertical'],
                            observed_enemies=[dict(id=e['id'],kind=e['class'],distance=e['distance']) for e in obs['enemies']],
                            request=[action['forward'], action['side']],
                            applied=[applied['forward'], applied['side']]))
            assert found == set(selected)
            seed_set = set(selected.values())
            assert len(seed_set) == 1
            seed = seed_set.pop()
            condition = 'recovery' if recipes[seed].endswith('-recovery') else 'ordinary'
            for group in ('all', condition):
                groups[str(root), group].update(counts)
            cases.append(dict(corpus=str(root), seed=seed, episode=recipes[seed], condition=condition,
                counts=dict(counts), movement_interventions=dict(reasons),
                reduced_movement_reasons=dict(zero_reasons), rejection_profiles=dict(profiles), examples=examples))
            corpus_rows += counts['rows']
        assert corpus_rows == processed['rows']
    assert all(sha(path) == digest for path, digest in pins.items())
    args.out.mkdir(parents=True)
    rows = [dict(corpus=root, condition=condition, counts=dict(counts))
            for (root, condition), counts in sorted(groups.items())]
    reduced_reasons = collections.Counter()
    for case in cases:
        reduced_reasons.update(case['reduced_movement_reasons'])
    save(args.out / 'report.json', dict(state='complete', rows=rows, cases=cases,
        reduced_movement_reasons=dict(reduced_reasons), inputs=pins,
        neural_execution=False, new_games=False, training_performed=False,
        scope='Profiles are observed conditions, not proof that a rejected move was safe or that a guard caused defeat. Recorded TRAIN action-to-command attribution; next-state travel only for adjacent same-life frames. '
              'Intervention labels explain recorded reductions, not whether guards are physically necessary. '
              'Attack requests do not establish actual shots or misses. Different corpora are not paired outcomes.'))
    save(args.out / 'complete.json', dict(state='complete', report_sha256=sha(args.out / 'report.json')))
    print(json.dumps(dict(rows=rows, reduced_movement_reasons=dict(reduced_reasons))))


if __name__ == '__main__':
    main()

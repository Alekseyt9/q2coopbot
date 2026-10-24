"""TRAIN-only pre-impact grenade observations; no neural execution or action ranking."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha


LEADS = (1, 3, 5, 10)
IDENTITY = ('map', 'spawncount', 'connection', 'life', 'actor')


def context_at(history, event, lead, reference):
    """Require an exact earlier NETWORK frame and the same observation identity."""
    assert lead in LEADS and lead > 0
    step = history.get(event['frame']-lead)
    if step is None:
        return None, 'missing_frame'
    identity = step['observation']['identity']
    if any(identity[key] != reference[key] for key in IDENTITY):
        return None, 'different_identity'
    assert identity['frame'] < event['frame']
    return step, None


def describe(step, event):
    observation = step['observation']
    projectiles = sorted(observation.get('visible_projectiles') or [], key=lambda p: p['distance'])
    assert len({p['id'] for p in projectiles}) == len(projectiles)
    candidate = next((p for p in projectiles if p['id'] == event['inflictor'] and p['class'] == 'grenade'), None)
    slot = next((i for i, p in enumerate(projectiles[:4]) if p is candidate), None) if candidate else None
    requested, applied = step['action'], step['applied_action']
    request_norm = math.hypot(requested['forward'], requested['side'])
    applied_norm = math.hypot(applied['forward'], applied['side'])
    attacker = next((e for e in observation.get('enemies') or [] if e['id'] == event['attacker']), None)
    return dict(index=step['index'], frame=observation['identity']['frame'],
        any_grenade_visible=any(p['class'] == 'grenade' for p in projectiles),
        any_grenade_in_slots=any(p['class'] == 'grenade' for p in projectiles[:4]),
        same_id_candidate_visible=candidate is not None, same_id_candidate_in_slots=slot is not None,
        same_id_candidate_slot=slot,
        same_id_candidate_velocity_known=candidate is not None and candidate.get('observed_velocity') is not None,
        same_id_candidate=dict(candidate) if candidate is not None else None,
        lifetime_identity_proven=False, nearest_four_classes=[p['class'] for p in projectiles[:4]],
        horizontal_speed=math.hypot(*observation['velocity'][:2]), health=observation['health'],
        on_ground=observation['on_ground'], request_norm=request_norm, applied_norm=applied_norm,
        request_reduced_near_zero=request_norm >= .25 and applied_norm < .1,
        movement_interventions=sorted({i['reason'] for i in step.get('interventions') or [] if i['component'] == 'movement'}),
        attacker_visible=attacker is not None, attacker_distance=attacker['distance'] if attacker else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('workflow', 'risk', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    a = ap.parse_args()
    assert not a.out.exists()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = pins.get(str(path)) or sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    def lines(path):
        with path.open(encoding='utf-8-sig') as stream:
            return [json.loads(line) for line in stream]

    pin(__file__)
    pin(Path(__file__).with_name('combat_projectile_type_features.py'))
    seal = read(pin(a.risk/'complete.json'))
    risk = read(pin(a.risk/'report.json', seal['report_sha256']))
    assert seal['state'] == risk['state'] == 'complete' and risk['version'] == 'combat_train_incoming_risk_v2'
    assert not risk['neural_execution'] and not risk['training_performed'] and not risk['new_games']
    assert risk['arm'] in ('typed', 'neutral') and len(risk['cases']) == 96
    for path, digest in risk['inputs'].items():
        pin(path, digest)
    workflow = read(pin(a.workflow/'complete.json'))
    assert workflow['state'] == 'complete' and workflow['source_unchanged']
    assert workflow['native_train_cases'] == workflow['cuda_train_cases'] == 192
    expected = Counter()
    for mod in (6, 7):
        source = risk['incoming_sources'].get('class:monster_gunner/mod:'+str(mod), {})
        expected.update({key: source.get(key, 0) for key in ('events', 'received_health_damage', 'lethal_events')})
    groups, reasons, events, seen = defaultdict(Counter), defaultdict(Counter), [], set()
    for case in risk['cases']:
        branch = a.workflow/f"round-{case['round']:02d}"/risk['arm']/'corpus'
        receipt = read(pin(branch/'complete.json'))
        protocol = read(pin(branch/'protocol.json', receipt['protocol_sha256']))
        assert protocol['train_only'] and all(x['split'] == 'train' for x in protocol['assignments'])
        folder = branch/'processing-train'/f"case-{case['job']:04d}"
        meta = read(pin(folder/'cuda/report.json'))
        assert meta['numerical_verification'] == 'cuda_verified_v1'
        accepted = lines(pin(folder/'cuda/rollout.jsonl', meta['rollout_sha256']))
        paths = list((folder/'native').glob('replay-*/steps.jsonl'))
        assert len(paths) == 1
        step_path = paths[0].resolve()
        effect_path = step_path.with_name('server_outcomes.jsonl')
        steps = {s['index']: s for s in lines(pin(step_path, meta['source_sha256'][str(step_path)]))}
        effects = {r['step']: r for r in lines(pin(effect_path, meta['source_sha256'][str(effect_path)]))}
        history = {}
        for row in accepted:
            step = steps[row['index']]
            identity = step['observation']['identity']
            assert row['seed'] == case['seed'] and row['frame'] == identity['frame']
            assert identity['life'] == 1 and step['owner'] == 'provider' and step['native_step']
            assert effects[row['index']]['available']
            # Preserve the latest accepted execution if two commands share a network frame.
            previous = history.get(identity['frame'])
            assert previous is None or previous['index'] < step['index']
            history[identity['frame']] = step
        for row in accepted:
            step, effect = steps[row['index']], effects[row['index']]
            identity = step['observation']['identity']
            for ordinal, event in enumerate(effect['events']):
                if not (event['target'] == identity['actor'] and event['attacker_class'] == 'monster_gunner'
                        and event['mod'] in (6, 7)):
                    continue
                damage = max(event['health_before'], 0)-max(event['health_after'], 0)
                if damage <= 0:
                    continue
                assert (event['map'], event['spawncount']) == (identity['map'], identity['spawncount'])
                window = step['native_step']
                assert window['begin_frame'] <= event['frame'] <= window['end_frame']
                token = (case['round'], case['job'], row['index'], ordinal)
                assert token not in seen
                seen.add(token)
                lethal = event['health_before'] > 0 >= event['health_after']
                groups['events'].update(events=1, received_health_damage=damage, lethal_events=int(lethal))
                contexts = {}
                for lead in LEADS:
                    before, missing = context_at(history, event, lead, identity)
                    context = describe(before, event) if before else None
                    contexts[str(lead)] = dict(context=context, missing_reason=missing)
                    keys = [f'all/lead:{lead}', f'mod:{event["mod"]}/lead:{lead}', f'loadout:{case["loadout"]}/lead:{lead}']
                    if lethal:
                        keys.append(f'lethal/lead:{lead}')
                    for key in keys:
                        counts = groups[key]
                        counts['events'] += 1
                        if context is None:
                            counts['missing_'+missing] += 1
                            continue
                        counts['available'] += 1
                        for field in ('any_grenade_visible', 'any_grenade_in_slots', 'same_id_candidate_visible',
                                'same_id_candidate_in_slots', 'same_id_candidate_velocity_known',
                                'request_reduced_near_zero', 'attacker_visible'):
                            counts[field] += int(context[field])
                        counts['same_id_candidate_outside_slots'] += int(context['same_id_candidate_visible'] and not context['same_id_candidate_in_slots'])
                        counts['speed_le10'] += int(context['horizontal_speed'] <= 10)
                        counts['request_nonzero'] += int(context['request_norm'] >= .25)
                        counts['applied_nonzero'] += int(context['applied_norm'] >= .25)
                        reasons[key].update(context['movement_interventions'])
                events.append(dict(round=case['round'], job=case['job'], seed=case['seed'], episode=case['episode'],
                    loadout=case['loadout'], monster_count=case['monster_count'], index=row['index'], event_ordinal=ordinal,
                    event=event, received_health_damage=damage, lethal=lethal, contexts=contexts))
    assert groups['events'] == expected, (groups['events'], expected)
    for lead in LEADS:
        assert groups[f'all/lead:{lead}']['events'] == len(events)
        assert groups[f'lethal/lead:{lead}']['events'] == expected['lethal_events']
    assert all(sha(p) == h for p, h in pins.items())
    a.out.mkdir(parents=True)
    report = dict(state='complete', version='combat_grenade_preimpact_v1', arm=risk['arm'], cases=96,
        groups={k: dict(v) for k, v in sorted(groups.items())},
        movement_interventions={k: dict(v) for k, v in sorted(reasons.items())}, events=events, inputs=pins,
        neural_execution=False, new_games=False, training_performed=False, lifetime_identity_proven=False,
        scope='Accepted own-policy TRAIN observations at exact NETWORK frames 1/3/5/10 before gunner grenade HP events. '
              'Missing frames are not imputed. Native future inflictor/attacker/mod are diagnostic labels only. '
              'Same-ID grenade candidates do not prove lifetime identity or rule out entity reuse. '
              'Visibility, velocity availability and movement describe observations/actions, not causal survivability or action ranking. '
              'No held-out cases, privileged policy features, teacher labels or fitting.')
    save(a.out/'report.json', report)
    save(a.out/'complete.json', dict(state='complete', report_sha256=sha(a.out/'report.json')))
    print(dict(arm=risk['arm'], groups=report['groups']), flush=True)


if __name__ == '__main__':
    main()

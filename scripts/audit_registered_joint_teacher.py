"""Audit raw teacher target/history coverage before any joint BC label export.

No neural execution, no training labels, and no future-state reconstruction.
Go remains authoritative for feature construction and action availability.
"""
import argparse
import collections
import json
import pathlib

from process_combat_architecture_pool import read, save, sha


IDENTITY = ('map', 'connection', 'spawncount', 'actor', 'life')


def target_choice(policy):
    """Resolve only the explicit enemy teacher; never invent a nearest target."""
    obs = policy['observation']
    entity = policy.get('teacher_aim_entity')
    if policy.get('teacher_aim_source') != 'enemy' or not entity:
        return None, 'non_enemy_aim'
    # Python's stable ordering matches policy.TargetEnemies, including ties.
    enemies = sorted(obs.get('enemies') or [], key=lambda e: e['distance'])[:8]
    for slot, enemy in enumerate(enemies, 1):
        if enemy['id'] == entity:
            if not enemy.get('clear_shot'):
                return None, 'enemy_not_clear'
            if enemy.get('observed_track') is None:
                return None, 'enemy_track_unknown'
            return dict(identity=obs['identity'], entity=entity,
                        track=enemy['observed_track'], slot=slot), 'explicit_visible_target'
    return None, 'enemy_outside_observed_slots'


def previous_intent(previous, obs):
    """Equivalent one-frame identity gate; input is the PAST teacher choice."""
    if previous is None:
        return None
    before, now = previous['identity'], obs['identity']
    if any(before[k] != now[k] for k in IDENTITY) or before['frame'] + 1 != now['frame']:
        return None
    return {k: previous[k] for k in ('identity', 'entity', 'track')}


def audit_rows(rows, generation, release):
    counts = collections.Counter()
    previous = None
    seen = set()
    for row in rows:
        policy = row.get('combat_policy') or {}
        obs = policy.get('observation')
        if obs is None:
            continue
        identity = obs['identity']
        if (identity['spawncount'] != generation or identity['frame'] < release
                or identity['life'] != 1 or obs['health'] <= 0):
            previous = None
            continue
        key = tuple(identity[k] for k in IDENTITY + ('frame',))
        assert key not in seen, 'Duplicate teacher decision identity'
        seen.add(key)
        selection = policy['selection']
        assert selection['owner'] == selection['mode'] == 'rules'
        action = policy['applied']
        assert action['identity'] == identity
        counts['released_alive_frames'] += 1
        counts['raw_previous_target_present'] += obs.get('previous_target_intent') is not None
        past = previous_intent(previous, obs)
        counts['past_teacher_intent_reconstructable'] += past is not None
        if past is not None:
            counts['past_teacher_still_clear_same_track'] += any(
                e['id'] == past['entity'] and e.get('observed_track') == past['track']
                and e.get('clear_shot') for e in obs.get('enemies') or [])
        choice, reason = target_choice(policy)
        counts[reason] += 1
        counts['attack_frames'] += bool(action['attack'])
        counts['attack_without_explicit_visible_target'] += bool(action['attack']) and choice is None
        counts['movement_frames'] += bool(action['forward'] or action['side'])
        counts['weapon_request_frames'] += bool(action['weapon'])
        counts['vertical_' + action['vertical']] += 1
        # Explicit enemy and available none are different from ambiguous aim.
        # Do not assign a no-target training label to a navigation/prop decision.
        previous = choice
    return dict(counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    seal = read(args.corpus / 'complete.json')
    assert seal['state'] == 'complete' and seal['version'] == 'registered_rules_teacher_corpus_v1'
    assert not seal['training_performed'] and not seal['neural_execution']
    pins = {str(args.corpus / 'complete.json'): sha(args.corpus / 'complete.json')}
    assert sha(args.corpus / 'protocol.json') == seal['protocol_sha256']
    assert sha(args.corpus / 'pool/report.json') == seal['pool_sha256']
    for item in seal['inputs']:
        assert sha(item['path']) == item['sha256']
        pins[item['path']] = item['sha256']
    cases = []
    for case in seal['cases']:
        capture = read(pathlib.Path(case['root']) / 'report.json')
        worker = pathlib.Path(capture['results'][0]['root'])
        manifest = read(pathlib.Path(case['root']) / 'manifest.json')
        from native_combat_release import release_boundary
        with (worker / 'server.log').open(encoding='utf-8-sig') as stream:
            generation, release, _ = release_boundary(stream, manifest, case['seed'])
        assert release == case['network_release_frame']
        with (worker / 'bot.jsonl').open(encoding='utf-8-sig') as stream:
            counts = audit_rows(map(json.loads, stream), generation, release)
        assert counts['released_alive_frames'] == case['counts']['released_frames']
        cases.append(dict(seed=case['seed'], split=case['split'], counts=counts))
    totals = {split: dict(sum((collections.Counter(c['counts']) for c in cases if c['split'] == split),
                              collections.Counter())) for split in ('train', 'development')}
    for path, digest in pins.items():
        assert sha(path) == digest
    save(args.out, dict(state='complete', version='registered_joint_teacher_readiness_v1',
        cases=cases, totals=totals, inputs=[dict(path=p, sha256=h) for p, h in pins.items()],
        script_sha256=sha(__file__), training_performed=False, neural_execution=False,
        feature_export_complete=False, promotion=False,
        scope='Raw demonstration readiness only. Past-only target intent is reconstructable, not yet exported as features. Explicit visible teacher target mapping is a candidate label, not proof of optimality or Go availability. Non-enemy aim is ambiguous and never labelled as no target. Native matched dispatch proof and joint supervised masks are still required.'))
    print(json.dumps(dict(cases=len(cases), totals=totals)))


if __name__ == '__main__':
    main()

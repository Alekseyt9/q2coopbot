"""Audit visible policy target intent in closed captures; no neural inference."""
import argparse
import collections
import json
import pathlib

from process_combat_architecture_pool import read, save, sha


def audit(root):
    report = read(root / 'report.json')
    assert report['state'] == 'complete' and report['source_unchanged']
    assert report['usable_captures'] == len(report['jobs'])
    cases = []
    for job in sorted(report['jobs'], key=lambda j: j['job']):
        assert not job['error'] and job['mode'] == 'learned'
        case = pathlib.Path(job['root'])
        capture = read(case / 'report.json')
        assert capture['capture_complete'] and capture['provenance_valid']
        assert len(capture['results']) == 1
        result = capture['results'][0]
        assert result['capture_valid'] and result['generated_start']['confirmed']
        trace = pathlib.Path(result['root']) / 'bot.jsonl'
        counts = collections.Counter()
        chosen = collections.Counter()
        attacking = collections.Counter()
        runs = []
        previous = None
        seen = set()
        for line in trace.open(encoding='utf-8-sig'):
            row = json.loads(line)
            c = row.get('combat_policy') or {}
            selection = c.get('selection') or {}
            if selection.get('owner') != 'provider':
                previous = None
                continue
            obs, intent, applied = c['observation'], c['proposed'], c['applied']
            identity = obs['identity']
            if identity['life'] != 1 or obs['health'] <= 0:
                previous = None
                continue
            frame = identity['frame']
            frame_key = tuple(identity.get(k) for k in ('map', 'spawncount', 'connection', 'actor', 'life', 'frame'))
            assert frame_key not in seen, 'Duplicate provider decision frame'
            seen.add(frame_key)
            enemies = {e['id']: e for e in obs['enemies']}
            entity = intent.get('target_entity', 0)
            track = intent.get('target_track', 0)
            enemy = enemies.get(entity)
            if entity:
                assert enemy and enemy.get('observed_track') == track, 'Unobserved target intent'
            name = enemy['class'] if enemy else 'none'
            key = (entity, track)
            counts['provider_frames'] += 1
            chosen[name] += 1
            if applied['attack']:
                counts['attack_command_frames'] += 1
                attacking[name] += 1
            if any(e['class'] == 'monster_parasite' and e['distance'] < 288 for e in enemies.values()):
                counts['parasite_near_frames'] += 1
                if name == 'monster_parasite':
                    counts['parasite_near_selected_frames'] += 1
            if previous and frame_key[:-1] == previous['world'] and frame == previous['frame'] + 1:
                if key != previous['key']:
                    counts['intent_changes'] += 1
                    old = enemies.get(previous['key'][0])
                    if entity and previous['key'][0] and old and old.get('observed_track') == previous['key'][1]:
                        counts['changes_while_previous_still_observed'] += 1
                        if old.get('clear_shot') and enemy.get('clear_shot'):
                            counts['changes_with_both_clear'] += 1
            if not runs or previous is None or key != previous['key'] or frame != previous['frame'] + 1 or frame_key[:-1] != previous['world']:
                runs.append(dict(entity=entity, track=track, monster_class=name, first_frame=frame, last_frame=frame, frames=1))
            else:
                runs[-1]['last_frame'] = frame
                runs[-1]['frames'] += 1
            previous = dict(key=key, frame=frame, world=frame_key[:-1])
        cases.append(dict(job=job['job'], seed=job['seed'], map=result['generated_fixture']['map'],
                          episode=result['generated_fixture']['episode_id'], counts=dict(counts),
                          chosen_frames=dict(chosen), attack_intent_frames=dict(attacking), runs=runs,
                          goal=bool(result['goal_stop']), death=bool(result['death_stop']),
                          native_incoming=result['first_life']['incoming_by_attacker_class'],
                          native_outgoing=result['first_life']['outgoing_by_target_class'],
                          trace_sha256=sha(trace), capture_sha256=sha(case / 'report.json')))
    return dict(version='registered_target_intent_audit_v1', pool_sha256=sha(root / 'report.json'), cases=cases,
                scope='Actual captured provider intent, keyed by entity and visible track, not changing distance-sorted slots. Changes require consecutive same-world frames; clear is client BSP evidence. Attack commands are not native shots or hit attribution. No target choice is labelled useless by this audit.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), 'Fresh audit output required'
    result = audit(args.pool.resolve())
    save(args.out, result)
    maps = {}
    for case in result['cases']:
        maps.setdefault(case['map'], collections.Counter()).update(case['counts'])
    print(json.dumps(maps), flush=True)


if __name__ == '__main__':
    main()

"""Audit visible policy target intent in closed captures; no neural inference."""
import argparse
import collections
import json
import pathlib
import math

from process_combat_architecture_pool import read, save, sha
from process_registered_combat_pool_cuda import select_learned_jobs


def audit(root, plan_index=None):
    pool_sha=sha(root/'report.json')
    report = read(root / 'report.json')
    assert report['state'] == 'complete' and report['source_unchanged']
    assert report['usable_captures'] == len(report['jobs'])
    jobs=select_learned_jobs(report,plan_index)
    cases = []
    for job in jobs:
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
        first_provider_frame=first_attack_frame=None
        stationary_run=max_stationary_run=0
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
            enemies = {e['id']: e for e in (obs.get('enemies') or [])}
            entity = intent.get('target_entity', 0)
            track = intent.get('target_track', 0)
            enemy = enemies.get(entity)
            if entity:
                assert enemy and enemy.get('observed_track') == track, 'Unobserved target intent'
            name = enemy['class'] if enemy else 'none'
            key = (entity, track)
            counts['provider_frames'] += 1
            if first_provider_frame is None:
                first_provider_frame=frame
            speed=math.hypot(*obs['velocity'][:2])
            moving=speed>=10
            counts['observed_moving_frames' if moving else 'observed_stationary_frames']+=1
            requested_move=math.hypot(applied['forward'],applied['side'])>.1
            counts['requested_move_frames']+=int(requested_move)
            counts['stationary_requested_move_frames']+=int(not moving and requested_move)
            counts['large_yaw_command_frames']+=int(abs(applied['yaw_delta_degrees'])>=30)
            counts['large_pitch_command_frames']+=int(abs(applied['pitch_delta_degrees'])>=30)
            stationary_clear=not moving and any(e.get('clear_shot') for e in enemies.values())
            consecutive=previous and frame_key[:-1]==previous['world'] and frame==previous['frame']+1
            stationary_run=(stationary_run+1 if consecutive else 1) if stationary_clear else 0
            max_stationary_run=max(max_stationary_run,stationary_run)
            counts['stationary_clear_enemy_frames']+=int(stationary_clear)
            counts['stationary_clear_requested_move_frames']+=int(stationary_clear and requested_move)
            counts['stationary_clear_no_requested_move_frames']+=int(stationary_clear and not requested_move)
            chosen[name] += 1
            if applied['attack']:
                if first_attack_frame is None:
                    first_attack_frame=frame
                counts['attack_command_frames'] += 1
                counts['stationary_attack_command_frames']+=int(not moving)
                counts['attack_with_no_selected_target_frames']+=int(not entity)
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
                          first_provider_frame=first_provider_frame,first_attack_frame=first_attack_frame,
                          initial_attack_delay_frames=first_attack_frame-first_provider_frame if first_attack_frame is not None else None,
                          max_stationary_clear_enemy_run_frames=max_stationary_run,
                          trace_sha256=sha(trace), capture_sha256=sha(case / 'report.json')))
    assert sha(root/'report.json')==pool_sha
    return dict(version='registered_target_intent_audit_v2', pool_sha256=pool_sha,
                selected_plan_index=plan_index,script_sha256=sha(pathlib.Path(__file__)),cases=cases,
                scope='Actual captured first-life provider intent, keyed by entity and visible track, not distance-sorted slots. Consecutive same-world frames required for switches/runs; clear is client BSP evidence. Observed horizontal speed>=10 defines moving; requested movement norm>0.1 and applied turn magnitude>=30 degrees are descriptive. Initial delay begins at first provider frame, not reset/weapon setup. Attack frames are not native shots or hit attribution. Stationary/turn/switch actions are not labelled useless, and a maximum run is not a causal penalty recommendation. No neural execution; whole mixed pool acceptance is required before explicit learned plan selection.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    parser.add_argument('--plan-index',type=int)
    args = parser.parse_args()
    assert not args.out.exists(), 'Fresh audit output required'
    result = audit(args.pool.resolve(),args.plan_index)
    save(args.out, result)
    maps = {}
    for case in result['cases']:
        maps.setdefault(case['map'], collections.Counter()).update(case['counts'])
    print(json.dumps(maps), flush=True)


if __name__ == '__main__':
    main()

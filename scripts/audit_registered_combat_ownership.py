"""Check which controller issued commands in accepted first-life battles."""
import argparse
import collections
import json
import pathlib

from process_combat_architecture_pool import read, save, sha
from process_registered_combat_pool_cuda import select_learned_jobs
from native_combat_release import release_boundary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', type=pathlib.Path, required=True)
    parser.add_argument('--plan-index', type=int, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    pool_path = args.pool / 'report.json'
    pool_hash = sha(pool_path)
    pool = read(pool_path)
    jobs = select_learned_jobs(pool, args.plan_index)
    cases = []
    for job in jobs:
        capture_path = pathlib.Path(job['root']) / 'report.json'
        capture = read(capture_path)
        assert capture['capture_complete'] and capture['provenance_valid'] and len(capture['results']) == 1
        result = capture['results'][0]
        assert result['capture_valid'] and result['dispatch_valid']
        manifest_path = capture_path.parent / 'manifest.json'
        manifest = read(manifest_path)
        game_release = manifest['release_game_frame']
        assert manifest['synchronous'] and manifest['fixed_world_hold'] and game_release > 0
        log = pathlib.Path(result['root']) / 'server.log'
        log_hash = sha(log)
        with log.open(encoding='utf-8-sig') as stream:
            release_generation, release_frame, native_game_release = release_boundary(stream, manifest, job['seed'])
        trace = pathlib.Path(result['root']) / 'bot.jsonl'
        trace_hash = sha(trace)
        owners = collections.Counter()
        setup_owners = collections.Counter()
        fallbacks = collections.Counter()
        rules_attack_events = []
        setup_rules_attack_frames = 0
        visible_rules = visible_rules_attack = unknown_rules_action = unknown_observation = early_released = 0
        seen = set()
        for line in trace.open(encoding='utf-8-sig'):
            row = json.loads(line)
            policy = row.get('combat_policy') or {}
            selection = policy.get('selection') or {}
            owner = selection.get('owner')
            if not owner:
                continue
            observation = policy.get('observation')
            if not observation:
                unknown_observation += 1
                continue
            if observation['identity']['life'] != 1 or observation['health'] <= 0:
                continue
            identity = observation['identity']
            key = tuple(identity.get(k) for k in ('map','spawncount','connection','actor','life','frame'))
            assert key not in seen, 'Duplicate controller ownership frame'
            seen.add(key)
            if identity['spawncount'] != release_generation or identity['frame'] < release_frame:
                setup_owners[owner] += 1
                if owner == 'rules' and (policy.get('applied') or {}).get('attack'):
                    setup_rules_attack_frames += 1
                continue
            owners[owner] += 1
            early_released += identity['frame'] < game_release
            if owner == 'provider':
                continue
            assert owner == 'rules', 'Unknown controller owner'
            fallbacks[selection.get('fallback') or 'unspecified'] += 1
            if any(e.get('clear_shot') for e in (observation.get('enemies') or [])):
                visible_rules += 1
                applied = policy.get('applied')
                if applied is None:
                    unknown_rules_action += 1
                else:
                    visible_rules_attack += int(applied.get('attack', False))
                    if applied.get('attack'):
                        rules_attack_events.append(dict(frame=identity['frame'],fallback=selection.get('fallback')))
        assert sha(trace) == trace_hash
        assert sha(log) == log_hash
        cases.append(dict(job=job['job'], seed=job['seed'], release_frame=release_frame,
                          release_generation=release_generation, native_release_game_frame=native_game_release,
                          formerly_dropped_released_frames=early_released, server_log_sha256=log_hash,
                          owners=owners, setup_owners=setup_owners, setup_rules_attack_frames=setup_rules_attack_frames, fallbacks=fallbacks, rules_attack_events=rules_attack_events,
                          visible_enemy_rules_frames=visible_rules, visible_enemy_rules_attack_frames=visible_rules_attack,
                          visible_rules_missing_applied_frames=unknown_rules_action, unknown_observation_rows=unknown_observation,
                          goal=bool(result['goal_stop']), capture_sha256=sha(capture_path), manifest_sha256=sha(manifest_path), trace_sha256=trace_hash))
    assert sha(pool_path) == pool_hash
    fields = ('visible_enemy_rules_frames','visible_enemy_rules_attack_frames','visible_rules_missing_applied_frames','unknown_observation_rows')
    totals = {field:sum(case[field] for case in cases) for field in fields}
    totals['owners'] = dict(sum((collections.Counter(c['owners']) for c in cases), collections.Counter()))
    totals['setup_owners'] = dict(sum((collections.Counter(c['setup_owners']) for c in cases), collections.Counter()))
    totals['setup_rules_attack_frames'] = sum(c['setup_rules_attack_frames'] for c in cases)
    totals['formerly_dropped_released_frames'] = sum(c['formerly_dropped_released_frames'] for c in cases)
    save(args.out, dict(version='registered_combat_ownership_audit_v3', state='complete', pool_sha256=pool_hash,
                        selected_plan_index=args.plan_index, script_sha256=sha(pathlib.Path(__file__)), cases=cases, totals=totals,
                        release_helper_sha256=sha(pathlib.Path(__file__).with_name('native_combat_release.py')),
                        scope='Post-release first-life observed policy ownership from the native start marker network frame/generation; game release clock is not a network cutoff. Pre-release setup frames/commands reported separately. Clear enemies use client BSP evidence; rules attack commands are not bullet/damage attribution. Unknown observation/action rows are reported explicitly. Zero observed clear-enemy rules frames does not prove ownership of unlogged decisions. No training or neural execution.'))
    print(json.dumps(totals))


if __name__ == '__main__':
    main()

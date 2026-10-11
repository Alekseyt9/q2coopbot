"""Closed trace diagnostics: proposed/applied actions, not motion or shot success."""
import argparse
import collections
import hashlib
import json
import math
import pathlib
import re

def read(p):
    return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    with pathlib.Path(p).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def audit(trace, release, native_release, generation):
    counts = collections.Counter()
    reasons = collections.Counter()
    by_weapon = collections.defaultdict(collections.Counter)
    seen = set()
    with trace.open(encoding='utf-8-sig') as stream:
        for line in stream:
            c = json.loads(line).get('combat_policy') or {}
            owner = (c.get('selection') or {}).get('owner')
            if not owner:
                continue
            assert owner in ('provider','rules'), 'Unknown action owner'
            o = c['observation']
            identity = o['identity']
            if identity['life'] != 1 or o['health'] <= 0 or identity['frame'] < release:
                continue
            assert identity['spawncount']==generation, 'Native release generation mismatch'
            key = tuple(identity[k] for k in ('map','spawncount','connection','actor','life','frame'))
            assert key not in seen, 'Duplicate owned frame'
            seen.add(key)
            if owner == 'rules':
                visible = any(e.get('clear_shot') for e in o.get('enemies') or [])
                applied = c.get('applied')
                counts.update(dict(released_rules_frames=1,visible_enemy_rules_frames=int(visible),
                    visible_enemy_rules_attack_frames=int(visible and bool((applied or {}).get('attack'))),
                    visible_rules_missing_applied_frames=int(visible and applied is None)))
                continue
            p, a = c['proposed'], c['applied']
            for action in (p, a):
                assert tuple(action['identity'][k] for k in ('map','spawncount','connection','actor','life','frame')) == key
                assert isinstance(action['attack'], bool)
                assert action['vertical'] in ('release','jump','crouch')
                for component in ('forward','side','yaw_delta_degrees','pitch_delta_degrees'):
                    assert math.isfinite(action[component])
            assert isinstance(o['on_ground'], bool)
            visible = any(e.get('clear_shot') for e in o.get('enemies') or [])
            blocked_fire = p['attack'] and not a['attack']
            proposed_move = max(abs(p['forward']),abs(p['side'])) >= .05 or p['vertical'] != 'release'
            suppressed_move = proposed_move and a['forward'] == 0 and a['side'] == 0 and a['vertical'] == 'release'
            changed_move = max(abs(p['forward']-a['forward']),abs(p['side']-a['side'])) > .01 or p['vertical'] != a['vertical']
            events = dict(provider_frames=1, legacy_native_frame_cutoff_provider_frames=int(identity['frame']>=native_release), clear_enemy_frames=int(visible),
                proposed_attack_frames=int(p['attack']), applied_attack_frames=int(a['attack']),
                attack_suppressed_frames=int(blocked_fire),
                attack_suppressed_clear_enemy_frames=int(blocked_fire and visible),
                proposed_movement_frames=int(proposed_move),
                movement_command_fully_suppressed_frames=int(suppressed_move),
                movement_changed_beyond_rounding_frames=int(changed_move),
                airborne_movement_suppressed_frames=int(suppressed_move and not o['on_ground']))
            counts.update(events)
            by_weapon[o['weapon']].update(events)
            interventions = (c.get('selection') or {}).get('interventions') or []
            if blocked_fire:
                labels = {x['reason'] for x in interventions if x['component']=='attack'}
                if c.get('limit_reason'):
                    labels.add(c['limit_reason'])
                reasons.update('attack/'+x for x in labels or {'unreported'})
            if changed_move:
                labels = {x['reason'] for x in interventions if x['component'] in ('movement','vertical')}
                if c.get('move_limit_reason'):
                    labels.add(c['move_limit_reason'])
                reasons.update('movement/'+x for x in labels or {'unreported'})
    return dict(counts=dict(counts), reasons=dict(reasons), by_equipped_weapon=dict(by_weapon))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pool',type=pathlib.Path,required=True)
    p.add_argument('--plan-index',type=int,required=True)
    p.add_argument('--out',type=pathlib.Path,required=True)
    args=p.parse_args()
    assert not args.out.exists()
    source=args.pool/'report.json'; digest=sha(source); pool=read(source)
    assert pool['state']=='complete' and pool['source_unchanged']
    assert pool['usable_captures']==len(pool['jobs'])
    assert 0 <= args.plan_index < len(pool['plans']), 'Invalid plan index'
    binding=pool['plans'][args.plan_index]
    assert sha(binding['path'])==binding['sha256']
    plan=read(binding['path']);cases=[]
    for job in pool['jobs']:
        if job['plan_index']!=args.plan_index:
            continue
        assert job['mode']=='learned' and not job['error']
        capture_path=pathlib.Path(job['root'])/'report.json'; capture=read(capture_path)
        assert capture['capture_complete'] and capture['provenance_valid'] and len(capture['results'])==1
        result=capture['results'][0]
        assert result['capture_valid'] and result['dispatch_valid']
        manifest_path=capture_path.parent/'manifest.json';manifest=read(manifest_path)
        assert manifest['synchronous'] and manifest['fixed_world_hold'] and manifest['release_game_frame']>0
        trace=pathlib.Path(result['root'])/'bot.jsonl'; trace_hash=sha(trace)
        log=pathlib.Path(result['root'])/'server.log';log_hash=sha(log)
        pattern=re.compile(r'^sv_test_combat spawncount=(\d+) server_frame=(\d+) g_test_combat_start game_frame=(\d+) ready=1 seed=(\d+)$')
        releases=[]
        with log.open(encoding='utf-8-sig') as stream:
            for line in stream:
                if 'g_test_combat_start game_frame=' in line:
                    match=pattern.fullmatch(line.strip())
                    assert match, 'Malformed native release telemetry'
                    releases.append(tuple(map(int,match.groups())))
        assert len(releases)==1, 'Exactly one native release required'
        generation,network_frame,game_frame,seed=releases[0]
        assert game_frame==manifest['release_game_frame'] and seed==job['seed']
        data=audit(trace,network_frame,game_frame,generation)
        assert sha(trace)==trace_hash
        assert sha(log)==log_hash
        task=plan['tasks'][job['task_index']]
        cases.append(dict(job=job['job'],seed=job['seed'],episode=task['episode']['id'],loadout=task['episode']['recipe']['loadout'],
            network_release_frame=network_frame,native_release_game_frame=game_frame,release_generation=generation,server_log_sha256=log_hash,
            trace_sha256=trace_hash,capture_sha256=sha(capture_path),manifest_sha256=sha(manifest_path),**data))
    assert cases and sha(source)==digest
    totals=sum((collections.Counter(c['counts']) for c in cases),collections.Counter())
    reasons=sum((collections.Counter(c['reasons']) for c in cases),collections.Counter())
    out=dict(version='executed_action_intervention_audit_v2',state='complete',pool_sha256=digest,
        script_sha256=sha(__file__),selected_plan_index=args.plan_index,cases=cases,totals=dict(totals),reasons=dict(reasons),
        scope='Post-release first-life provider intents versus applied commands. Observation network frame is compared to the outer native release server_frame, not inner game_frame; legacy cutoff count supplied separately for reconciliation. Movement reductions smaller than .01 ignored as rounding unless vertical changes. Suppressed commands do not imply stationary bodies: existing momentum remains. Attack intents do not count native shots or accuracy. Airborne trajectory and weapon firing restrictions are separate execution interventions; not labels of useless model decisions.')
    args.out.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(cases=len(cases),totals=totals,reasons=reasons)))

if __name__=='__main__':
    main()

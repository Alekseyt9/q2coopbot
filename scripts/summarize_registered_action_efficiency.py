"""Join sealed behavior and native damage audits; no inference or reward fitting."""
import argparse
import collections
import hashlib
import json
import pathlib


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def summary(rows):
    counts = collections.Counter()
    for behavior, _ in rows:
        counts.update(behavior['counts'])
    frames = counts['provider_frames']
    attack = counts['attack_command_frames']
    delays = [d['first_monster_health_damage_delay_native_frames'] for _, d in rows
              if d['first_monster_health_damage_delay_native_frames'] is not None]
    commands = [b['initial_attack_delay_frames'] for b, _ in rows
                if b['initial_attack_delay_frames'] is not None]
    def fraction(key, denominator=frames):
        return counts[key] / denominator if denominator else None
    return dict(cases=len(rows), provider_frames=frames,
                stationary_fraction=fraction('observed_stationary_frames'),
                stationary_clear_enemy_fraction=fraction('stationary_clear_enemy_frames'),
                stationary_attack_command_fraction=fraction('stationary_attack_command_frames', attack),
                large_yaw_command_fraction=fraction('large_yaw_command_frames'),
                attack_without_target_fraction=fraction('attack_with_no_selected_target_frames', attack),
                target_changes_per_100_frames=100*fraction('intent_changes') if frames else None,
                changes_previous_still_observed_per_100_frames=100*fraction('changes_while_previous_still_observed') if frames else None,
                changes_both_targets_clear_per_100_frames=100*fraction('changes_with_both_clear') if frames else None,
                max_stationary_clear_enemy_run_frames=max((b['max_stationary_clear_enemy_run_frames'] for b, _ in rows), default=0),
                cases_stationary_clear_at_least_20_frames=sum(b['max_stationary_clear_enemy_run_frames'] >= 20 for b, _ in rows),
                never_attack_command_cases=sum(b['initial_attack_delay_frames'] is None for b, _ in rows),
                initial_attack_command_delay_frames_mean=sum(commands)/len(commands) if commands else None,
                no_monster_health_damage_cases=len(rows)-len(delays),
                first_monster_health_damage_delay_native_frames_mean=sum(delays)/len(delays) if delays else None,
                post_release_monster_health_damage=sum(d['post_release_monster_health_damage'] for _, d in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audit', action='append', required=True, help='LABEL=behavior-audit.json')
    ap.add_argument('--damage', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), 'Fresh output required'
    digest = sha(args.damage)
    damage = read(args.damage)
    assert damage['state'] == 'complete' and damage['version'] == 'registered_native_damage_timing_v1'
    native = {c['job']: c for c in damage['cases']}
    assert len(native) == len(damage['cases']), 'Duplicate damage job'
    variants = {}; sample = None
    for binding in args.audit:
        label, filename = binding.split('=', 1)
        path = pathlib.Path(filename).resolve(); before = sha(path); audit = read(path)
        assert label and label not in variants
        assert audit['version'] == 'registered_target_intent_audit_v3', 'Post-release behavior required'
        assert audit['pool_sha256'] == damage['pool_sha256']
        rows = []; groups = collections.defaultdict(list)
        keys = set()
        for b in audit['cases']:
            d = native[b['job']]
            key = (b['episode'], b['seed'], b['map']); assert key not in keys; keys.add(key)
            assert key == (d['episode'], d['seed'], d['map'])
            assert b['capture_sha256'] == d['capture_sha256'] and b['manifest_sha256'] == d['manifest_sha256']
            assert b['goal'] == d['goal'] and audit['selected_plan_index'] == d['plan_index']
            row = (b, d); rows.append(row)
            outcome = 'goal' if b['goal'] else 'death' if b['death'] else 'unfinished'
            group = 'recovery' if b['episode'].endswith('-recovery') else 'ordinary'
            for name in (outcome, group, group+'/'+outcome):
                groups[name].append(row)
        assert rows
        if sample is None: sample = keys
        assert keys == sample, 'Variants must use the same fixtures and seeds'
        variants[label] = dict(audit_sha256=before, selected_plan_index=audit['selected_plan_index'],
                               totals=summary(rows), groups={k:summary(v) for k,v in sorted(groups.items())})
        assert sha(path) == before
    assert sha(args.damage) == digest
    result = dict(version='registered_action_efficiency_summary_v1', state='complete',
                  pool_sha256=damage['pool_sha256'], damage_sha256=digest, script_sha256=sha(pathlib.Path(__file__)),
                  cases_per_variant=len(sample), variants=variants,
                  scope='Descriptive post-release first-life provider behavior joined to sealed native monster health damage. Command-delay and native-damage-delay clocks stay separate. Large yaw means applied command >=30 degrees; stationary means horizontal speed <10. A stationary run is consecutive observed frames. These are not confirmed waste, hit accuracy, causal effects, or reward recommendations. Native shot/impact evidence is required to diagnose bullets hitting walls. No inference, fitting, or promotion.')
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k:v['totals'] for k,v in variants.items()}, indent=2))


if __name__ == '__main__':
    main()

"""Summarize accepted native behavior audits without executing a neural model."""
import argparse
import collections
import hashlib
import json
import pathlib


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(cases):
    counts = collections.Counter()
    chosen = collections.Counter()
    delays = []
    never_attack = 0
    for case in cases:
        counts.update(case['counts'])
        chosen.update(case['chosen_frames'])
        delay = case['initial_attack_delay_frames']
        if delay is None:
            never_attack += 1
        else:
            delays.append(delay)
    frames = counts['provider_frames']
    clear = counts['stationary_clear_enemy_frames']
    return dict(cases=len(cases), counts=dict(counts), chosen_frames=dict(chosen),
                stationary_clear_no_request_fraction=(counts['stationary_clear_no_requested_move_frames'] / clear if clear else None),
                stationary_provider_fraction=(counts['observed_stationary_frames'] / frames if frames else None),
                first_attack_delay_frames_mean=(sum(delays) / len(delays) if delays else None),
                first_attack_delay_frames_max=max(delays, default=None), never_attacked_cases=never_attack,
                max_stationary_clear_enemy_run_frames=max((c['max_stationary_clear_enemy_run_frames'] for c in cases), default=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', action='append', required=True, help='LABEL=accepted_audit.json')
    parser.add_argument('--out', type=pathlib.Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), 'Fresh output required'
    variants = {}
    pool_hash = keys = None
    for argument in args.audit:
        label, filename = argument.split('=', 1)
        assert label and label not in variants
        path = pathlib.Path(filename).resolve()
        before = digest(path)
        audit = json.loads(path.read_text(encoding='utf-8-sig'))
        assert audit['version'] == 'registered_target_intent_audit_v2'
        cases = audit['cases']
        identities = {(c['episode'], c['seed'], c['map']) for c in cases}
        assert cases and len(identities) == len(cases)
        if pool_hash is None:
            pool_hash, keys = audit['pool_sha256'], identities
        assert audit['pool_sha256'] == pool_hash and identities == keys, 'Different pool or fixture sample'
        by_map = collections.defaultdict(list)
        for case in cases:
            by_map[case['map']].append(case)
        variants[label] = dict(audit_path=str(path), audit_sha256=before,
                               selected_plan_index=audit['selected_plan_index'],
                               totals=summarize(cases), maps={name:summarize(rows) for name, rows in sorted(by_map.items())})
        assert digest(path) == before, 'Input changed during summary'
    result = dict(version='registered_combat_behavior_summary_v1', state='complete',
                  pool_sha256=pool_hash, cases_per_variant=len(keys), variants=variants,
                  scope='Descriptive first-life provider commands and observed motion from accepted audits. Delays begin at first provider frame, excluding harness setup. Stationary fractions and target/turn counts do not prove wasted actions; attack commands are not bullets or hit accuracy. No training or neural execution.')
    with args.out.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(json.dumps({name:value['totals'] for name, value in variants.items()}, indent=2))


if __name__ == '__main__':
    main()

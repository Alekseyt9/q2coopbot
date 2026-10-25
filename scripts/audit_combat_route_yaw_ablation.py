"""Paired native prefix proof for a last-row route-yaw ablation; no neural work."""
import argparse
import copy
from pathlib import Path
from audit_combat_contact_loss_probe import load_steps
from audit_combat_joint_aim_grid import physical_equal_with_entity_bijection
from audit_combat_memory_route_zero import canonical_metrics
from audit_combat_motor_restore import route_active
from evaluate_registered_combat_models import read, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--screen', required=True, type=Path)
    ap.add_argument('--prepared', required=True, type=Path)
    ap.add_argument('--prior-screen', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    assert not a.out.exists()
    root = a.screen.resolve()
    pins = {}

    def pin(p):
        p = Path(p).resolve()
        pins[str(p)] = sha(p)
        return p

    spec = read(pin(root/'spec.json'))
    pool = read(pin(root/'pool/report.json'))
    quality = read(pin(root/'acceptance.json'))
    prep = read(pin(a.prepared/'report.json'))
    assert pool['state'] == 'complete' and pool['source_unchanged']
    assert quality['state'] == 'passed' and quality['numerical_verification_complete']
    assert spec['selection_kind'] == 'known_regression_diagnostic'
    assert prep['all_other_fields_exact'] and not prep['training_performed']
    control_path = next(v['model'] for v in spec['variants'] if v['label'] == 'source')
    candidate_path = next(v['model'] for v in spec['variants'] if v['label'] == 'zero')
    control, candidate = read(pin(control_path)), read(pin(candidate_path))
    reconstructed = copy.deepcopy(candidate)
    original_row = control['memory_route_movement']['layers'][-1]
    last = reconstructed['memory_route_movement']['layers'][-1]
    assert last['bias'][2] == 0 and not any(last['weight'][2])
    last['weight'][2] = original_row['weight'][2]
    last['bias'][2] = original_row['bias'][2]
    assert reconstructed == control, 'More than the route yaw output changed'
    jobs = {j['plan_index']: j for j in pool['jobs']}
    groups = []
    for ci in range(spec['independent_worlds']):
        variants = [v for v in spec['variants'] if v['context_index'] == ci]
        native, features = {}, {}
        for v in variants:
            if v['label'] not in ('source', 'zero'):
                continue
            assert sha(v['model']) == v['model_sha256']
            folder = root/f'case-{v["index"]:02d}'
            native[v['label']] = {
                s['observation']['identity']['frame']: s
                for s in load_steps(pin(next((folder/'native').glob('replay-*/steps.jsonl'))))
                if s['owner'] == 'provider' and s['observation']['identity']['life'] == 1
            }
            features[v['label']] = {
                r['frame']: r['features']
                for r in load_steps(pin(folder/'cuda/rollout.jsonl'))
            }
        left, right = native['source'], native['zero']
        frames = sorted(set(left) & set(right))
        assert frames
        first = None
        forward, reverse = {}, {}
        prefix = 0
        for frame in frames:
            x, y = left[frame], right[frame]
            assert physical_equal_with_entity_bijection(x['observation'], y['observation'], forward, reverse), 'Physical divergence before command divergence'
            if x['sent_command'] != y['sent_command']:
                assert route_active(features['source'][frame]) and route_active(features['zero'][frame])
                first = dict(frame=frame, changed_fields=[k for k in x['sent_command'] if x['sent_command'][k] != y['sent_command'][k]],
                             control_command=x['sent_command'], zero_yaw_command=y['sent_command'])
                break
            prefix += 1
        groups.append(dict(seed=variants[0]['context']['seed'], identical_physical_command_prefix_rows=prefix,
                           first_command_divergence=first,
                           identical_through_shared_provider_trace=first is None))
    # The preselected failing seed must reproduce its previous control outcome.
    old_spec = read(pin(a.prior_screen/'spec.json'))
    old_pool = read(pin(a.prior_screen/'pool/report.json'))
    old_v = next(v for v in old_spec['variants'] if v['label'] == 'zero' and v['context']['seed'] == 1172325051)
    new_v = next(v for v in spec['variants'] if v['label'] == 'source' and v['context']['seed'] == 1172325051)
    assert old_v['model_sha256'] == new_v['model_sha256'] and old_v['context']['instance'] == new_v['context']['instance']
    old_job = next(j for j in old_pool['jobs'] if j['plan_index'] == old_v['index'])
    old_result = read(pin(Path(old_job['root'])/'report.json'))['results'][0]
    new_result = read(pin(Path(jobs[new_v['index']]['root'])/'report.json'))['results'][0]
    # Whole-life summaries include rules warmup before provider frame98. Those
    # counters can vary by one warmup frame; prove the actual combat trace in
    # full instead of calling unequal whole-life aggregates identical.
    old_native = a.prior_screen/f'case-{old_v["index"]:02d}'/'native'
    new_native = root/f'case-{new_v["index"]:02d}'/'native'
    traces = []
    for folder in (old_native, new_native):
        traces.append({s['observation']['identity']['frame']: s
                       for s in load_steps(pin(next(folder.glob('replay-*/steps.jsonl'))))
                       if s['observation']['identity']['frame'] >= 98})
    old_trace, new_trace = traces
    assert old_trace.keys() == new_trace.keys() and old_trace
    forward, reverse = {}, {}
    for frame in sorted(old_trace):
        x, y = old_trace[frame], new_trace[frame]
        assert physical_equal_with_entity_bijection(x['observation'], y['observation'], forward, reverse)
        assert x['sent_command'] == y['sent_command'] and x['owner'] == y['owner']
    assert old_result['goal_stop'] == new_result['goal_stop'] and old_result['death_stop'] == new_result['death_stop']
    assert old_result['first_life']['damage'] == new_result['first_life']['damage']
    save(a.out, dict(version='combat_route_yaw_zero_causal_audit_v1', state='passed', groups=groups,
                     known_combat_control_reproduced=True, known_combat_trace_rows=len(old_trace),
                     whole_life_summary_identical=canonical_metrics(old_result) == canonical_metrics(new_result),
                     reproduction_scope='Full native physical observations, commands, ownership from frame98 including fallback, plus goal/death/damage. Whole-life pre98 warmup counters are not claimed identical.',
                     only_yaw_output_row_changed=True,
                     inputs=pins, promotion=False, training_performed=False,
                     scope='Known failure plus three new seeds at one site. No population, campaign or optimal-action claim.'))
    print(groups, flush=True)


if __name__ == '__main__':
    main()

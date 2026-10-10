"""Select one actor from sealed development RNG arms; never dispatch test data."""
import argparse
import pathlib

from process_combat_architecture_pool import read, save, sha
from run_combat_spatial_ppo import wait_process


def select(evaluation, groups, out):
    status, protocol, quality, ownership = [read(evaluation / p) for p in
        ('progress.json', 'protocol.json', 'quality-report.json', 'control-ownership.json')]
    assert status['stage'] == 'complete' and status['diagnostics_complete']
    for filename, field in [('quality-report.json', 'quality_report_sha256'),
        ('diagnostics-acceptance.json', 'diagnostics_acceptance_sha256'),
        ('ownership-acceptance.json', 'ownership_acceptance_sha256'), ('storage-final.json', 'physical_storage_sha256')]:
        assert sha(evaluation / filename) == status[field]
    assert quality['state'] == ownership['state'] == 'complete'
    assert quality['protocol_sha256'] == ownership['protocol_sha256'] == sha(evaluation / 'protocol.json')
    assert ownership['quality_sha256'] == sha(evaluation / 'quality-report.json')
    entries = {e['model'] + '-' + e['label']: e for e in protocol['evaluations']}
    variants = {v['variant']: v for v in quality['variants']}
    reference = variants[protocol['comparison_reference']]
    candidates = []
    for name, arms in groups.items():
        assert len(arms) == 2 and len(set(arms)) == 2
        actor_hashes, offsets, conditions = [], [], []
        for arm in arms:
            entry, summary, control = entries[arm], variants[arm], ownership['groups'][arm]
            assert entry.get('deterministic') is False
            assert not control['counts'].get('rules_with_clear_target', 0)
            assert not control['fallback_reasons'].get('pilot_equip_not_ready', 0)
            assert summary['episodes'] == reference['episodes'] == protocol['episodes_per_model']
            assert sha(entry['plan']) == entry['plan_sha256']
            plan = read(entry['plan'])
            assert sha(plan['model_path']) == plan['model_sha256'] == entry['deterministic_weights_sha256']
            actor_hashes.append(entry['source_weights_sha256'])
            offsets.append(plan.get('policy_sampling_seed_offset', 0))
            conditions.append({t['episode']['id']: (t['seeds'], t['instances']) for t in plan['tasks']})
        assert actor_hashes[0] and actor_hashes[0] == actor_hashes[1], 'Different actors cannot form RNG arms'
        assert offsets[0] != offsets[1] and conditions[0] == conditions[1]
        wins = sum(variants[a]['wins'] for a in arms)
        deaths = sum(variants[a]['deaths'] for a in arms)
        candidates.append(dict(name=name, arms=arms, actor_sha256=actor_hashes[0],
            wins_sum=wins, deaths_sum=deaths, mean_wins=wins / 2, mean_deaths=deaths / 2,
            mean_received_damage=sum(variants[a]['mean_received_damage'] for a in arms) / 2))
    assert candidates
    candidates.sort(key=lambda c: (-c['wins_sum'], c['deaths_sum'], c['mean_received_damage'], c['name']))
    chosen = candidates[0]
    eligible = (chosen['wins_sum'] > 2 * reference['wins'] or
        chosen['wins_sum'] == 2 * reference['wins'] and chosen['deaths_sum'] < 2 * reference['deaths'])
    decision = dict(state='selection_complete', selected=chosen, candidates=candidates, eligible=eligible,
        reference=dict(variant=reference['variant'], wins=reference['wins'], deaths=reference['deaths'], episodes=reference['episodes']),
        quality_sha256=sha(evaluation / 'quality-report.json'), ownership_sha256=sha(evaluation / 'control-ownership.json'),
        protocol_sha256=sha(evaluation / 'protocol.json'),
        scope='Development-only selection: sum of two distinct declared stochastic arms of one actor, '
        'tie deaths then damage. Eligibility compares average wins/deaths to rules. '
        'RNG repeats share engine conditions, not160 independent cases or statistical proof. '
        'No test dispatch/reservation, training or promotion; held-out inventory must be checked separately.')
    save(out / 'decision.json', decision)
    return dict(state='selection_complete', eligible=eligible, selected=chosen['name'],
                mean_wins=chosen['mean_wins'], reference_wins=reference['wins'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    parser.add_argument('--groups', nargs='+', required=True, help='actor=variant-a,variant-b')
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    evaluation, out = args.evaluation.resolve(), args.out.resolve()
    groups = {}
    for specification in args.groups:
        name, arms = specification.split('=', 1)
        assert name not in groups
        groups[name] = arms.split(',')
    assert not out.exists()
    out.mkdir()
    save(out / 'selection-protocol.json', dict(evaluation=str(evaluation), protocol_sha256=sha(evaluation / 'protocol.json'),
        groups=groups, observed_process_pid=args.wait_pid, test_dispatch=False))
    try:
        if args.wait_pid:
            wait_process(args.wait_pid, out / 'progress.json', 'waiting_for_sealed_development')
        assert sha(evaluation / 'protocol.json') == read(out / 'selection-protocol.json')['protocol_sha256']
        result = select(evaluation, groups, out)
        save(out / 'progress.json', result)
        print(result, flush=True)
    except Exception as error:
        save(out / 'progress.json', dict(state='failed', error=str(error)))
        raise

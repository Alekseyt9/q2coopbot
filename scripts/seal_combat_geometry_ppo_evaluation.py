"""Apply the fixed geometry-PPO endpoint gate; CPU metadata arithmetic only."""
import argparse
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation', 'spec', 'protocol', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    a = ap.parse_args(); assert not a.out.exists(); pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve(); h = sha(path)
        assert expected is None or expected == h, str(path)
        pins[str(path)] = h; return path

    seal = read(pin(a.evaluation/'complete.json')); spec = read(pin(a.spec, seal['spec_sha256']))
    protocol = read(pin(a.protocol, spec['experiment_protocol_sha256']))
    assert protocol['version'] == 'combat_geometry_delta_ppo_v1'
    assert seal['native_cases'] == 40 and seal['cuda_cases'] == 32
    assert [v['label'] for v in spec['variants']] == ['source', 'initial', 'B', 'G', 'rules']
    totals = {}; pairs = {}
    for c in spec['comparisons']:
        q = read(pin(a.evaluation/('quality-'+c['name']+'.json')))
        assert q['state'] == 'complete' and q['before_pool_sha256'] == q['after_pool_sha256'] == seal['pool_sha256']
        assert len(q['pairs']) == 8
        for side, label in (('before', c['before']), ('after', c['after'])):
            if label in totals: assert totals[label] == q['totals'][side]
            totals[label] = q['totals'][side]
        pairs[c['name']] = dict(gained=sum(not p['metrics']['before']['goal'] and p['metrics']['after']['goal'] for p in q['pairs']),
            lost=sum(p['metrics']['before']['goal'] and not p['metrics']['after']['goal'] for p in q['pairs']))
    shots = read(pin(a.evaluation/'machinegun-outcomes.json')); efficiency = read(pin(a.evaluation/'action-efficiency.json'))
    assert shots['state'] == efficiency['state'] == 'complete'
    proofs = {}
    for label in ('initial', 'B', 'G'):
        rows = []
        for p in sorted((a.evaluation/('processing-'+label)).glob('case-*/cuda/cuda-verification.json')):
            r = read(pin(p)); assert r['state'] == 'passed' and r['device'] == 'cuda'
            assert r['frame_hash_candidate_assignments_verified'] and r['original_latent_transform_verified']
            assert r['shared_candidate_precision'] == 'float64' and r['shared_candidate_all_logits_max_error'] < .001
            assert r['shared_candidate_all_probabilities_max_error'] < .0001
            rows.append(r)
        assert len(rows) == 8
        proofs[label] = dict(cases=8, frames=sum(r['shared_candidate_frames'] for r in rows), max_logit_error=max(r['shared_candidate_all_logits_max_error'] for r in rows))
    checks = {}; passed = []
    initial, rules = totals['initial'], totals['rules']
    for arm in ('B', 'G'):
        t = totals[arm]
        tests = dict(goals_initial=t['goal'] >= initial['goal']+2, goals_rules=t['goal'] >= rules['goal'],
            deaths=t['death'] <= initial['death'], received=t['received'] <= 1.1*initial['received'])
        if arm == 'G': tests['goals_frozen_head'] = t['goal'] >= totals['B']['goal']
        checks[arm] = tests
        if all(tests.values()): passed.append(arm)
    chosen = sorted(passed, key=lambda k: (-totals[k]['goal'], totals[k]['death'], totals[k]['received'], k))[0] if passed else None
    for path, h in pins.items(): assert sha(Path(path)) == h, path
    result = dict(state='complete', version='combat_geometry_ppo_endpoint_decision_v1', totals=totals, pairs=pairs, checks=checks,
        gate_passed=bool(passed), candidates_admitted_for_expansion=passed, selected_for_expansion=chosen,
        machinegun={k: v['all'] for k, v in shots['variants'].items()}, behavior={k: v['totals'] for k, v in efficiency['variants'].items()},
        cuda_proofs=proofs, inputs=pins, promotion=False, training_performed=False,
        scope='Fixed small endpoint screen; admission allows broader evaluation only. No claim of campaign superiority.')
    a.out.mkdir(parents=True); save(a.out/'report.json', result)
    save(a.out/'complete.json', dict(state='complete', report_sha256=sha(a.out/'report.json')))
    print(dict(totals=totals, pairs=pairs, checks=checks, gate_passed=bool(passed), selected_for_expansion=chosen))


if __name__ == '__main__': main()

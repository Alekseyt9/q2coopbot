"""Bind four live boundary proofs and exact physical replays to one objective."""
import argparse
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('acceptance', 'reward', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    a = ap.parse_args(); assert not a.out.exists(); pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve(); h = sha(path)
        assert expected is None or h == expected, str(path)
        pins[str(path)] = h; return path

    seal = read(pin(a.acceptance/'complete.json'))
    r = read(pin(a.acceptance/'report.json', seal['report_sha256']))
    assert r['state'] == 'complete' and r['native_cases'] == r['cuda_cases'] == 4 and not r['ordinary_training_eligible']
    assert set(r['boundary_types']) == {'goal','death','horizon','unfinished'}
    reward = read(pin(a.reward)); assert reward['handoff_bootstrap'] == 'nonterminal_v1'
    p = read(pin(a.acceptance/'protocol.json'))
    original = {}
    for path, h in p['inputs'].items():
        pin(path, h)
        if Path(path).name == 'report.json' and Path(path).parent.name == 'pool':
            old = read(path)
            for job in old['jobs']:
                capture_path = Path(job['root'])/'report.json'
                # Only replayed cases are needed, not the whole old corpus.
                if job['seed'] in {v['seed'] for v in r['boundary_types'].values()}:
                    result = read(pin(capture_path))['results'][0]
                    original[job['seed']] = result['generated_fixture']
    pool = read(pin(a.acceptance/'pool/report.json', r['pool_sha256']))
    assert pool['state'] == 'complete' and pool['source_unchanged']
    for index, label in enumerate(p['labels']):
        root = a.acceptance/label; entry = r['boundary_types'][label]
        credit = read(pin(root/'native-credit/report.json', entry['credit_sha256']))
        gae = read(pin(root/'advantage-credit/report.json', entry['gae_sha256']))
        assert credit['state'] == gae['state'] == 'complete' and not credit['unassigned_native_effects']
        assert credit['spacing_telescoping_max_error'] < 1e-8 and gae['gae_closed_form_max_error'] < 1e-8
        for h in credit['handoffs']: assert h['bootstrap_contract_verified'] and h['handoff_bootstrap'] == 'nonterminal_v1'
        metas = list((root/'processing-train').glob('case-*/cuda/report.json')); assert len(metas) == 1
        meta = read(pin(metas[0])); assert meta['reward_config_sha256'] == sha(a.reward) and meta['handoff_bootstrap'] == 'nonterminal_v1'
        proof = read(pin(metas[0].with_name('cuda-verification.json'), meta['cuda_verification_sha256']))
        assert proof['state'] == 'passed' and proof['device'] == 'cuda'
        assert proof['frame_hash_candidate_assignments_verified'] and proof['original_latent_transform_verified']
        job = next(j for j in pool['jobs'] if j['plan_index'] == index)
        result = read(pin(Path(job['root'])/'report.json'))['results'][0]
        assert result['generated_fixture'] == original[job['seed']], 'Physical replay differs'
        if label == 'goal': assert entry['goal'] and not entry['death']
        if label == 'death': assert entry['death'] and not entry['goal']
        if label == 'horizon': assert not entry['goal'] and not entry['death'] and entry['retained_bootstraps'] > 0
        if label == 'unfinished': assert any(not h['confirmed_goal'] and not h['bootstrap_zero'] for h in entry['handoffs'])
    result = dict(state='complete', version='combat_nonterminal_boundary_acceptance_v1', native_cases=4, cuda_cases=4,
        physical_starts_exact=True, reward_sha256=sha(a.reward), handoff_bootstrap='nonterminal_v1', boundary_types=r['boundary_types'],
        ordinary_training_eligible=False, training_performed=False, promotion=False, inputs=pins, script_sha256=sha(__file__))
    a.out.mkdir(parents=True); save(a.out/'report.json', result)
    save(a.out/'complete.json', dict(state='complete', report_sha256=sha(a.out/'report.json')))
    print(dict(state='complete', native_cases=4, cuda_cases=4, physical_starts_exact=True, reward_sha256=sha(a.reward)))


if __name__ == '__main__': main()

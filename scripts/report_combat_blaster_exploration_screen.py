"""Close a predeclared fresh blaster screen; weights-only policy is not training."""
import argparse
from pathlib import Path
import subprocess
import sys

from evaluate_registered_combat_models import read, save, sha
from report_combat_movement_variance_confirmation import paired_summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'protocol', 'spec'):
        ap.add_argument('--'+name, type=Path, required=True)
    a = ap.parse_args()
    assert not (a.root/'result').exists()
    selection = read(a.root/'selection.json')
    assert all(sha(path) == digest for path, digest in selection['source_files'].items())
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    pin(__file__)
    pin(a.root/'selection.json')
    protocol, spec = read(pin(a.protocol)), read(pin(a.spec))
    assert protocol['version'] == 'combat_blaster_exploration_fresh_screen_v1'
    assert spec['experiment_protocol_sha256'] == sha(a.protocol) and spec['expansion_gate'] == protocol['gate']
    evaluation = a.root/'evaluation'
    closed = read(pin(evaluation/'complete.json'))
    assert closed['state'] == 'complete' and closed['native_cases'] == 24 and closed['cuda_cases'] == 16
    assert closed['spec_sha256'] == sha(a.spec) and not closed['training_performed'] and not closed['promotion']
    fixture = read(pin(evaluation/'fixture-proof.json'))
    assert fixture['spec_sha256'] == sha(a.spec) and fixture['families'] == 8
    assert {m['label']: m['model_sha256'] for m in fixture['models']} == protocol['model_sha256']
    totals, comparisons = {}, {}
    for item in closed['comparisons']:
        data = read(pin(evaluation/('quality-'+item['name']+'.json'), item['sha256']))
        assert item['after'] == 'candidate' and len(data['pairs']) == 8
        assert len({p['seed'] for p in data['pairs']}) == 8
        totals[item['before']] = data['totals']['before']
        if 'candidate' in totals:
            assert totals['candidate'] == data['totals']['after']
        totals['candidate'] = data['totals']['after']
        comparisons[item['name']] = paired_summary(data['pairs'])
    assert set(totals) == {'source', 'candidate', 'rules'}
    candidate = totals['candidate']
    gate = protocol['gate']
    checks = dict(additional_goals_source=candidate['goal'] >= totals['source']['goal']+gate['additional_goals_over_source'],
        goals_at_least_rules=candidate['goal'] >= totals['rules']['goal'])
    for label in ('source', 'rules'):
        checks['deaths_no_more_than_'+label] = candidate['death'] <= totals[label]['death']
        checks['received_within_limit_'+label] = gate['received_denominator']*candidate['received'] <= gate['received_numerator']*totals[label]['received']
    for label in ('source', 'candidate'):
        ownership = read(pin(evaluation/('ownership-'+label+'.json')))
        assert ownership['pool_sha256'] == closed['pool_sha256'] and len(ownership['cases']) == 8
        checks['model_owns_visible_combat_'+label] = all(ownership['totals'][k] == 0 for k in
            ('visible_enemy_rules_frames', 'visible_enemy_rules_attack_frames', 'visible_rules_missing_applied_frames', 'unknown_observation_rows'))
    quality = a.root/'native-quality'
    assert not quality.exists(), 'Fresh native-action audit required'
    with (a.root/'native-quality.log').open('w', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('audit_registered_native_action_quality.py')),
            '--evaluation', str(evaluation), '--out', str(quality)], stdout=log, stderr=log,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    assert result.returncode == 0
    seal = read(pin(quality/'complete.json'))
    actions = read(pin(quality/'report.json', seal['report_sha256']))
    assert all(sha(path) == digest for path, digest in selection['source_files'].items())
    assert all(sha(path) == digest for path, digest in pins.items())
    output = a.root/'result'
    output.mkdir()
    save(output/'report.json', dict(state='complete', version=protocol['version'], gate_passed=all(checks.values()),
        checks=checks, totals=totals, comparisons=comparisons, actions=actions['variants'], inputs=pins,
        training_performed=False, promotion=False, scope=protocol['scope']))
    lines = ['# Fresh blaster exploration screen', '', '| Variant | Goals / 8 | Deaths | Kills | Received HP |',
        '|---|---:|---:|---:|---:|']
    for label in ('source', 'candidate', 'rules'):
        t = totals[label]
        lines.append(f"| {label} | {t['goal']} | {t['death']} | {t['kills']} | {t['received']} |")
    lines += ['', f'Gate: {all(checks.values())}. Promotion: false.']
    (output/'results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    save(output/'complete.json', dict(state='complete', report_sha256=sha(output/'report.json'), markdown_sha256=sha(output/'results.md')))
    save(a.root/'complete.json', dict(state='complete', native_cases=24, cuda_cases=16, source_unchanged=True,
        result_seal_sha256=sha(output/'complete.json'), training_performed=False, promotion=False))
    print(dict(gate_passed=all(checks.values()), totals=totals))


if __name__ == '__main__':
    main()

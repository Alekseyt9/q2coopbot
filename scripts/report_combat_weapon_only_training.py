"""Seal a fixed weapon-only pilot; report negative results without changing its gate."""
import argparse
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha
from report_combat_movement_variance_confirmation import paired_summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('workflow', 'protocol', 'spec', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    a = ap.parse_args(); assert not a.out.exists()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve(); digest = sha(path)
        assert expected is None or expected == digest, str(path)
        pins[str(path)] = digest
        return path

    protocol, spec = read(pin(a.protocol)), read(pin(a.spec))
    run = read(pin(a.workflow/'run.json'))
    assert run['configs'][str(a.protocol.resolve())] == sha(a.protocol)
    assert run['configs'][str(a.spec.resolve())] == sha(a.spec)
    assert all(sha(path) == digest for path,digest in run['source_files'].items())
    evaluation = a.workflow/'evaluation'
    closed = read(pin(evaluation/'complete.json'))
    assert closed['state'] == 'complete' and closed['native_cases'] == 40 and closed['cuda_cases'] == 32
    assert closed['spec_sha256'] == sha(a.spec) and not closed['training_performed'] and not closed['promotion']
    fixture = read(pin(evaluation/'fixture-proof.json'))
    assert fixture['spec_sha256'] == sha(a.spec) and fixture['families'] == 8
    labels = {v['label'] for v in spec['variants']}
    assert labels == {'source', 'control-trained', 'explore-static', 'explore-trained', 'rules'}
    totals, comparisons = {}, {}
    for comparison in closed['comparisons']:
        data = read(pin(evaluation/('quality-'+comparison['name']+'.json'), comparison['sha256']))
        assert comparison['after'] == 'explore-trained' and len(data['pairs']) == 8
        assert len({p['seed'] for p in data['pairs']}) == 8
        totals[comparison['before']] = data['totals']['before']
        if 'explore-trained' in totals:
            assert totals['explore-trained'] == data['totals']['after']
        totals['explore-trained'] = data['totals']['after']
        comparisons[comparison['name']] = paired_summary(data['pairs'])
    assert set(totals) == labels
    trained = totals['explore-trained']; gate = protocol['screen_gate']
    checks = dict(additional_goals_control=trained['goal'] >= totals['control-trained']['goal']+gate['additional_goals_over_control_trained'],
        additional_goals_static=trained['goal'] >= totals['explore-static']['goal']+gate['additional_goals_over_explore_static'],
        goals_at_least_rules=trained['goal'] >= totals['rules']['goal'])
    for label in ('control-trained', 'rules'):
        checks['deaths_no_more_than_'+label] = trained['death'] <= totals[label]['death']
        checks['received_bounded_'+label] = gate['received_denominator']*trained['received'] <= gate['received_numerator']*totals[label]['received']
    for label in sorted(labels-{'rules'}):
        ownership = read(pin(evaluation/('ownership-'+label+'.json')))
        assert ownership['pool_sha256'] == closed['pool_sha256'] and len(ownership['cases']) == 8
        checks['model_ownership_'+label] = all(ownership['totals'][k] == 0 for k in
            ('visible_enemy_rules_frames', 'visible_enemy_rules_attack_frames', 'visible_rules_missing_applied_frames', 'unknown_observation_rows'))
    updates = []
    for arm in ('control', 'explore'):
        target = a.workflow/arm/'update'; seal = read(pin(target/'complete.json'))
        report = read(pin(target/'report.json', seal['report_sha256']))
        assert report['training_performed'] and report['scope_updates'] == 1
        assert report['nonweapon_outputs_frozen_exact'] and report['checkpoint_roundtrip_exact']
        assert report['disposable_resume_adam_step_exact'] and report['disposable_critic_resume_adam_step_exact']
        updates.append(dict(arm=arm, rows=report['rows'], actor_steps=report['actor_steps'], critic_steps=report['critic_steps'],
                            weights_sha256=report['weights_sha256']))
    native_seal = read(pin(a.workflow/'native-quality/complete.json'))
    native = read(pin(a.workflow/'native-quality/report.json', native_seal['report_sha256']))
    assert all(sha(path) == digest for path,digest in pins.items())
    assert all(sha(path) == digest for path,digest in run['source_files'].items())
    a.out.mkdir()
    save(a.out/'report.json', dict(state='complete', gate_passed=all(checks.values()), checks=checks,
        totals=totals, comparisons=comparisons, updates=updates, actions=native['variants'], inputs=pins,
        promotion=False, scope=protocol['scope']))
    lines = ['# Weapon-only training pilot', '', '| Variant | Goals / 8 | Deaths | Kills | Received HP |',
             '|---|---:|---:|---:|---:|']
    for label in ('source', 'control-trained', 'explore-static', 'explore-trained', 'rules'):
        t = totals[label]
        lines.append(f"| {label} | {t['goal']} | {t['death']} | {t['kills']} | {t['received']} |")
    lines += ['', f'Expansion gate: {all(checks.values())}. Promotion: false.']
    (a.out/'results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    save(a.out/'complete.json', dict(state='complete', report_sha256=sha(a.out/'report.json'), markdown_sha256=sha(a.out/'results.md')))
    print(dict(gate_passed=all(checks.values()), totals=totals))


if __name__ == '__main__':
    main()

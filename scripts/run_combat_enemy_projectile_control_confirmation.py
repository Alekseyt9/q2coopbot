"""Fresh confirmation of a post-hoc neutral endpoint; no fitting or promotion."""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import subprocess
import sys

from evaluate_registered_combat_models import read, save, sha
from report_combat_movement_variance_confirmation import paired_summary
from audit_combat_projectile_training_exposure import monster_damage_kind
from run_combat_diverse_registry_ppo import process_alive


VERSION = 'combat_enemy_projectile_control_confirmation_v1'


def summarize(root, protocol_path, spec_path):
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = pins.get(str(path)) or sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    protocol = read(pin(protocol_path))
    spec = read(pin(spec_path))
    assert protocol['version'] == VERSION
    prerequisite = read(pin(protocol['prerequisite_report'], protocol['prerequisite_report_sha256']))
    assert not prerequisite['gate_passed'], 'The failed typed hypothesis stays failed'
    prior_root = Path(protocol['prerequisite_report']).parents[1]
    prior = read(pin(prior_root/'complete.json', protocol['prerequisite_workflow_sha256']))
    assert prior['state'] == 'complete' and prior['source_unchanged']
    assert prior['native_train_cases'] == prior['cuda_train_cases'] == 192
    prior_seal = read(pin(prior_root/'result/complete.json', prior['result_seal_sha256']))
    assert prior_seal['report_sha256'] == protocol['prerequisite_report_sha256']
    closed = read(pin(root/'evaluation/complete.json'))
    assert closed['state'] == 'complete' and closed['native_cases'] == 72 and closed['cuda_cases'] == 48
    assert not closed['training_performed'] and not closed['promotion']
    assert closed['spec_sha256'] == sha(spec_path)
    assert spec['experiment_protocol_sha256'] == sha(protocol_path)
    assert spec['expansion_gate'] == protocol['gate'] and spec['split'] == 'confirmation'
    fixture = read(pin(root/'evaluation/fixture-proof.json'))
    assert fixture['spec_sha256'] == closed['spec_sha256'] and fixture['families'] == 24
    models = {m['label']: m['model_sha256'] for m in fixture['models']}
    assert set(models) == {'initial', 'candidate'}
    for label, endpoint in protocol['endpoints'].items():
        folder = Path(endpoint['path'])
        seal = read(pin(folder/'complete.json', endpoint['seal_sha256']))
        pin(folder/'weights.json', endpoint['weights_sha256'])
        assert models[label] == seal['weights_sha256'] == endpoint['weights_sha256']
        report = read(pin(folder/'report.json', seal['report_sha256']))
        assert report['device'] == 'cuda' and report['scope_updates'] == (4 if label == 'candidate' else 0)
    pool = read(pin(root/'evaluation/pool/report.json', closed['pool_sha256']))
    assert pool['source_unchanged'] and pool['usable_captures'] == len(pool['jobs']) == 72
    train_seeds = set()
    for number in range(1, 5):
        for arm in ('typed', 'neutral'):
            corpus = prior_root/f'round-{number:02d}'/arm/'corpus'
            receipt = read(pin(corpus/'complete.json'))
            collection = read(pin(corpus/'protocol.json', receipt['protocol_sha256']))
            for assignment in collection['assignments']:
                assert assignment['split'] == 'train'
                train_seeds.update(assignment['seeds'])
    assert len(train_seeds) == 192
    prior_eval = read(pin(prior_root/'evaluation/complete.json'))
    prior_seeds = {j['seed'] for j in read(pin(prior_root/'evaluation/pool/report.json', prior_eval['pool_sha256']))['jobs']}
    infighting, corpse_contacts = 0, 0
    for job in pool['jobs']:
        capture = read(pin(Path(job['root'])/'report.json'))
        start = capture['results'][0]['generated_start']
        assert start['confirmed']
        assert next(e for e in start['entities'] if e['class'] == 'monster_soldier_light')['health'] == 20
        logs = list(Path(job['root']).glob('worker-*-episode-*/server.log'))
        assert len(logs) == 1
        with pin(logs[0]).open(encoding='utf-8-sig') as stream:
            for line in stream:
                if line.startswith('sv_test_damage '):
                    fields = dict(part.split('=', 1) for part in line.split() if '=' in part)
                    kind = monster_damage_kind(fields)
                    infighting += kind == 'infighting'
                    corpse_contacts += kind == 'corpse_contact'
    totals, comparisons = {}, {}
    for item in closed['comparisons']:
        data = read(pin(root/'evaluation'/('quality-'+item['name']+'.json'), item['sha256']))
        assert item['after'] == 'candidate' and len(data['pairs']) == 24
        assert len({p['seed'] for p in data['pairs']}) == 24
        assert not (train_seeds | prior_seeds).intersection(p['seed'] for p in data['pairs'])
        totals[item['before']] = data['totals']['before']
        if 'candidate' in totals:
            assert totals['candidate'] == data['totals']['after']
        totals['candidate'] = data['totals']['after']
        groups = defaultdict(list)
        for pair in data['pairs']:
            episode = pair['episode']
            groups['map:'+pair['map']].append(pair)
            groups['site:01' if '-site-01-' in episode else 'site:02'].append(pair)
            groups['trio' if '-parasite-' in episode else 'duo'].append(pair)
            groups['loadout:'+episode.rsplit('-', 1)[1]].append(pair)
        assert len(groups) == 11
        for name, rows in groups.items():
            assert len(rows) == (6 if name.startswith('map:') else 8 if name.startswith('loadout:') else 12)
        comparisons[item['name']] = dict(paired=paired_summary(data['pairs']),
            strata={name: paired_summary(rows) for name, rows in sorted(groups.items())})
    assert set(totals) == {'initial', 'candidate', 'rules'}
    gate, candidate = protocol['gate'], totals['candidate']
    checks = dict(no_live_monster_infighting=infighting == 0)
    for label in ('initial', 'rules'):
        checks['additional_goals_'+label] = candidate['goal'] >= totals[label]['goal']+gate['additional_goals_over_'+label]
        checks['deaths_no_more_than_'+label] = candidate['death'] <= totals[label]['death']
        checks['received_within_limit_'+label] = (gate['received_denominator']*candidate['received']
            <= gate['received_numerator']*totals[label]['received'])
    for label in ('initial', 'candidate'):
        ownership = read(pin(root/'evaluation'/('ownership-'+label+'.json')))
        assert ownership['pool_sha256'] == closed['pool_sha256'] and len(ownership['cases']) == 24
        checks['model_owns_visible_combat_'+label] = all(ownership['totals'][k] == 0 for k in
            ('visible_enemy_rules_frames', 'visible_enemy_rules_attack_frames',
             'visible_rules_missing_applied_frames', 'unknown_observation_rows')) and not any(
                 c['goal'] and c['rules_attack_events'] for c in ownership['cases'])
    quality_seal = read(pin(root/'native-quality/complete.json'))
    quality = read(pin(root/'native-quality/report.json', quality_seal['report_sha256']))
    assert all(sha(p) == h for p, h in pins.items())
    result = root/'result'
    result.mkdir()
    save(result/'report.json', dict(state='complete', version=VERSION, gate_passed=all(checks.values()),
        checks=checks, totals=totals, comparisons=comparisons, actions=quality['variants'], inputs=pins,
        native_infighting_events=infighting, native_monster_corpse_contacts=corpse_contacts,
        training_performed=False, promotion=False, selection=protocol['selection'], scope=protocol['scope']))
    lines = ['# Independent neutral control confirmation', '',
        '| Variant | Goals / 24 | Deaths | Kills | Received HP |', '|---|---:|---:|---:|---:|']
    for label in ('initial', 'candidate', 'rules'):
        t = totals[label]
        lines.append(f"| {label} | {t['goal']} | {t['death']} | {t['kills']} | {t['received']} |")
    lines += ['', f'Gate: {all(checks.values())}. Promotion: false.']
    (result/'results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    save(result/'complete.json', dict(state='complete', report_sha256=sha(result/'report.json'),
        markdown_sha256=sha(result/'results.md')))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'spec', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--recover-empty-startup', type=Path)
    ap.add_argument('--recovery-proof', type=Path)
    a = ap.parse_args()
    repo, root = Path(__file__).resolve().parents[1], a.out.resolve()
    assert not root.exists(), 'Fresh output required; failed data must be retained'
    protocol = read(a.protocol)
    assert protocol['version'] == VERSION and not protocol['training_performed']
    assert protocol['native_cases'] == 72 and protocol['cuda_cases'] == 48
    spec = read(a.spec)
    assert spec['experiment_protocol_sha256'] == sha(a.protocol)
    assert spec['split'] == 'confirmation' and spec['count'] == 1 and spec['max_instances'] == 16
    assert spec['expansion_gate'] == protocol['gate'] and len(set(spec['episodes'])) == 24
    assert sha(protocol['prerequisite_report']) == protocol['prerequisite_report_sha256']
    assert not read(protocol['prerequisite_report'])['gate_passed']
    assert sha(Path(protocol['prerequisite_report']).parents[1]/'complete.json') == protocol['prerequisite_workflow_sha256']
    variants = {v['label']: v for v in spec['variants']}
    assert set(variants) == {'initial', 'candidate', 'rules'}
    for label, endpoint in protocol['endpoints'].items():
        folder = Path(endpoint['path'])
        assert sha(folder/'complete.json') == endpoint['seal_sha256']
        assert sha(folder/'weights.json') == read(folder/'complete.json')['weights_sha256'] == endpoint['weights_sha256']
        assert Path(variants[label]['update']).resolve() == folder.resolve()
        assert variants[label]['cuda_replay'] and variants[label]['deterministic'] is False
    assert variants['rules']['mode'] == 'rules' and not variants['rules']['cuda_replay']
    assert bool(a.recover_empty_startup) == bool(a.recovery_proof)
    recovery_digest = None
    if a.recover_empty_startup:
        original = a.recover_empty_startup.resolve()
        state = read(original/'progress.json')
        assert state['stage'] == 'failed' and not process_alive(state.get('runner_pid'))
        assert read(original/'evaluation/progress.json')['stage'] == 'failed'
        admission = read(a.recovery_proof)
        recovery_digest = sha(a.recovery_proof)
        assert admission['version'] == 'combat_enemy_projectile_empty_startup_recovery_v1'
        assert admission['original_root'] == str(original)
        assert admission['original_run_sha256'] == sha(original/'run.json')
        assert admission['original_pool_sha256'] == sha(original/'evaluation/pool/report.json')
        assert admission['retained_cases'] == 64 and admission['empty_startup_cases'] == 8
        replacements = {str((repo/p).resolve()): item for p, item in admission['source_revisions'].items()}
        assert set(replacements) == {str(Path(__file__).resolve()), str(repo/'scripts/recover_registered_combat_startup.py')}
        for path, digest in read(original/'run.json')['source_files'].items():
            if path in replacements:
                item = replacements[path]
                assert digest == item['before_sha256'] == sha(item['before_snapshot'])
                assert sha(path) == item['after_sha256']
            else:
                assert sha(path) == digest, path
    paths = {a.protocol.resolve(), a.spec.resolve()}
    if a.recovery_proof:
        paths.add(a.recovery_proof.resolve())
    for folder, suffixes in ((repo/'scripts', ('.py', '.ps1')), (repo/'cmd', ('.go',)),
            (repo/'internal', ('.go',)), (repo.parent/'yquake2/src', ('.c', '.h'))):
        paths.update(p.resolve() for p in folder.rglob('*') if p.is_file() and p.suffix in suffixes)
    fence = {str(p): sha(p) for p in sorted(paths)}
    root.mkdir()
    save(root/'run.json', dict(version=VERSION, source_files=fence, native_cases=72, cuda_cases=48,
        training_performed=False, promotion=False, startup_recovery_sha256=recovery_digest))
    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    try:
        evaluation_commands = [('evaluation', ['evaluate_registered_combat_models.py', '--spec', a.spec.resolve(), '--out', root/'evaluation'])]
        if a.recover_empty_startup:
            evaluation_commands = [
                ('recover-native', ['recover_registered_combat_startup.py', '--root', a.recover_empty_startup.resolve()/'evaluation',
                    '--spec', a.spec.resolve(), '--out', root/'evaluation', '--port', '41400']),
                ('evaluation', ['finalize_recovered_registered_evaluation.py', '--evaluation', root/'evaluation', '--spec', a.spec.resolve()])]
        for label, command in evaluation_commands + [
            ('native-quality', ['audit_registered_native_action_quality.py', '--evaluation', root/'evaluation', '--out', root/'native-quality'])]:
            assert all(sha(p) == h for p, h in fence.items()), 'Source/config changed'
            with (root/(label+'.log')).open('w', encoding='utf-8') as log:
                child = subprocess.Popen([sys.executable, str(repo/'scripts'/command[0]), *map(str, command[1:])],
                    cwd=repo, env=env, stdout=log, stderr=log,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                save(root/'progress.json', dict(stage=label, runner_pid=os.getpid(), child_pid=child.pid))
                assert child.wait() == 0, str(root/(label+'.log'))
        save(root/'progress.json', dict(stage='result', runner_pid=os.getpid(), child_pid=None))
        summarize(root, a.protocol, a.spec)
        assert all(sha(p) == h for p, h in fence.items()), 'Source/config changed'
        save(root/'complete.json', dict(state='complete', version=VERSION, source_unchanged=True,
            native_cases=72, cuda_cases=48, result_seal_sha256=sha(root/'result/complete.json'),
            training_performed=False, promotion=False, startup_recovery_sha256=recovery_digest,
            original_source_unchanged=recovery_digest is None))
        save(root/'progress.json', dict(stage='complete', runner_pid=os.getpid(), child_pid=None))
    except BaseException as error:
        save(root/'progress.json', dict(stage='failed', runner_pid=os.getpid(), error=str(error)))
        raise


if __name__ == '__main__':
    main()

"""Compare arbitrary sealed native variants on identical generated conditions."""
import argparse
import pathlib

from process_combat_architecture_pool import read, save, sha
from run_combat_spatial_ppo import wait_process


def report(root, pairs):
    files = ('protocol.json', 'quality-report.json', 'recovery/verified-members.json', 'diagnostics-acceptance.json')
    hashes = {name: sha(root / name) for name in files}
    protocol, quality, proof, diagnostics = [read(root / name) for name in files]
    assert quality['state'] == proof['state'] == diagnostics['state'] == 'complete'
    assert quality['protocol_sha256'] == proof['protocol_sha256'] == hashes['protocol.json']
    assert diagnostics['quality_report_sha256'] == hashes['quality-report.json']
    entries = {e['model'] + '-' + e['label']: e for e in protocol['evaluations']}
    assert len(entries) == len(protocol['evaluations'])
    requested = [pair.split(':') for pair in pairs]
    assert all(len(pair) == 2 and pair[0] != pair[1] for pair in requested)
    for baseline, candidate in requested:
        a, b = read(entries[baseline]['plan']), read(entries[candidate]['plan'])
        assert a.get('policy_sampling_seed_offset', 0) == b.get('policy_sampling_seed_offset', 0), 'Policy RNG arms differ'
    variants, conditions, sources = {}, {}, []
    for name in sorted({name for pair in requested for name in pair}):
        entry = entries[name]
        assert sha(entry['plan']) == entry['plan_sha256']
        plan = read(entry['plan'])
        rows, fixtures = {}, {}
        for index, task in enumerate(plan['tasks']):
            for seed, fixture in zip(task['seeds'], task['instances'], strict=True):
                key = (task['episode']['id'], seed)
                assert key not in rows
                folder = pathlib.Path(entry['root']) / f"case-{index}-{task['modes'][0]}" / f's-{seed}'
                path = folder / 'report.json'
                digest = sha(path)
                assert proof['members'][str(folder)]['report_sha256'] == digest
                capture = read(path)
                assert capture['capture_complete'] and capture['provenance_valid'] and len(capture['results']) == 1
                result = capture['results'][0]
                assert result['seed'] == seed and result['generated_fixture'] == fixture
                assert result['generated_start']['confirmed'] and result['dataset']['command_proof']['accepted']
                life = result['first_life']
                damage = life['damage']
                rows[key] = dict(win=(result.get('goal_stop') or {}).get('reason') == 'combat_goal_complete',
                    death=life['end_reason'] == 'first_observed_death',
                    received_damage=damage['received_health_damage'],
                    outgoing_damage=sum(v.get('monster_health_damage', 0) for v in damage.get('by_mod', [])))
                fixtures[key] = fixture
                sources.append(dict(variant=name, path=str(path), sha256=digest))
        assert len(rows) == protocol['episodes_per_model']
        summary = next(v for v in quality['variants'] if v['variant'] == name)
        assert sum(r['win'] for r in rows.values()) == summary['wins']
        assert sum(r['death'] for r in rows.values()) == summary['deaths']
        variants[name], conditions[name] = rows, fixtures
    comparisons = []
    for baseline, candidate in requested:
        before, after = variants[baseline], variants[candidate]
        assert before.keys() == after.keys() and conditions[baseline] == conditions[candidate], 'Conditions differ'
        differences = []
        for key in sorted(before):
            a, b = before[key], after[key]
            differences.append(dict(family=key[0], seed=key[1],
                win_delta=int(b['win']) - int(a['win']), death_delta=int(b['death']) - int(a['death']),
                received_damage_delta=b['received_damage'] - a['received_damage'],
                outgoing_damage_delta=b['outgoing_damage'] - a['outgoing_damage']))
        comparisons.append(dict(baseline=baseline, candidate=candidate, episodes=len(differences),
            gained_wins=sum(d['win_delta'] == 1 for d in differences), lost_wins=sum(d['win_delta'] == -1 for d in differences),
            win_delta=sum(d['win_delta'] for d in differences), death_delta=sum(d['death_delta'] for d in differences),
            mean_received_damage_delta=sum(d['received_damage_delta'] for d in differences) / len(differences),
            mean_outgoing_damage_delta=sum(d['outgoing_damage_delta'] for d in differences) / len(differences),
            episodes_detail=differences))
    assert hashes == {name: sha(root / name) for name in files}
    scope = ('Differences candidate minus baseline on exact generated fixture/engine-seed and policy RNG offset matches. '
        'Each RNG arm reported separately; arms sharing engine conditions are not independent sample doubling. '
        'Descriptive development comparison, not causal attribution to individual actions or untouched test. No neural replay.')
    save(root / 'paired-effects.json', dict(state='complete', comparisons=comparisons, source_sha256=hashes,
        native_reports=sources, protocol_sha256=hashes['protocol.json'], scope=scope))
    lines = ['# Paired native effects', '', '| Baseline | Candidate | Gained / lost wins | Win delta | Death delta | Received damage delta |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for c in comparisons:
        lines.append(f"| {c['baseline']} | {c['candidate']} | {c['gained_wins']} / {c['lost_wins']} | {c['win_delta']:+d} | {c['death_delta']:+d} | {c['mean_received_damage_delta']:+.2f} |")
    (root / 'paired-effects.md').write_text('\n'.join(lines + ['', scope, '']), encoding='utf-8')
    return [{k: v for k, v in c.items() if k != 'episodes_detail'} for c in comparisons]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--pairs', nargs='+', required=True)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.wait_pid:
            wait_process(args.wait_pid, root / 'paired-effects-progress.json')
        print(report(root, args.pairs), flush=True)
        save(root / 'paired-effects-progress.json', dict(state='complete', report_sha256=sha(root / 'paired-effects.json')))
    except Exception as error:
        save(root / 'paired-effects-progress.json', dict(state='failed', error=str(error)))
        raise

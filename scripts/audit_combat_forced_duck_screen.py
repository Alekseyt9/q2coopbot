"""Paired development quality screen; all neural replay stays on CUDA."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from combat_teacher_ownership import rules_ownership
from compare_registered_combat_pools import runtime_bindings
from process_combat_architecture_pool import read, save, sha


def run(protocol, root, out):
    assert not out.exists()
    spec, pool = read(protocol), read(root/'report.json')
    assert spec['version'] == 'combat_forced_duck_screen_v1'
    assert pool['state'] == 'complete' and pool['source_unchanged'] and len(pool['jobs']) == 24
    assert sha(protocol) == pool['protocol_sha256']
    pins = {**spec['source_files'], str(protocol):sha(protocol), str(Path(__file__).resolve()):sha(Path(__file__))}
    assert all(sha(Path(p)) == digest for p, digest in pins.items())
    model = Path(spec['model'])
    assert sha(model) == spec['model_sha256']
    out.mkdir(parents=True)
    shutil.copyfile(__file__, out/Path(__file__).name)
    env = dict(os.environ, GOCACHE=str(Path.cwd()/'workspace/build/gocache'))

    def command(label, cmd):
        with (out/(label+'.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(list(map(str, cmd)), env=env, stdout=log, stderr=log)
        assert result.returncode == 0, label

    exporter = (out/'q2ppo-data.exe').resolve()
    command('build', ['go', 'build', '-o', exporter, './cmd/q2ppo-data'])
    groups = {k:[] for k in ('control', 'request', 'rules')}
    bindings, tasks = {}, []
    for job in sorted(pool['jobs'], key=lambda j:j['job']):
        assert not job['error']
        folder, label = Path(job['root']), job['label']
        cap, manifest = read(folder/'report.json'), read(folder/'manifest.json')
        assert cap['capture_complete'] and cap['provenance_valid']
        row = cap['results'][0]
        assert row['capture_valid'] and row['dispatch_valid'] and row['seed_confirmed']
        case = next(c for c in spec['cases'] if c['name'] == job['case_name'])
        assert row['generated_fixture'] == case['fixture'] and row['seed'] == case['seed']
        binding = dict(files=runtime_bindings(row),
            source=manifest['source_fingerprint'], client=manifest['client_sha256'], exporter=manifest['exporter_sha256'],
            reward=manifest['reward_config_sha256'], native=manifest['native_source_fingerprint'],
            environment={k:manifest[k] for k in ('timescale','skill','release_game_frame','fixed_world_hold','monster_no_infighting','standard_monster_spawn_height','post_frame_rng_reset','synchronous','stop_on_goal')})
        if job['seed'] in bindings:
            assert binding == bindings[job['seed']], 'Paired environment differs'
        bindings[job['seed']] = binding
        assert manifest.get('diagnostic_forced_duck_frame', 0) == (-1 if label == 'request' else 0)
        if label == 'rules':
            assert manifest['provider'] == 'rules' and not manifest.get('model_weights_sha256')
        else:
            assert manifest['provider'] == 'learned' and manifest['model_weights_sha256'].lower() == sha(model)
        step_path = Path(row['root'])/'dataset/steps.jsonl'
        pins[str(step_path)] = sha(step_path)
        steps = [json.loads(line) for line in step_path.open(encoding='utf-8-sig')]
        goal, end = row.get('goal_stop'), row['first_life']['end_frame']
        active = [s for s in steps if s['observation']['identity']['life'] == 1 and s['observation']['health'] > 0
                  and s['observation']['identity']['frame'] <= (goal['kill_frame'] if goal else end)]
        ownership = rules_ownership(active, goal, end)
        if label == 'rules':
            assert all(s['owner'] == 'rules' and not s.get('sample') for s in active)
        damage = row['first_life']['damage']
        record = dict(map=case['name'], seed=job['seed'], goal=bool(goal),
            death=bool(row['death_stop'] or row['first_life']['minimum_health'] <= 0),
            kills=sum(m['monster_kills'] for m in damage['by_mod']),
            dealt=sum(m['monster_health_damage'] for m in damage['by_mod']), received=damage['received_health_damage'],
            rules_combat_rows=ownership['rules_combat_rows'],
            forced_duck_events=0)
        trace_path = Path(row['root'])/'bot.jsonl'
        pins[str(trace_path)] = sha(trace_path)
        for line in trace_path.open(encoding='utf-8-sig'):
            c = json.loads(line).get('combat_policy') or {}
            tags = [v for v in c.get('selection', {}).get('interventions', []) or [] if v['reason'] == 'diagnostic_forced_duck_request']
            if tags:
                assert label == 'request' and len(tags) == 1
                assert c['observation']['ducked'] and c['observation']['on_ground']
                assert c['proposed_command']['Up'] == c['applied_command']['Up'] == 0
                assert all(c['proposed_command'][k] == c['applied_command'][k] for k in ('Forward','Side'))
                record['forced_duck_events'] += 1
        record['autonomous_goal'] = record['goal'] and not record['death'] and (label == 'rules' or ownership['rules_combat_rows'] == 0)
        groups[label].append(record)
        if label != 'rules':
            directory = (out/f"case-{job['job']}").resolve()
            directory.mkdir()
            command(f"export-{job['job']}", [exporter, '--batch', folder, '--model', model, '--cuda-only', '--out', directory/'native'])
            tasks.append(dict(model=str(model), model_sha256=sha(model), data=str(directory/'native'),
                out=str(directory/'cuda'), native_report_sha256=sha(directory/'native/report.json')))
    totals = {k:{f:sum(r[f] for r in values) for f in ('goal','autonomous_goal','death','kills','dealt','received','forced_duck_events')} for k, values in groups.items()}
    save(out/'native-results.json', dict(totals=totals, cases=groups))
    save(out/'cuda-request.json', dict(version='combat_cuda_finalize_batch_v1',compact_finalized_streams=True,tasks=tasks))
    command('cuda', [sys.executable, 'scripts/finalize_combat_cuda_batch.py', '--request', out/'cuda-request.json', '--receipt-root', out/'cuda-batch'])
    assert len(tasks) == 16
    for task in tasks:
        meta, proof = read(Path(task['out'])/'report.json'), read(Path(task['out'])/'cuda-verification.json')
        assert proof['state'] == 'passed' and proof['device'] == 'cuda'
        if meta.get('diagnostic_forced_duck_frame'):
            assert meta['version'] == 'combat_diagnostic_forced_duck_rollout_v1'
            assert meta['diagnostic_only'] and not meta['ppo_training_ready'] and not proof['ordinary_training_eligible']
    old, new = {r['seed']:r for r in groups['control']}, {r['seed']:r for r in groups['request']}
    lost = [s for s in old if old[s]['autonomous_goal'] and not new[s]['autonomous_goal']]
    gate = totals['request']['autonomous_goal'] > totals['control']['autonomous_goal'] and not lost and totals['request']['death'] <= totals['control']['death']
    assert all(sha(Path(p)) == digest for p,digest in pins.items())
    save(out/'report.json', dict(state='complete', totals=totals, cases=groups, lost_source_goals=lost,
        advance=gate, promotion=False, inputs=pins, scope=spec['scope'], native_cases=24, cuda_cases=16))
    save(out/'complete.json', dict(state='complete', report_sha256=sha(out/'report.json')))
    print(json.dumps(dict(totals=totals, advance=gate), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for k in ('protocol','root','out'):
        parser.add_argument('--'+k,required=True,type=Path)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.root.resolve(), args.out.resolve())

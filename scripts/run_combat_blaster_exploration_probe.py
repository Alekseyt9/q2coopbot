"""Native/CUDA diagnosis of source, metadata-zero and blaster exploration profiles."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from evaluate_registered_combat_models import read, save, sha
from audit_combat_checked_ground_native_probe import normalized
from combat_equipped_weapon_features import equipped_weapon_index, CATEGORIES


def behavior(step, score=True, following=True):
    sample = dict(step['sample'])
    sample.pop('version', None)
    if not score:
        sample.pop('log_probability', None)
    result = dict(observation=step['observation'], action=step['action'], sent_command=step['sent_command'], sample=sample)
    if following:
        result.update(applied_action=step['applied_action'], next_observation=step.get('next_observation'))
    return normalized(result)


def summarize(root, protocol, prepared, out=None):
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    pin(__file__)
    pin(Path(__file__).with_name('combat_equipped_weapon_features.py'))
    pool = read(pin(root/'pool/report.json'))
    assert pool['state'] == 'complete' and pool['source_unchanged'] and pool['usable_captures'] == 6
    rows, details = {}, []
    for i, variant in enumerate(prepared['variants']):
        pin(variant['model'], variant['model_sha256'])
        for repeat in range(2):
            index = 2*i+repeat
            job = next(j for j in pool['jobs'] if j['plan_index'] == index)
            assert not job['error'] and job['seed'] == protocol['context']['seed']
            capture = read(pin(Path(job['root'])/'report.json'))
            assert capture['capture_complete'] and capture['provenance_valid']
            result = capture['results'][0]
            assert result['generated_fixture'] == protocol['context']['instance']
            processing = root/f'processing-{index:02d}'
            processed = read(pin(processing/'report.json'))
            assert processed['state'] == 'complete' and processed['device'] == 'cuda' and len(processed['cases']) == 1
            folder = processing/f"case-{job['job']:04d}"
            meta = read(pin(folder/'cuda/report.json'))
            assert meta['numerical_verification'] == 'cuda_verified_v1' and meta['model_sha256'] == variant['model_sha256']
            proof = read(pin(folder/'cuda/cuda-verification.json', meta['cuda_verification_sha256']))
            assert proof['state'] == 'passed' and proof['device'] == 'cuda'
            path = next((folder/'native').glob('replay-*/steps.jsonl')).resolve()
            with pin(path, meta['source_sha256'][str(path)]).open(encoding='utf-8-sig') as stream:
                steps = [json.loads(line) for line in stream]
            active = [s for s in steps if s['owner'] == 'provider' and s.get('sample') and s.get('native_step')
                and s['observation']['identity']['life'] == 1 and s['observation']['health'] > 0]
            rows[variant['label'], repeat] = active
            requests = [s for s in active if s['action'].get('weapon')]
            shotgun = [s for s in active if equipped_weapon_index(s['observation']) == 2]
            damage = result['first_life']['damage']
            details.append(dict(label=variant['label'], repeat=repeat, seed=job['seed'],
                goal=bool(result['goal_stop'] and result['goal_stop']['health'] > 0),
                death=bool(result['death_stop']), death_frame=(result['death_stop'] or {}).get('death_frame'),
                kills=sum(x['monster_kills'] for x in damage['by_mod']),
                dealt=sum(x['monster_health_damage'] for x in damage['by_mod']), received=damage['received_health_damage'],
                first_shotgun_request=next((s['observation']['identity']['frame'] for s in requests if s['action']['weapon'] == 'Shotgun'), None),
                first_shotgun_equipped=shotgun[0]['observation']['identity']['frame'] if shotgun else None,
                requests=dict(Counter(s['action']['weapon'] for s in requests)),
                equipped_frames=dict(Counter(CATEGORIES[equipped_weapon_index(s['observation'])] for s in active)),
                raw_equipped_frames=dict(Counter(s['observation']['weapon'] for s in active))))
    checks = {}
    for label in ('source', 'zero', 'blaster4'):
        a, b = rows[label, 0], rows[label, 1]
        checks[label+'/cold_repeat_exact'] = len(a) == len(b) and all(behavior(x) == behavior(y) for x, y in zip(a, b))
    for repeat in range(2):
        a, b = rows['source', repeat], rows['zero', repeat]
        checks[f'zero/source/{repeat}/full_exact'] = len(a) == len(b) and all(behavior(x) == behavior(y) for x, y in zip(a, b))
    original_path = pin(protocol['context']['original_trace'], protocol['context']['original_trace_sha256'])
    original = {}
    with original_path.open(encoding='utf-8-sig') as stream:
        for raw in map(json.loads, stream):
            c = raw.get('combat_policy') or {}
            sample = (c.get('selection') or {}).get('sample')
            observation = c.get('observation') or {}
            if sample and observation['identity']['life'] == 1 and observation['health'] > 0:
                frame = observation['identity']['frame']
                assert frame not in original
                original[frame] = dict(observation=observation, action=c['proposed'], sent_command=raw['sent_command'], sample=sample)
    checks['source/historical_behavior_exact'] = all(
        (frame := s['observation']['identity']['frame']) in original and
        behavior(s, following=False) == behavior(original[frame], following=False) for s in rows['source', 0])
    divergences = []
    for repeat in range(2):
        source = {s['observation']['identity']['frame']: s for s in rows['source', repeat]}
        candidate = {s['observation']['identity']['frame']: s for s in rows['blaster4', repeat]}
        shared = sorted(source.keys() & candidate.keys())
        changed = next((f for f in shared if source[f]['sample'].get('weapon', 0) != candidate[f]['sample'].get('weapon', 0)), None)
        prefix = [f for f in shared if changed is None or f < changed]
        # Calibration changes log probability before a sampled action changes.
        # Preserve all behavior, memory and other samples; omit that score only.
        checks[f'candidate/{repeat}/prefix_before_weapon_selection'] = all(
            behavior(source[f], score=False, following=False) == behavior(candidate[f], score=False, following=False) for f in prefix)
        checks[f'candidate/{repeat}/first_selection_state_exact'] = changed is None or normalized(
            source[changed]['observation']) == normalized(candidate[changed]['observation'])
        divergences.append(dict(repeat=repeat, first_changed_weapon_selection=changed))
    for index in range(6):
        ownership = read(pin(root/f'ownership-{index:02d}.json'))
        assert len(ownership['cases']) == 1 and ownership['pool_sha256'] == sha(root/'pool/report.json')
        checks[f'model_ownership/{index}'] = all(ownership['totals'][key] == 0 for key in
            ('visible_enemy_rules_frames', 'visible_enemy_rules_attack_frames', 'visible_rules_missing_applied_frames', 'unknown_observation_rows'))
    assert all(sha(path) == digest for path, digest in pins.items())
    output = root if out is None else out
    if out is not None:
        assert not out.exists()
        out.mkdir()
    save(output/'report.json', dict(state='complete', version='combat_blaster_exploration_probe_result_v2_observed_aliases',
        technical_gate_passed=all(checks.values()), checks=checks, details=details, divergences=divergences, inputs=pins,
        original_report_sha256=sha(root/'report.json') if out is not None else None,
        equipped_name_normalization='existing_v10_observed_weapon_aliases',
        native_cases=6, cuda_cases=6, training_performed=False, promotion=False,
        scope=protocol['scope']+' Spawncount alone normalized; entity IDs retained. Candidate prefix excludes log-probability score, not sampled behavior. '
              'One intentionally reused TRAIN scene with cold repeats; no independent-quality or optimal-weapon claim.'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'protocol', 'registry', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    a = ap.parse_args()
    repo, root = Path(__file__).resolve().parents[1], a.out.resolve()
    assert not root.exists() and shutil.disk_usage('F:/').free >= 32*1024**3
    protocol = read(a.protocol)
    seal = read(a.prepared/'complete.json')
    assert seal['report_sha256'] == sha(a.prepared/'report.json')
    prepared = read(a.prepared/'report.json')
    assert prepared['device'] == 'cuda' and prepared['protocol_sha256'] == sha(a.protocol)
    assert prepared['zero_control_exact'] and prepared['actor_value_parameters_exact'] and prepared['legacy_grenade_gate_exact']
    assert len(prepared['variants']) == 3 and protocol['repeats'] == 2 and protocol['native_cases'] == protocol['cuda_cases'] == 6
    assert sha(protocol['prerequisite_report']) == protocol['prerequisite_report_sha256']
    paths = {a.protocol.resolve(), (a.prepared/'report.json').resolve(), (a.prepared/'complete.json').resolve()}
    for folder, suffixes in ((repo/'scripts', ('.py', '.ps1')), (repo/'cmd', ('.go',)),
            (repo/'internal', ('.go',)), (repo.parent/'yquake2/src', ('.c', '.h'))):
        paths.update(p.resolve() for p in folder.rglob('*') if p.is_file() and p.suffix in suffixes)
    fence = {str(p): sha(p) for p in sorted(paths)}
    for path, digest in prepared['inputs'].items():
        assert sha(path) == digest
        fence[path] = digest
    registry = a.registry.resolve()
    fence[str(registry)] = sha(registry)
    descriptors = []
    for filename in read(registry)['files']:
        path = registry.parent/filename
        fence[str(path)] = sha(path)
        descriptors.append(read(path))
    episode = next(e for e in descriptors if e['id'] == protocol['context']['episode'])
    original_capture = Path(protocol['context']['original_capture'])
    original_manifest = original_capture.parent/'manifest.json'
    reward = (repo/episode['recipe']['reward_config']).resolve()
    assert sha(reward) == read(original_manifest)['reward_config_sha256'].lower()
    for path in (original_capture, original_manifest, reward):
        fence[str(path)] = sha(path)
    for variant in prepared['variants']:
        assert sha(variant['model']) == variant['model_sha256']
        fence[variant['model']] = variant['model_sha256']
    root.mkdir()
    save(root/'run.json', dict(source_files=fence, native_cases=6, cuda_cases=6, training_performed=False, promotion=False))
    env = dict(os.environ, GOCACHE=str(repo/'workspace/build/gocache'), GOTOOLCHAIN='auto', PYTHONIOENCODING='utf-8')

    def run(label, values):
        assert all(sha(p) == h for p, h in fence.items()), 'Source/input changed'
        with (root/(label+'.log')).open('w', encoding='utf-8') as log:
            child = subprocess.Popen(list(map(str, values)), cwd=repo, env=env, stdout=log, stderr=log,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            save(root/'progress.json', dict(stage=label, runner_pid=os.getpid(), child_pid=child.pid))
            assert child.wait() == 0, str(root/(label+'.log'))

    try:
        planner = root/'q2episode.exe'
        run('build-planner', ['go', 'build', '-o', planner, './cmd/q2episode'])
        context = protocol['context']
        plans = []
        for i, variant in enumerate(prepared['variants']):
            for repeat in range(2):
                index = 2*i+repeat
                path = root/f'plan-{index:02d}.json'
                run(f'plan-{index:02d}', [planner, '--registry', registry, '--root', repo, '--episodes', context['episode'],
                    '--split', 'train', '--mode', 'learned', '--model', variant['model'], '--count', 1,
                    '--seed-offset', context['offset'], '--checked-wall-slide', '--out', path, '--artifacts', root/f'captures-{index:02d}'])
                plan = read(path)
                assert len(plan['tasks']) == 1 and plan['tasks'][0]['instances'] == [context['instance']]
                assert plan['tasks'][0]['seeds'] == [context['seed']]
                plans.append(path)
        quote = lambda p: "'"+str(p).replace("'", "''")+"'"
        command = '& '+quote(repo/'scripts/run_registered_combat_pool.ps1')+' -Plans @('+','.join(map(quote, plans))+') -MaxInstances 16 -Port '+str(protocol['port'])+' -OutputRoot '+quote(root/'pool')
        run('native-pool', [shutil.which('pwsh'), '-NoProfile', '-Command', command])
        pool = read(root/'pool/report.json')
        assert pool['state'] == 'complete' and pool['source_unchanged'] and pool['usable_captures'] == 6
        for index in range(6):
            run(f'cuda-{index:02d}', [sys.executable, repo/'scripts/process_registered_combat_pool_cuda.py',
                '--pool', root/'pool', '--plan-index', index, '--out', root/f'processing-{index:02d}'])
            run(f'ownership-{index:02d}', [sys.executable, repo/'scripts/audit_registered_combat_ownership.py',
                '--pool', root/'pool', '--plan-index', index, '--out', root/f'ownership-{index:02d}.json'])
        save(root/'progress.json', dict(stage='comparison', runner_pid=os.getpid(), child_pid=None))
        summarize(root, protocol, prepared)
        assert all(sha(p) == h for p, h in fence.items()), 'Source/input changed'
        save(root/'complete.json', dict(state='complete', report_sha256=sha(root/'report.json'),
            native_cases=6, cuda_cases=6, source_unchanged=True, training_performed=False, promotion=False))
        save(root/'progress.json', dict(stage='complete', runner_pid=os.getpid(), child_pid=None))
    except BaseException as error:
        save(root/'progress.json', dict(stage='failed', runner_pid=os.getpid(), error=str(error)))
        raise


if __name__ == '__main__':
    main()

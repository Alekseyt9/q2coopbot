"""Audit a single-frame native rocket intervention. No neural inference or fitting."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def normalize(value):
    if isinstance(value, dict):
        return {key: 0 if key == 'spawncount' else normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


def signature(capture):
    return normalize({key: capture[key] for key in
                      ('observation', 'proposed', 'proposed_command', 'applied_command')} |
                     {'sample': capture['selection'].get('sample')})


def run(protocol, root, out):
    assert not out.exists(), 'Fresh audit output required'
    spec, native = read(protocol), read(root / 'report.json')
    assert spec['version'] == 'combat_rocket_surface_native_probe_v1'
    assert native['state'] == 'complete' and native['source_unchanged']
    assert native['protocol_sha256'] == sha(protocol)
    assert len(native['jobs']) == spec['native_games'] == len(spec['cases']) * 8
    pins = {str(protocol): sha(protocol), str(root / 'report.json'): sha(root / 'report.json'),
            str(Path(__file__).resolve()): sha(__file__), **spec['inputs']}
    assert all(sha(path) == digest for path, digest in pins.items())
    checks, details, traces = {}, [], {}
    cases = {case['name']: case for case in spec['cases']}
    for job in sorted(native['jobs'], key=lambda item: item['job']):
        assert not job['error']
        jobroot = Path(job['root'])
        report, manifest = read(jobroot / 'report.json'), read(jobroot / 'manifest.json')
        result = report['results'][0]
        assert report['capture_complete'] and report['provenance_valid']
        assert result['dispatch_valid'] and result['capture_valid'] and result['seed_confirmed']
        case, label, repeat = cases[job['case_name']], job['label'], job['repeat']
        frame = case['frame']
        assert manifest.get('diagnostic_rocket_aim_frame', 0) == (0 if label == 'source' else frame)
        path = Path(result['root']) / 'bot.jsonl'
        pins[str(path)] = sha(path)
        values = {}
        for line in path.open(encoding='utf-8-sig'):
            capture = json.loads(line).get('combat_policy') or {}
            observation = capture.get('observation') or {}
            identity = observation.get('identity') or {}
            # Stop at the native first-life goal/death boundary. Process teardown
            # can capture a different number of later, already completed ticks.
            if (identity.get('life') == 1 and observation.get('health', 0) > 0
                    and identity['frame'] <= result['first_life']['end_frame']):
                current = identity['frame']
                assert current not in values, 'Duplicate first-life observation'
                values[current] = capture
        traces[job['case_name'], label, repeat] = values
        current, following = values[frame], values[frame + 1]
        key = f'{job["case_name"]}/{label}/{repeat}'
        tags = [i for i in current['selection'].get('interventions', []) or []
                if i['reason'] == 'diagnostic_rocket_surface_aim']
        checks[key + '/intervention_count'] = len(tags) == (0 if label == 'source' else 2)
        checks[key + '/real_rocket_consumption'] = (
            current['observation']['ammo'] - following['observation']['ammo'] == 1
            and current['applied_command']['Buttons'] & 1 == 1
            and following['observation']['gun_frame'] == 6)
        unexpected = [f for f, c in values.items() if f != frame and any(
            i['reason'] == 'diagnostic_rocket_surface_aim'
            for i in c['selection'].get('interventions', []) or [])]
        checks[key + '/only_root_intervention'] = not unexpected
        first, goal = result['first_life'], result.get('goal_stop') or {}
        # Fixed six-tick horizon ends before another ammo decrement. This is
        # separate from the full fight, whose later decisions can adapt.
        window_end = frame + 6
        checks[key + '/single_rocket_window'] = all(
            f in values and values[f]['observation']['ammo'] == following['observation']['ammo']
            for f in range(frame + 1, window_end + 1))
        log = Path(result['root']) / 'server.log'
        pins[str(log)] = sha(log)
        early = dict(monster_health_damage=0, monster_kills=0, self_health_damage=0,
                     direct_health_damage=0, splash_health_damage=0)
        for line in log.read_text(encoding='utf-8-sig').splitlines():
            if not line.startswith('sv_test_damage '):
                continue
            event = dict(re.findall(r'(\w+)=([^\s]+)', line))
            if not (frame < int(event['server_frame']) <= window_end
                    and int(event['attacker']) == first['bot_entity'] and int(event['mod']) in (8, 9)):
                continue
            damage = max(0, int(event['health_before'])) - max(0, int(event['health_after']))
            if event['target_class'].startswith('monster_'):
                early['monster_health_damage'] += damage
                early['monster_kills'] += int(event['health_before']) > 0 >= int(event['health_after'])
                early['direct_health_damage' if event['mod'] == '8' else 'splash_health_damage'] += damage
            if int(event['target']) == first['bot_entity']:
                early['self_health_damage'] += damage
        details.append(dict(case=job['case_name'], label=label, repeat=repeat,
                            goal=goal.get('reason') == 'combat_goal_complete',
                            end_frame=first['end_frame'], health=first['minimum_health'],
                            received=first['damage']['received_health_damage'],
                            self_damage=first['damage']['self_health_damage'],
                            damage_by_mod=first['damage']['by_mod'],
                            first_rocket_window=dict(start_exclusive=frame, end_inclusive=window_end, **early)))
    for name, case in cases.items():
        frame = case['frame']
        source = traces[name, 'source', 0]
        for label in ('source', 'body', 'floor', 'wall'):
            left, right = traces[name, label, 0], traces[name, label, 1]
            checks[f'{name}/{label}/cold_repeat_exact'] = (
                left.keys() == right.keys() and all(signature(left[f]) == signature(right[f]) for f in left))
            for repeat in (0, 1):
                current = traces[name, label, repeat]
                prefix = [f for f in source if f < frame]
                key = f'{name}/{label}/{repeat}'
                checks[key + '/exact_prefix'] = bool(prefix) and all(
                    f in current and signature(source[f]) == signature(current[f]) for f in prefix)
                checks[key + '/exact_root_observation_sample'] = all(
                    normalize(current[frame][field]) == normalize(source[frame][field])
                    for field in ('observation', 'proposed', 'proposed_command')) and (
                    normalize(current[frame]['selection'].get('sample')) ==
                    normalize(source[frame]['selection'].get('sample')))
                a, b = current[frame]['applied_command'], source[frame]['applied_command']
                checks[key + '/non_angle_command_exact'] = all(
                    a[k] == b[k] for k in b if k not in ('Pitch', 'Yaw'))
    assert all(sha(path) == digest for path, digest in pins.items()), 'Audit input changed'
    out.mkdir(parents=True)
    report = dict(version='combat_rocket_surface_native_audit_v1', state='complete',
                  mechanical_gate_passed=all(checks.values()), checks=checks, details=details,
                  native_games=len(native['jobs']), independent_worlds=len(cases), inputs=pins,
                  training_performed=False, ordinary_ppo_eligible=False, promotion=False,
                  scope='One root aim intervention per run. Cold repeats are not independent worlds. '
                        'Whole-battle results include later policy adaptation, not isolated first-rocket effects.')
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(mechanical_gate_passed=report['mechanical_gate_passed'],
                          failed=[key for key, passed in checks.items() if not passed],
                          details=details), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('protocol', 'root', 'out'):
        parser.add_argument('--' + field, type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.root.resolve(), args.out.resolve())

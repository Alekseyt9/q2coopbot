"""Describe target switching and per-enemy native damage in a closed evaluation."""
import argparse
import collections
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha
from native_combat_release import release_boundary


def previous_target_still_visible(observation):
    previous = observation.get('previous_target_intent')
    if not previous:
        return False
    before, now = previous['identity'], observation['identity']
    if before['frame'] + 1 != now['frame'] or any(before[k] != now[k] for k in ('map', 'connection', 'spawncount', 'actor', 'life')):
        return False
    enemies = sorted(observation['enemies'], key=lambda e: e['distance'])[:8]
    return any(e['id'] == previous['entity'] and e.get('observed_track') == previous['track']
               and e.get('clear_shot') is True for e in enemies)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), 'Fresh output required'
    root, pins = args.evaluation.resolve(), {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    pin(__file__)
    closed = read(pin(root / 'complete.json'))
    assert closed['state'] == 'complete' and not closed['training_performed']
    fixture = read(pin(root / 'fixture-proof.json'))
    assert fixture['spec_sha256'] == closed['spec_sha256']
    pool = read(pin(root / 'pool/report.json', closed['pool_sha256']))
    assert pool['state'] == 'complete' and pool['source_unchanged']
    labels = {i: Path(p['path']).stem.removesuffix('-plan') for i, p in enumerate(fixture['plans'])}
    cases = []
    for job in sorted(pool['jobs'], key=lambda j: j['job']):
        if job['mode'] != 'learned':
            continue
        label = labels[job['plan_index']]
        folder = root / ('processing-' + label) / f"case-{job['job']:04d}"
        meta = read(pin(folder / 'cuda/report.json'))
        verification = read(pin(folder / 'cuda/cuda-verification.json', meta['cuda_verification_sha256']))
        assert verification['state'] == 'passed' and verification['device'] == 'cuda'
        capture = read(pin(Path(job['root']) / 'report.json'))
        result = capture['results'][0]
        assert capture['capture_complete'] and capture['provenance_valid'] and result['dispatch_valid']
        manifest = read(pin(Path(job['root']) / 'manifest.json'))
        log = pin(Path(result['root']) / 'server.log')
        with log.open(encoding='utf-8-sig') as stream:
            _, release, _ = release_boundary(stream, manifest, job['seed'])
        steps_path = next((folder / 'native').glob('replay-*/steps.jsonl'))
        effects_path = steps_path.with_name('server_outcomes.jsonl')
        for path in (steps_path, effects_path):
            pin(path, meta['source_sha256'][str(path.resolve())])
        import json
        with steps_path.open(encoding='utf-8-sig') as stream:
            steps = list(map(json.loads, stream))
        with effects_path.open(encoding='utf-8-sig') as stream:
            effects = {e['step']: e for e in map(json.loads, stream)}
        stop = result['goal_stop']['kill_frame'] if result['goal_stop'] else result['first_life']['end_frame']
        counts, ledger, damaged, first_kill_classes = collections.Counter(), {}, set(), None
        for step in steps:
            o = step['observation']
            if not (step['owner'] == 'provider' and step.get('sample') and step.get('native_step')
                    and o['identity']['life'] == 1 and o['health'] > 0 and release <= o['identity']['frame'] < stop):
                continue
            execution = step.get('server_execution') or {}
            assert execution.get('matched') and execution.get('window_exclusive') and not execution.get('recovery_commands')
            counts['provider_frames'] += 1
            previous_valid = previous_target_still_visible(o)
            counts['previous_target_still_visible_frames'] += previous_valid
            selected = step['action'].get('target_entity', 0)
            if previous_valid and selected != o['previous_target_intent']['entity']:
                counts['switched_while_previous_target_still_visible'] += 1
                counts['switched_to_none' if selected == 0 else 'switched_to_another_target'] += 1
            effect = effects.get(step['index'])
            if not effect or not effect['available']:
                continue
            for event in effect['events']:
                if event['attacker'] != step['native_step']['actor'] or not event['target_class'].startswith('monster_'):
                    continue
                key = str(event['target'])
                entry = ledger.setdefault(key, dict(entity=event['target'], kind=event['target_class'], health_damage=0,
                    damage_events=0, first_hit_frame=event['frame'], last_hit_frame=event['frame'],
                    first_recorded_health_before=event['health_before'], last_recorded_hit_health_after=event['health_after'], kills=0))
                assert entry['kind'] == event['target_class']
                damage = max(0, event['health_before']) - max(0, event['health_after'])
                assert damage >= 0
                entry['health_damage'] += damage
                entry['damage_events'] += 1
                entry['last_hit_frame'] = event['frame']
                entry['last_recorded_hit_health_after'] = event['health_after']
                if damage:
                    damaged.add(key)
                killed = event['health_before'] > 0 and event['health_after'] <= 0
                entry['kills'] += killed
                if killed and first_kill_classes is None:
                    first_kill_classes = sorted(ledger[k]['kind'] for k in damaged)
        cases.append(dict(label=label, map=result['generated_fixture']['map'], episode=result['generated_fixture']['episode_id'],
                          seed=job['seed'], counts=dict(counts), native_damage_by_enemy=list(ledger.values()),
                          own_damaged_enemies_before_first_kill=first_kill_classes,
                          own_damaged_enemy_count=len(damaged),
                          own_provider_kills=sum(e['kills'] for e in ledger.values()),
                          own_provider_health_damage=sum(e['health_damage'] for e in ledger.values()),
                          completed_goal=bool(result['goal_stop']), death=bool(result['death_stop'])))
    for path, digest in pins.items():
        assert sha(path) == digest, path
    args.out.mkdir(parents=True)
    save(args.out / 'report.json', dict(state='complete', version='combat_native_damage_distribution_v1', cases=cases,
        inputs=pins, neural_execution=False, new_games=False, training_performed=False, promotion=False,
        scope='Post-release first-life provider windows only. Native health is offline diagnostic, never policy input. '
              'Last hit HP is not guaranteed end-of-fight HP. Switching with the prior enemy visible can be justified; '
              'distribution across enemies does not prove a bad decision. Entity IDs are local to each case.'))
    save(args.out / 'complete.json', dict(state='complete', report_sha256=sha(args.out / 'report.json')))
    print(dict(cases=len(cases), maps=sorted({c['map'] for c in cases})))


if __name__ == '__main__':
    main()

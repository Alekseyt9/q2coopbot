"""Describe incoming damage and pre-command movement on accepted own-policy TRAIN rows."""
import argparse
import collections
import json
import math
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workflow', type=Path, required=True)
    ap.add_argument('--arm', choices=('ablated', 'observed'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    assert not a.out.exists()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = pins.get(str(path)) or sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    def lines(path):
        with path.open(encoding='utf-8-sig') as stream:
            return [json.loads(line) for line in stream]

    def sealed(path):
        seal = read(pin(path / 'complete.json'))
        assert seal['state'] == 'complete'
        return read(pin(path / 'report.json', seal['report_sha256']))

    pin(__file__)
    root = a.workflow.resolve()
    closed = read(pin(root / 'complete.json'))
    assert closed['state'] == 'complete' and closed['source_unchanged']
    assert closed['native_train_cases'] == closed['cuda_train_cases'] == 192
    groups, sources = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    cases, rounds, used_seeds = [], [], set()
    expected_total = collections.Counter()
    for number in range(1, 5):
        branch = root / f'round-{number:02d}' / a.arm
        update = sealed(branch / 'update')
        assert update['device'] == 'cuda' and update['scope_updates'] == number
        credit, advantage = sealed(branch / 'native-credit'), sealed(branch / 'advantage-credit')
        assert credit['neural_execution'] is False and advantage['neural_execution'] is False
        for path, digest in credit['inputs'].items():
            pin(path, digest)
        corpus = branch / 'corpus'
        receipt = read(pin(corpus / 'complete.json'))
        protocol = read(pin(corpus / 'protocol.json', receipt['protocol_sha256']))
        assert protocol['train_only'] and all(item['split'] == 'train' for item in protocol['assignments'])
        pool = read(pin(receipt['source_pool_report'], receipt['pool_sha256']))
        assert pool['state'] == 'complete' and pool['source_unchanged']
        jobs = {job['job']: job for job in pool['jobs']}
        processing = corpus / 'processing-train'
        processed = read(pin(processing / 'report.json', receipt['processed'][0]['sha256']))
        verified = {case['job']: case for case in credit['cases']}
        assert len(processed['cases']) == len(verified) == 24
        expected_total.update({k: credit['totals'][k] for k in
            ('rows', 'received_health_damage', 'monster_health_damage', 'monster_kills', 'deaths')})
        rounds.append(dict(round=number, totals=credit['totals'], reward_components=credit['component_sums'],
            advantage_groups={name: {key: value for key, value in data.items() if key != 'component_gae_mean'}
                for name, data in advantage['groups'].items()}))
        for case in processed['cases']:
            job = jobs[case['job']]
            assert case['split'] == 'train' and job['seed'] == case['seed'] and not job['error']
            assert job['seed'] not in used_seeds
            used_seeds.add(job['seed'])
            path = processing / f"case-{job['job']:04d}"
            meta = read(pin(path / 'cuda/report.json'))
            assert meta['numerical_verification'] == 'cuda_verified_v1'
            roll = lines(pin(path / 'cuda/rollout.jsonl', meta['rollout_sha256']))
            files = list((path / 'native').glob('replay-*/steps.jsonl'))
            assert len(files) == 1
            step_path = files[0].resolve()
            outcome_path = step_path.with_name('server_outcomes.jsonl')
            steps = {s['index']: s for s in lines(pin(step_path, meta['source_sha256'][str(step_path)]))}
            outcomes = {s['step']: s for s in lines(pin(outcome_path, meta['source_sha256'][str(outcome_path)]))}
            count = collections.Counter()
            stationary_run, previous_frame = 0, None
            for row in roll:
                s, outcome = steps[row['index']], outcomes[row['index']]
                o, nxt, proposed, applied = s['observation'], s['next_observation'], s['action'], s['applied_action']
                identity = o['identity']
                assert row['seed'] == job['seed'] and row['frame'] == identity['frame']
                assert identity['life'] == 1 and s['owner'] == 'provider' and outcome['available']
                assert s['native_step']['begin_frame'] == identity['frame']
                speed = math.hypot(*o['velocity'][:2])
                stationary_run = (stationary_run + 1 if previous_frame == row['frame'] - 1 else 1) if speed <= 10 else 0
                previous_frame = row['frame']
                requested = math.hypot(proposed['forward'], proposed['side'])
                executed = math.hypot(applied['forward'], applied['side'])
                travel = math.dist(o['position'][:2], nxt['position'][:2])
                visible = o.get('enemies') or []
                parasite_range = min((e['distance'] for e in visible if e['class'] == 'monster_parasite'), default=math.inf)
                incoming = [event for event in outcome['events'] if event['target'] == identity['actor']
                    and max(event['health_before'], 0) > max(event['health_after'], 0)]
                received = sum(max(e['health_before'], 0) - max(e['health_after'], 0) for e in incoming)
                assert received == outcome['received_health_damage']
                values = dict(rows=1, received_health_damage=received,
                    monster_health_damage=outcome['monster_health_damage'], monster_kills=outcome['monster_kills'],
                    deaths=outcome['deaths'], incoming_events=len(incoming), damage_rows=int(received > 0),
                    attack_frames=int(proposed['attack']), requested_nonzero=int(requested >= .25),
                    applied_nonzero=int(executed >= .25), displacement_units=travel)
                keys = ['all', f'round:{number}', 'map:' + identity['map'],
                    'stationary_le10' if speed <= 10 else 'moving_gt10',
                    'visible_enemy' if visible else 'no_visible_enemy',
                    'has_target' if proposed.get('target_entity', 0) else 'no_target']
                if stationary_run >= 5:
                    keys.append('stationary_at_least5_consecutive_observations')
                if o['on_ground'] and visible and speed < 40:
                    keys.append('standing_grounded_visible_lt40')
                if requested >= .25 and executed < .1:
                    keys.append('large_request_applied_near_zero')
                if executed >= .25 and travel < 1:
                    keys.append('large_applied_travel_below1')
                if parasite_range < 128:
                    keys.append('visible_parasite_below128')
                if o['health'] <= 30:
                    keys.append('observed_health_at_most30')
                for key in keys:
                    groups[key].update(values)
                count.update(values)
                for event in incoming:
                    damage = max(event['health_before'], 0) - max(event['health_after'], 0)
                    known = next((e for e in visible if e['id'] == event['attacker']), None)
                    kind = event['attacker_class']
                    state = 'visible' if known else 'not_currently_visible'
                    band = ('below128' if known['distance'] < 128 else '128to256' if known['distance'] < 256 else '256plus') if known else 'unknown_current_range'
                    for key in ('all', 'class:' + kind, 'class:' + kind + '/' + state, 'range:' + band):
                        sources[key].update(events=1, received_health_damage=damage,
                            stationary_events=int(speed <= 10), standing_visible_events=int(bool(o['on_ground'] and visible and speed < 40)),
                            observed_attacker_events=int(known is not None),
                            lethal_events=int(event['health_before'] > 0 >= event['health_after']))
            assert count['rows'] == verified[job['job']]['rows']
            for key in ('received_health_damage', 'monster_health_damage', 'monster_kills', 'deaths'):
                assert count[key] == verified[job['job']]['counts'].get(key, 0), key
            cases.append(dict(round=number, job=job['job'], seed=job['seed'], counts=dict(count),
                goal=verified[job['job']]['goal'], death=verified[job['job']]['death']))
        print(dict(round=number, accepted_rows=credit['totals']['rows']), flush=True)
    assert len(cases) == len(used_seeds) == 96
    for key, value in expected_total.items():
        assert groups['all'][key] == value, key
    assert sources['all']['received_health_damage'] == groups['all']['received_health_damage']
    assert all(sha(path) == digest for path, digest in pins.items())
    a.out.mkdir(parents=True)
    report = dict(state='complete', version='combat_train_incoming_risk_v1', arm=a.arm, cases=cases,
        groups={key: dict(value) for key, value in sorted(groups.items())},
        incoming_sources={key: dict(value) for key, value in sorted(sources.items())}, rounds=rounds, inputs=pins,
        neural_execution=False, training_performed=False, new_games=False,
        scope='Accepted own-policy TRAIN windows only, reusing sealed raw-native effect/window proof and CUDA provenance. '
              'States precede commands; HP events occur within the exclusive execution window. '
              'Same-window movement need not cause or prevent an already launched attack. '
              'Groups overlap and exposures differ; no causal action ranking or new reward/model claim. '
              'Native attacker class is diagnostic only, never a new privileged policy input. No held-out data or fitting.')
    save(a.out / 'report.json', report)
    save(a.out / 'complete.json', dict(state='complete', report_sha256=sha(a.out / 'report.json')))
    print(dict(arm=a.arm, cases=len(cases), totals=dict(groups['all']), sources={k: dict(v) for k, v in sources.items() if k.startswith('class:') and '/' not in k}))


if __name__ == '__main__':
    main()

"""Replay native TRAIN captures with legacy, zero-control and proposal-cost rewards."""
import argparse
import copy
import json
import math
import subprocess
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('corpus', 'exporter', 'legacy', 'zero', 'reward', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest == expected
        pins[str(path)] = digest
        return path

    pin(__file__)
    for path in (args.exporter, args.legacy, args.zero, args.reward):
        pin(path)
    legacy, zero, reward = map(read, (args.legacy, args.zero, args.reward))
    expected = copy.deepcopy(legacy)
    expected.update(version='combat_reward_v14', blocked_ground_request=0)
    assert expected == zero
    expected['blocked_ground_request'] = -.01
    assert expected == reward
    closed = read(pin(args.corpus / 'complete.json'))
    protocol = read(pin(args.corpus / 'protocol.json', closed['protocol_sha256']))
    assert protocol['train_only']
    assignment = protocol['assignments'][0]
    assert assignment['split'] == 'train'
    plan = read(pin(assignment['plan'], assignment['plan_sha256']))
    recipes = {seed: task['episode']['id'] for task in plan['tasks'] for seed in task['seeds']}
    pool = read(pin(args.corpus / 'pool/report.json', closed['pool_sha256']))
    chosen = {}
    for job in sorted(pool['jobs'], key=lambda j: j['job']):
        condition = 'recovery' if recipes[job['seed']].endswith('-recovery') else 'ordinary'
        chosen.setdefault(condition, job)
    assert set(chosen) == {'ordinary', 'recovery'}
    args.out.mkdir(parents=True)
    cases = []
    for condition, job in sorted(chosen.items()):
        capture = read(pin(Path(job['root']) / 'report.json'))
        assert capture['capture_complete'] and capture['provenance_valid']
        result = capture['results'][0]
        root = Path(result['root'])
        paths = {name: pin(root / name) for name in ('bot.jsonl', 'server.log', 'reset-expectation.json')}
        end = 'game_frame_limit'
        tail = []
        if result['goal_stop']:
            goal = read(pin(root / 'goal-stop.json'))
            assert goal == result['goal_stop']
            end = 'combat_goal_complete'
            tail = ['--goal-observed-frame', str(goal['observed_frame']), '--goal-kill-frame', str(goal['kill_frame'])]
        if result['death_stop']:
            assert not result['goal_stop']
            receipt = pin(root / 'death-stop.json')
            assert read(receipt) == result['death_stop']
            end = 'combat_first_life_death'
            tail = ['--death-stop', str(receipt)]
        outputs = {}
        for label, config in (('legacy', args.legacy), ('zero', args.zero), ('cost', args.reward)):
            out = args.out / condition / label
            command = [str(args.exporter.resolve()), '--trace', str(paths['bot.jsonl']),
                       '--server-log', str(paths['server.log']), '--reset-expectation', str(paths['reset-expectation.json']),
                       '--reward-config', str(config.resolve()), '--out', str(out.resolve()),
                       '--worker', 'worker-' + str(result['worker']), '--episode', 'seed-' + str(job['seed']),
                       '--end-reason', end, '--client-name', 'SoloRetreatBot', '--require-execution', '--synchronous'] + tail
            out.parent.mkdir(parents=True, exist_ok=True)
            with (out.parent / (label + '.log')).open('w', encoding='utf-8') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            outputs[label] = out
        assert len({sha(path / 'steps.jsonl') for path in outputs.values()}) == 1
        data = {label: [json.loads(line) for line in (path / 'rewards.jsonl').open(encoding='utf-8-sig')]
                for label, path in outputs.items()}
        assert len({len(rows) for rows in data.values()}) == 1
        changed = 0
        total_cost = 0.0
        for before, control, after in zip(data['legacy'], data['zero'], data['cost']):
            assert before['available'] == control['available'] == after['available']
            if not before['available']:
                assert before['reason'] == control['reason'] == after['reason']
                continue
            assert before['score'] == control['score']
            cost = after['components']['blocked_ground_request']
            assert cost in (0, -.01) and control['components']['blocked_ground_request'] == 0
            for item in (control, after):
                stripped = copy.deepcopy(item)
                stripped['version'] = before['version']
                stripped['score'] = before['score']
                del stripped['components']['blocked_ground_request']
                assert stripped == before
            assert math.isclose(after['score'] - before['score'], cost, abs_tol=1e-12)
            changed += cost != 0
            total_cost += cost
        cases.append(dict(condition=condition, seed=job['seed'], rows=len(data['legacy']),
                          changed_rows=changed, total_cost=total_cost, steps_exact=True, zero_score_exact=True))
        for path in outputs.values():
            for name in ('report.json', 'steps.jsonl', 'rewards.jsonl'):
                pin(path / name)
    assert sum(c['changed_rows'] for c in cases) > 0
    assert all(sha(path) == digest for path, digest in pins.items())
    save(args.out / 'report.json', dict(state='complete', cases=cases, inputs=pins,
        neural_execution=False, new_games=False, training_performed=False,
        scope='Native TRAIN replay only. Zero-control preserves exact scores; cost changes one component only. '
              'No live-new-game, CUDA fitting or improved-policy claim.'))
    save(args.out / 'complete.json', dict(state='complete', report_sha256=sha(args.out / 'report.json')))
    print(cases)


if __name__ == '__main__':
    main()

"""Explain stationary TRAIN requests using recorded movement interventions only."""
import argparse
import collections
import json
import math
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('workflow', 'risk', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args()
    assert not a.out.exists()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = pins.get(str(path)) or sha(path)
        assert expected is None or expected == digest, str(path)
        pins[str(path)] = digest
        return path

    def rows(path):
        with path.open(encoding='utf-8-sig') as stream:
            return [json.loads(line) for line in stream]

    pin(__file__)
    seal = read(pin(a.risk / 'complete.json'))
    risk = read(pin(a.risk / 'report.json', seal['report_sha256']))
    assert seal['state'] == risk['state'] == 'complete' and risk['arm'] == 'ablated'
    assert not risk['new_games'] and not risk['neural_execution']
    for path, digest in risk['inputs'].items():
        pin(path, digest)
    groups, candidates = collections.defaultdict(collections.Counter), []
    sampled_maps = set()
    for case in risk['cases']:
        base = a.workflow / f"round-{case['round']:02d}" / 'ablated/corpus'
        pool = read(pin(base / 'pool/report.json'))
        job = next(j for j in pool['jobs'] if j['job'] == case['job'])
        directory = base / 'processing-train' / f"case-{case['job']:04d}"
        meta = read(pin(directory / 'cuda/report.json'))
        accepted = rows(pin(directory / 'cuda/rollout.jsonl', meta['rollout_sha256']))
        step_path = next((directory / 'native').glob('replay-*/steps.jsonl')).resolve()
        effect_path = step_path.with_name('server_outcomes.jsonl')
        steps = {s['index']: s for s in rows(pin(step_path, meta['source_sha256'][str(step_path)]))}
        effects = {s['step']: s for s in rows(pin(effect_path, meta['source_sha256'][str(effect_path)]))}
        run, previous_frame = 0, None
        for row in accepted:
            s, effect = steps[row['index']], effects[row['index']]
            o, requested, applied = s['observation'], s['action'], s['applied_action']
            speed = math.hypot(*o['velocity'][:2])
            run = (run + 1 if previous_frame == row['frame'] - 1 else 1) if speed <= 10 else 0
            previous_frame = row['frame']
            request = math.hypot(requested['forward'], requested['side']) >= .25
            stopped = math.hypot(applied['forward'], applied['side']) < .1
            reasons = sorted({i['reason'] for i in s.get('interventions') or [] if i['component'] == 'movement'})
            values = dict(rows=1, received=effect['received_health_damage'], deaths=effect['deaths'],
                on_ground=int(o['on_ground']), requested_nonzero=int(request), applied_near_zero=int(stopped))
            names = ['all']
            if speed <= 10:
                names.append('stationary_le10')
            if run >= 5:
                names.append('stationary_at_least5')
            if speed <= 10 and request and stopped:
                names += ['stationary_requested_stopped', 'stopped_map:' + o['identity']['map']]
                names += ['stopped_reason:' + reason for reason in reasons or ['no_movement_intervention_record']]
                if run >= 5:
                    names += ['long_stationary_requested_stopped']
                    names += ['long_stopped_reason:' + reason for reason in reasons or ['no_movement_intervention_record']]
                map_name = o['identity']['map']
                if run >= 5 and o['on_ground'] and reasons == ['no_ground_support'] and map_name not in sampled_maps:
                    sampled_maps.add(map_name)
                    candidates.append(dict(round=case['round'], job=case['job'], seed=case['seed'], index=row['index'],
                        identity=o['identity'], position=o['position'], velocity=o['velocity'],
                        observed_ground_drop=(o.get('local_geometry') or {}).get('ground_drop'),
                        requested=requested, applied=applied, native_step=s['native_step'],
                        step_path=str(step_path), capture_root=job['root'],
                        selection='First accepted TRAIN state per map with five stationary observations, grounded, requested movement and no_ground_support stop. No outcome ranking.'))
            for name in names:
                groups[name].update(values)
    assert groups['all']['rows'] == risk['groups']['all']['rows']
    assert groups['stationary_le10']['rows'] == risk['groups']['stationary_le10']['rows']
    assert groups['stationary_at_least5']['rows'] == risk['groups']['stationary_at_least5_consecutive_observations']['rows']
    assert all(sha(path) == digest for path, digest in pins.items())
    a.out.mkdir(parents=True)
    save(a.out / 'report.json', dict(state='complete', version='combat_train_motion_stalls_v1',
        groups={k: dict(v) for k, v in sorted(groups.items())}, support_probe_candidates=candidates, inputs=pins,
        neural_execution=False, new_games=False, training_performed=False,
        scope='Own TRAIN rows only. Recorded intervention reason explains command cancellation, not whether cancellation was physically necessary. '
              'First-state candidates require separate static/native counterfactual proof before any runtime change. No new safety bypass or policy input.'))
    save(a.out / 'complete.json', dict(state='complete', report_sha256=sha(a.out / 'report.json')))
    print(dict(groups={k: dict(v) for k, v in sorted(groups.items())}, candidates=len(candidates)))


if __name__ == '__main__':
    main()

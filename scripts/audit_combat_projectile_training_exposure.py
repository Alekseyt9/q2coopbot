"""Offline pre-fit mechanism coverage. Native labels never enter the loss or features."""
import argparse
from collections import Counter
import math
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha
from combat_enemy_bolt_lifetimes import parse_lifetimes, match_lifetime
import combat_enemy_bolt_lifetimes


def monster_damage_kind(fields):
    if not (fields['attacker_class'].startswith('monster_') and fields['target_class'].startswith('monster_')):
        return None
    before,after,take=map(int,(fields['health_before'],fields['health_after'],fields['take']))
    if max(before,0)-max(after,0)>0:
        return 'self_damage' if int(fields['attacker'])==int(fields['target']) else 'infighting'
    return 'corpse_contact' if before<=0 and take>0 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('corpus', 'protocol', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--round', type=int, required=True)
    a = ap.parse_args()
    assert not a.out.exists()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = pins.get(str(path)) or sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    def load(path, expected=None):
        return read(pin(path, expected))

    pin(__file__)
    pin(combat_enemy_bolt_lifetimes.__file__)
    protocol = load(a.protocol)
    closed = load(a.corpus/'complete.json')
    assert closed['native_cases'] == closed['cuda_cases'] == protocol['cases_per_update']
    processing = a.corpus/'processing-train'
    collection = load(a.corpus/'protocol.json', closed['protocol_sha256'])
    assert collection['train_only']
    pool = load(closed['source_pool_report'], closed['pool_sha256'])
    jobs = {job['job']:job for job in pool['jobs']}
    report = load(processing/'report.json', closed['processed'][0]['sha256'])
    counts = Counter({k:0 for k in ('enemy','self','other_player','unmatched','same_frame_reuse','enemy_velocity_known','native_infighting_events')})
    classes, enemy_cases, errors = Counter(), set(), []
    for case in report['cases']:
        job = jobs[case['job']]
        capture = load(Path(job['root'])/'report.json')
        start = capture['results'][0]['generated_start']
        assert start['confirmed']
        assert next(e for e in start['entities'] if e['class']=='monster_soldier_light')['health']==20
        folder = processing/f"case-{job['job']:04d}"
        meta = load(folder/'cuda/report.json')
        assert meta['numerical_verification']=='cuda_verified_v1'
        import json
        with pin(folder/'cuda/rollout.jsonl', meta['rollout_sha256']).open(encoding='utf-8-sig') as stream:
            accepted = {r['index'] for r in map(json.loads, stream)}
        counts['rows'] += len(accepted)
        log_path = next(Path(job['root']).glob('worker-*-episode-*/server.log')).resolve()
        log = pin(log_path, meta['source_sha256'][str(log_path)]).read_text(encoding='utf-8-sig').splitlines()
        lifetimes = parse_lifetimes(log)
        for line in log:
            if line.startswith('sv_test_damage '):
                fields = dict(part.split('=',1) for part in line.split() if '=' in part)
                kind=monster_damage_kind(fields)
                if kind=='infighting':
                    counts['native_infighting_events'] += 1
                elif kind:
                    counts['native_monster_'+kind+'_events'] += 1
        path = next((folder/'native').glob('replay-*/steps.jsonl')).resolve()
        with pin(path, meta['source_sha256'][str(path)]).open(encoding='utf-8-sig') as stream:
            for step in map(json.loads, stream):
                if step['index'] not in accepted:
                    continue
                o = step['observation']
                for projectile in sorted(o.get('visible_projectiles') or [], key=lambda p:p['distance'])[:4]:
                    classes[projectile['class']] += 1
                    if projectile['class']!='blaster_bolt':
                        continue
                    record, label = match_lifetime(lifetimes, o['identity'], projectile['id'])
                    counts[label] += 1
                    if label=='enemy' and projectile.get('observed_velocity') is not None:
                        counts['enemy_velocity_known'] += 1
                        enemy_cases.add(case['seed'])
                        errors.append(math.dist(projectile['observed_velocity'],record['native_initial_velocity']))
    floor = protocol['pre_fit_exposure']
    checks = dict(no_infighting=counts['native_infighting_events']==0,
        velocity_error_bounded=not errors or max(errors)<=floor['maximum_velocity_error'],
        grenade_present=classes['grenade']>0,
        first_update_enemy_coverage=(a.round!=1 or (counts['enemy_velocity_known']>=floor['first_update_minimum_enemy_velocity_rows']
            and len(enemy_cases)>=floor['first_update_minimum_enemy_velocity_cases'])))
    for path,digest in pins.items():
        assert sha(path)==digest, path
    a.out.mkdir(parents=True)
    result = dict(state='complete', version='combat_projectile_train_exposure_v1', gate_passed=all(checks.values()),
        checks=checks, counts=dict(counts), projectile_class_slots=dict(classes), enemy_velocity_cases=len(enemy_cases),
        enemy_velocity_error_maximum=max(errors,default=None), protocol_sha256=sha(a.protocol), inputs=pins,
        neural_execution=False, fitting=False, scope='Pre-fit data admission only; native ownership/velocity labels never enter policy inputs or PPO loss. Floor is a mechanism coverage requirement, not statistical sufficiency.')
    save(a.out/'report.json', result)
    save(a.out/'complete.json', dict(state='complete',report_sha256=sha(a.out/'report.json'),gate_passed=result['gate_passed']))
    print({k:v for k,v in result.items() if k!='inputs'})


if __name__=='__main__':
    main()

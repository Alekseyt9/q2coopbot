"""Describe motion, observed enemies and exact native incoming windows on TRAIN only."""
import argparse
import collections
import gzip
import json
import math
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha
from report_combat_machinegun_hits import scan
from report_registered_machinegun_outcomes import outcome


def distance_bin(distance):
    if distance is None:
        return 'unobserved'
    return '0-144' if distance <= 144 else '144-288' if distance <= 288 else '288-512' if distance <= 512 else '512+'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    closed_workflow = read(args.workflow / 'complete.json')
    assert closed_workflow['native_training_cases'] == closed_workflow['cuda_training_cases'] == 64
    pins, groups, cases, incoming_classes = {}, collections.defaultdict(collections.Counter), [], collections.defaultdict(collections.Counter)

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    pin(__file__)
    for name in ('report_combat_machinegun_hits.py', 'report_registered_machinegun_outcomes.py'):
        pin(Path(__file__).with_name(name))
    pin(args.workflow / 'complete.json')
    args.out.mkdir(parents=True)
    rows_path = args.out / 'frames.jsonl.gz'
    with gzip.open(rows_path, 'wt', encoding='utf-8', compresslevel=6) as output:
        for branch in ('zero', 'cost'):
            for number in (1, 2):
                root = args.workflow / branch / f'round-{number:02d}'
                corpus = root / 'corpus'
                closed = read(pin(corpus / 'complete.json'))
                protocol = read(pin(corpus / 'protocol.json', closed['protocol_sha256']))
                assert protocol['train_only'] and len(protocol['assignments']) == 1
                assignment = protocol['assignments'][0]
                assert assignment['split'] == 'train'
                plan = read(pin(assignment['plan'], assignment['plan_sha256']))
                recipes = {seed: task['episode']['id'] for task in plan['tasks'] for seed in task['seeds']}
                credit_seal = read(pin(root / 'native-credit/complete.json'))
                credit = read(pin(root / 'native-credit/report.json', credit_seal['report_sha256']))
                assert not credit['unassigned_native_effects']
                pool = read(pin(corpus / 'pool/report.json', closed['pool_sha256']))
                jobs = {job['job']: job for job in pool['jobs']}
                processed = closed['processed'][0]
                processing = read(pin(processed['report'], processed['sha256']))
                assert processing['device'] == 'cuda'
                corpus_rows = 0
                for item in processing['cases']:
                    assert item['split'] == 'train'
                    job = jobs[item['job']]
                    assert item['seed'] == job['seed'] and not job['error']
                    directory = corpus / 'processing-train' / f"case-{job['job']:04d}"
                    meta = read(pin(directory / 'cuda/report.json'))
                    rollout = pin(directory / 'cuda/rollout.jsonl', meta['rollout_sha256'])
                    with rollout.open(encoding='utf-8-sig') as stream:
                        ids = {row['index'] for row in map(json.loads, stream)}
                    assert len(ids) == meta['rows']
                    path = next(directory.glob('native/replay-*/steps.jsonl')).resolve()
                    pin(path, meta['source_sha256'][str(path)])
                    effects_path = path.with_name('server_outcomes.jsonl')
                    pin(effects_path, meta['source_sha256'][str(effects_path)])
                    with effects_path.open(encoding='utf-8-sig') as stream:
                        effects = {effect['step']: effect for effect in map(json.loads, stream)}
                    with path.open(encoding='utf-8-sig') as stream:
                        steps = [step for step in map(json.loads, stream) if step['index'] in ids]
                    assert len(steps) == len(ids)
                    windows = {tuple(step['native_step'][key] for key in ('spawncount', 'begin_frame', 'actor', 'sequence')): step for step in steps}
                    assert len(windows) == len(steps)
                    capture = read(pin(Path(job['root']) / 'report.json'))
                    log = pin(Path(capture['results'][0]['root']) / 'server.log')
                    with log.open(encoding='utf-8-sig') as stream:
                        shots = scan(stream, set(windows))
                    hits = collections.defaultdict(collections.Counter)
                    for shot in shots:
                        if shot['eligible']:
                            hits[windows[shot['window']]['index']]['mg_shots'] += 1
                            hits[windows[shot['window']]['index']]['mg_' + outcome(shot)] += 1
                    case_counts = collections.Counter()
                    condition = 'recovery' if recipes[job['seed']].endswith('-recovery') else 'ordinary'
                    for step in steps:
                        obs, nxt = step['observation'], step['next_observation']
                        assert obs['identity']['life'] == 1 and step['owner'] == 'provider' and step['sample']
                        assert nxt['identity']['frame'] == obs['identity']['frame'] + 1
                        effect = effects[step['index']]
                        assert effect['available']
                        visible = {enemy['id']: enemy for enemy in obs['enemies']}
                        nearest = min(visible.values(), key=lambda enemy: enemy['distance']) if visible else None
                        displacement = [b-a for a,b in zip(obs['position'][:2], nxt['position'][:2])]
                        travel = math.hypot(*displacement)
                        radial = None
                        if nearest and nearest['distance'] > 0:
                            xy = nearest['relative'][:2]
                            length = math.hypot(*xy)
                            if length > 0:
                                radial = sum(a*b for a,b in zip(displacement, xy)) / length
                        motion = 'still' if travel < 1 else 'approach' if radial is not None and radial > .5*travel else 'retreat' if radial is not None and radial < -.5*travel else 'lateral_or_unknown'
                        applied = step['applied_action']
                        events = []
                        counts = collections.Counter(rows=1, received=effect['received_health_damage'], own_damage=effect['monster_health_damage'],
                            kills=effect['monster_kills'], deaths=effect['deaths'], visible=bool(visible),
                            standing_before=math.hypot(*obs['velocity'][:2]) < 40,
                            moving_command=math.hypot(applied['forward'], applied['side']) >= .25)
                        counts.update(hits[step['index']])
                        measured = 0
                        for event in effect['events']:
                            if event['target'] != obs['identity']['actor']:
                                continue
                            damage = max(event['health_before'], 0) - max(event['health_after'], 0)
                            measured += damage
                            if not damage:
                                continue
                            enemy = visible.get(event['attacker'])
                            known = bool(enemy and enemy['class'] == event['attacker_class'])
                            timing = 'begin' if event['frame'] == obs['identity']['frame'] else 'end' if event['frame'] == nxt['identity']['frame'] else 'other'
                            counts['received_' + timing] += damage
                            attacker_class = event['attacker_class']
                            incoming_classes[branch, attacker_class]['damage'] += damage
                            incoming_classes[branch, attacker_class]['visible_damage'] += damage if known else 0
                            events.append(dict(damage=damage, timing=timing, attacker_class=attacker_class,
                                attacker_visible=known, observed_distance=enemy['distance'] if known else None))
                        assert measured == effect['received_health_damage']
                        for group in ('all', 'loadout:' + condition, 'motion:' + motion,
                                      'nearest:' + distance_bin(nearest['distance'] if nearest else None),
                                      'map:' + obs['identity']['map']):
                            groups[branch, group].update(counts)
                        case_counts.update(counts)
                        output.write(json.dumps(dict(branch=branch, round=number, seed=job['seed'], index=step['index'],
                            frame=obs['identity']['frame'], map=obs['identity']['map'], condition=condition,
                            position=obs['position'], next_position=nxt['position'], velocity=obs['velocity'],
                            on_ground=obs['on_ground'], weapon=obs['weapon'], ammo=obs['ammo'],
                            target=step['action'].get('target_entity', 0), applied=applied,
                            motion=motion, nearest_distance=nearest['distance'] if nearest else None,
                            visible_enemies=[{key:enemy.get(key) for key in ('id','class','distance','relative','velocity','observed_animation_frame','clear_shot')}
                                             for enemy in visible.values()], incoming=events, counts=dict(counts)), separators=(',', ':'))+'\n')
                    corpus_rows += len(steps)
                    cases.append(dict(branch=branch, round=number, seed=job['seed'], condition=condition, counts=dict(case_counts)))
                assert corpus_rows == credit['totals']['rows'] == processed['rows']
    assert all(sha(path) == digest for path,digest in pins.items())
    report = dict(state='complete', version='combat_train_motion_exposure_v1', cases=cases,
        groups=[dict(branch=branch, group=group, counts=dict(counts)) for (branch,group),counts in sorted(groups.items())],
        incoming_classes=[dict(branch=branch, attacker_class=kind, counts=dict(counts)) for (branch,kind),counts in sorted(incoming_classes.items())],
        inputs=pins, frames_sha256=sha(rows_path), neural_execution=False, training_performed=False, new_games=False,
        scope='TRAIN-only descriptive associations, not counterfactual movement benefits. Motion is measured after the command; '
              'damage at begin/end is separated. End damage may come from earlier attacks. Observed attacker position is before the command; '
              'native attacker labels are offline only and unknown visibility is never filled with a hidden position.')
    save(args.out / 'report.json', report)
    save(args.out / 'complete.json', dict(state='complete', report_sha256=sha(args.out / 'report.json'), frames_sha256=sha(rows_path)))
    print(dict(rows=sum(case['counts']['rows'] for case in cases), cases=len(cases), incoming_classes=report['incoming_classes']))


if __name__ == '__main__':
    main()

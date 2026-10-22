"""Join hypothetical expert recommendations with canonical own-policy features.

Executed learner commands have native provenance; expert commands do not. Keep
that distinction in the schema and never convert these labels into PPO samples.
"""
import argparse
import collections
import gzip
import json
import math
import pathlib

from evaluate_registered_combat_models import read, save, sha


def identity(value):
    return tuple(value[k] for k in ('map', 'connection', 'spawncount', 'actor', 'life', 'frame'))


def label(row):
    c = row['combat_policy']; o = c['observation']; a = row.get('combat_advice')
    if c['selection']['owner'] != 'provider' or not a or not a['available']:
        return None
    assert a['version'] == 'reactive_combat_advice_v2' and a['identity'] == o['identity']
    action = a['recommended_action']; cmd = a['recommended_command']
    assert action['identity'] == o['identity'] and not action['weapon']
    assert cmd['Impulse'] == 0 and cmd['Msec'] == 100
    assert action['forward'] == cmd['Forward']/400 and action['side'] == cmd['Side']/400
    assert action['attack'] == bool(cmd['Buttons'] & 1)
    assert 0 <= o['observation_age_ms'] <= 300 and o['health'] > 0
    enemies = sorted(o['enemies'], key=lambda e:e['distance'])[:8]
    target = 0
    if action.get('target_entity', 0):
        candidates = [i+1 for i,e in enumerate(enemies) if e['id'] == action['target_entity']
                      and e.get('observed_track') == action['target_track'] and e.get('observed_solid')
                      and e.get('clear_shot')]
        assert len(candidates) == 1, 'Recommended target must exist in current observed mask'
        target = candidates[0]
    else:
        assert a['decision']['aim_source'] == 'none' and not action['attack']
    decision = dict(target=target, target_mask=True, attack=action['attack'], attack_mask=True,
                    targetless_release=target == 0, reason='hypothetical_reactive_expert_on_learner_state')
    ground = None
    # No vertical supervision: joint movement/aim labels are restricted to the
    # same grounded standing release pose supported by the existing trainer.
    if o['on_ground'] and not o['ducked'] and cmd['Up'] == 0 and row['sent_command']['Up'] == 0:
        movement = [action['forward'], action['side']]
        assert all(math.isfinite(x) and abs(x) <= 1 for x in movement)
        angles = [action['yaw_delta_degrees'], action['pitch_delta_degrees']]
        assert all(math.isfinite(x) and abs(x) <= 180 for x in angles)
        mode = int(max(map(abs,angles)) < 15)
        bound = 15 if mode else 180
        # Finite inverse-tanh target at the exact protocol turn boundary.
        angles = [max(-bound*(1-1e-6), min(bound*(1-1e-6), x)) for x in angles]
        ground = dict(movement=movement, target=target, mode=mode, aim_degrees=angles,
                      kind='moving' if any(movement) else 'hold')
    return dict(ground=ground, decision=decision)


def main():
    ap = argparse.ArgumentParser()
    for name in ('corpus','parity','out'): ap.add_argument('--'+name,type=pathlib.Path,required=True)
    args = ap.parse_args(); assert not args.out.exists()
    root = args.corpus.resolve(); pins = {}
    def pin(path):
        path = pathlib.Path(path).resolve(); pins[str(path)] = sha(path); return path
    complete = read(pin(root/'complete.json')); assert complete['state'] == 'complete'
    parity = read(pin(args.parity)); assert parity['state'] == 'complete' and parity['matched_observations_and_commands']
    protocol = read(pin(root/'protocol.json')); assert protocol['record_combat_advice']
    pool = read(pin(root/'pool/report.json')); assert pool['state'] == 'complete' and pool['source_unchanged']
    assert complete['pool_sha256'] == sha(root/'pool/report.json') and all(not j['error'] for j in pool['jobs'])
    assert pool['usable_captures'] == len(pool['jobs']) == 48
    args.out.mkdir(parents=True); splits = {}; seen = set()
    for index, assignment in enumerate(protocol['assignments']):
        split = assignment['label']; assert split in ('train','development')
        plan = read(pin(assignment['plan'])); assert plan['record_combat_advice']
        assert all(t['split'] == ('train' if split == 'train' else 'validation') for t in plan['tasks'])
        process = read(pin(root/('processing-'+split)/'report.json'))
        assert process['state'] == 'complete' and process['device'] == 'cuda'
        counts = collections.Counter(); seeds = set(); path = args.out/(split+'.jsonl.gz')
        with gzip.open(path,'wt',encoding='utf-8',compresslevel=6) as output:
            for job in sorted((j for j in pool['jobs'] if j['plan_index'] == index),key=lambda j:j['job']):
                seeds.add(job['seed'])
                report = read(pin(pathlib.Path(job['root'])/'report.json')); assert report['capture_complete'] and report['provenance_valid']
                result = report['results'][0]; assert result['capture_valid']
                trace = pin(pathlib.Path(result['root'])/'bot.jsonl')
                with trace.open(encoding='utf-8-sig') as stream:
                    raw = {identity(r['combat_policy']['observation']['identity']):r for r in map(json.loads,stream)}
                seq = pin(root/('processing-'+split)/f"case-{job['job']:04d}"/'native/sequence.jsonl')
                with seq.open(encoding='utf-8') as stream:
                    for r in map(json.loads,stream):
                        assert r['seed'] == job['seed'] and len(r['features']) == 1147
                        candidates = [x for k,x in raw.items() if k[-1] == r['frame']]
                        assert len(candidates) == 1
                        x = candidates[0]; o = x['combat_policy']['observation']; labels = label(x)
                        counts['rows'] += 1
                        if labels is None:
                            counts['unavailable'] += 1
                            labels = dict(ground=None,decision=dict(target=0,target_mask=False,attack=False,attack_mask=False,targetless_release=False))
                        else:
                            counts['decision_labels'] += 1
                            counts['targetless_release'] += labels['decision']['targetless_release']
                            counts['attack_positive'] += labels['decision']['attack']
                            if labels['ground']:
                                counts['ground_labels'] += 1
                                counts['ground_'+labels['ground']['kind']] += 1
                                counts['ground_targetless'] += labels['ground']['target'] == 0
                        output.write(json.dumps(dict(seed=job['seed'],identity=o['identity'],features=r['features'],
                            expert_hypothetical=True,expert_native_matched=False,**labels),allow_nan=False)+'\n')
        assert not seen.intersection(seeds); seen.update(seeds)
        assert counts['ground_moving'] and counts['ground_hold'] and counts['attack_positive']
        splits[split] = dict(path=str(path.resolve()),sha256=sha(path),counts=dict(counts),seeds=sorted(seeds))
    assert set(splits) == {'train','development'}
    for p,h in pins.items(): assert sha(p) == h
    save(args.out/'complete.json',dict(state='complete',version='combat_reactive_advice_labels_v1',test_deferred=True,
        splits=splits,inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),
        training_performed=False,neural_execution=False,promotion=False,weapon_supervision=False,vertical_supervision=False,
        scope='Hypothetical reactive planner advice on actual learner states, not native-executed expert actions and not on-policy PPO samples. Canonical Go features include all learner history frames. Ground motion/aim restricted to standing grounded release pose; no-target horizon/hold included. No teacher optimality claim.'))
    print(args.out/'complete.json')


if __name__ == '__main__': main()

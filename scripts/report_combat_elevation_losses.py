"""Read-only observed elevation/visibility diagnostic on sealed learned cases."""
import argparse
from collections import Counter
import json
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); assert not args.out.exists()
    root = args.evaluation.resolve(); seal = read(root / 'complete.json')
    assert seal['state'] == 'complete' and sha(args.spec) == seal['spec_sha256']
    pool_path = root / 'pool/report.json'; pool = read(pool_path)
    assert sha(pool_path) == seal['pool_sha256'] and pool['state'] == 'complete'
    pins = {str(root / 'complete.json'): sha(root / 'complete.json'), str(args.spec): sha(args.spec), str(pool_path): sha(pool_path)}
    variants = {}
    for index, variant in enumerate(read(args.spec)['variants']):
        if variant['mode'] != 'learned': continue
        label = variant['label']; behavior_path = root / ('behavior-' + label + '.json')
        behavior = read(behavior_path)
        assert behavior['pool_sha256'] == seal['pool_sha256'] and behavior['selected_plan_index'] == index
        pins[str(behavior_path)] = sha(behavior_path)
        bindings = {case['job']: case for case in behavior['cases']}; cases = []
        for job in pool['jobs']:
            if job['plan_index'] != index: continue
            binding = bindings[job['job']]
            steps = root / ('processing-' + label) / ('case-%04d' % job['job']) / 'native/replay-0/steps.jsonl'
            rewards = steps.with_name('rewards.jsonl')
            pins[str(steps)] = sha(steps); pins[str(rewards)] = sha(rewards)
            initial = None; min_z = None; last = None; counts = Counter(); first_drop = None
            with steps.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    row = json.loads(line); obs = row['observation']; dispatch = row.get('server_execution') or {}
                    if row.get('owner') != 'provider' or obs['health'] <= 0 or not dispatch.get('matched') or not dispatch.get('window_exclusive'): continue
                    assert row.get('native_step') and obs['identity']['frame'] >= binding['release_frame']
                    position = obs['position']; frame = obs['identity']['frame']
                    if initial is None: initial = position
                    min_z = position[2] if min_z is None else min(min_z, position[2]); last = position
                    fallen = initial[2] - position[2] >= 96
                    if fallen and first_drop is None: first_drop = frame - binding['release_frame']
                    visible = {e['id'] for e in (obs.get('enemies') or [])}
                    above = [e for e in (obs.get('remembered_threats') or []) if e['id'] not in visible and e['age_frames'] > 0 and e['last_observed_relative'][2] >= 64]
                    counts['frames'] += 1
                    counts['below_initial_96'] += fallen
                    counts['no_visible_enemy'] += not visible
                    counts['no_visible_remembered_above_64'] += not visible and bool(above)
                    counts['below_initial_and_no_visible_remembered_above'] += fallen and not visible and bool(above)
            totals = Counter()
            with rewards.open(encoding='utf-8-sig') as stream:
                for line in stream: totals.update(json.loads(line)['components'])
            assert counts['frames'] > 0
            cases.append(dict(job=job['job'], seed=job['seed'], map=binding['map'], episode=binding['episode'],
                              goal=binding['goal'], death=binding['death'], initial_position=initial, last_position=last,
                              min_z=min_z, max_drop_from_initial=initial[2]-min_z, first_drop_frame=first_drop,
                              counts=dict(counts), reward_components=dict(totals), behavior_counts=binding['counts']))
        variants[label] = dict(cases=cases)
    for path, digest in pins.items(): assert sha(Path(path)) == digest
    save(args.out, dict(state='complete', variants=variants, inputs=[dict(path=p,sha256=d) for p,d in pins.items()],
         training_performed=False, promotion=False, script_sha256=sha(__file__),
         scope='Observed native-exclusive alive provider frames only. Relative to initial observed Z; remembered positions are stale observations. A drop can be intentional and is not causal proof of defeat. No hidden current enemy positions.'))
    for label, variant in variants.items():
        print(label, [(c['seed'], round(c['max_drop_from_initial'],1), c['counts']['below_initial_and_no_visible_remembered_above'],c['goal']) for c in variant['cases'] if c['map']=='base3'])


if __name__ == '__main__': main()

"""Small balanced comparison specs using the existing registry and evaluator."""
import argparse
import collections
import copy
import hashlib
import json
import pathlib

from evaluate_registered_combat_models import read, sha, save


STAGES = {'smoke': (1, 4, 4), 'screen': (2, 8, 4),
          'coverage': (2, 20, 8), 'confirmation': (4, 20, 8)}


def select_families(episodes, stage, salt):
    count, ordinary, recovery = STAGES[stage]
    ids = [ep['id'] for ep in episodes]
    assert len(ids) == len(set(ids))
    selected = []
    for is_recovery, budget in ((False, ordinary), (True, recovery)):
        groups = collections.defaultdict(list)
        for ep in episodes:
            if ep['id'].endswith('-recovery') == is_recovery:
                assert ep['status'] == 'runnable' and ep['timescale'] == 2
                groups[ep['map']].append(ep['id'])
        assert groups and sum(map(len, groups.values())) >= budget, 'Insufficient strata'
        for group in groups.values():
            group.sort(key=lambda name: hashlib.sha256((str(salt)+':'+name).encode()).hexdigest())
        # Round robin over maps prevents a small slice concentrating at one map.
        maps = sorted(groups)
        chosen = 0
        while chosen < budget:
            progressed = False
            for name in maps:
                if groups[name] and chosen < budget:
                    selected.append(groups[name].pop(0)); chosen += 1; progressed = True
            assert progressed
    assert len(selected) == ordinary+recovery
    return selected, count


def make_spec(template, episodes, stage, control, candidate, offset, salt, port):
    assert template['version'] == 'registered_combat_model_comparison_v1'
    assert control != candidate and type(offset) is int and offset >= 0
    by_label = {v['label']: v for v in template['variants']}
    assert len(by_label) == len(template['variants'])
    assert by_label[control]['mode'] == by_label[candidate]['mode'] == 'learned'
    assert by_label[control]['deterministic'] == by_label[candidate]['deterministic'], 'Execution modes differ'
    rules = [v for v in template['variants'] if v['mode'] == 'rules']
    assert len(rules) == 1
    selected, count = select_families(episodes, stage, salt)
    spec = copy.deepcopy(template)
    spec.update(episodes=selected, count=count, seed_offset=offset, port=port,
        split='confirmation' if stage == 'confirmation' else 'validation',
        variants=[copy.deepcopy(by_label[control]), copy.deepcopy(by_label[candidate]), copy.deepcopy(rules[0])],
        comparisons=[dict(name='candidate-vs-control', before=control, after=candidate),
                     dict(name='candidate-vs-rules', before=rules[0]['label'], after=candidate)])
    spec['hypothesis_stage'] = dict(version='combat_hypothesis_stage_v1', stage=stage,
        selection_seed=salt, cases_per_variant=count*len(selected), native_budget=3*count*len(selected),
        ordinary_cases=STAGES[stage][1]*count, recovery_cases=STAGES[stage][2]*count,
        purpose='fresh-confirmation' if stage == 'confirmation' else 'screening-not-promotion',
        automatic_promotion=False, training_performed=False)
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--template', type=pathlib.Path, required=True)
    ap.add_argument('--stage', choices=STAGES, required=True)
    ap.add_argument('--control', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--seed-offset', type=int, required=True)
    ap.add_argument('--selection-seed', type=int, default=20261012)
    ap.add_argument('--port', type=int, default=41700)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    args = ap.parse_args(); assert not args.out.exists()
    template = read(args.template); registry = pathlib.Path(template['registry'])
    index = read(registry)
    episodes = [read((registry.parent/pathlib.Path(name)).resolve()) for name in index['files']]
    spec = make_spec(template, episodes, args.stage, args.control, args.candidate,
        args.seed_offset, args.selection_seed, args.port)
    assert 1024 <= args.port <= 65520 and template['max_instances'] == 16
    # Include frozen one-level evaluation plans, not just top-level plan files.
    # The evaluator checks the actual generated seeds against this history.
    artifacts = pathlib.Path(__file__).resolve().parents[1]/'workspace/artifacts'
    old = set(template.get('historical_plans', []))
    old.update(str(p.resolve()) for p in artifacts.glob('*/*-plan.json'))
    spec['historical_plans'] = sorted(old)
    spec['hypothesis_stage'].update(template_sha256=sha(args.template), registry_sha256=sha(registry),
        generator_sha256=sha(__file__), selected_episode_canonical_sha256={ep['id']: hashlib.sha256(json.dumps(ep,sort_keys=True).encode()).hexdigest()
            for ep in episodes if ep['id'] in spec['episodes']})
    save(args.out, spec)
    print(json.dumps(spec['hypothesis_stage'], ensure_ascii=False))


if __name__ == '__main__': main()

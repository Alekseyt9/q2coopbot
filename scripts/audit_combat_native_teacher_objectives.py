"""Audit teacher-label sensitivity; never relabel records or fit a policy."""
import argparse
import json
from pathlib import Path

from combat_native_teacher_registry import VERSION, load_index, read, sha
from process_combat_architecture_pool import save


def dominates(a, b):
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--index', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = a.out.resolve()
    assert not out.exists()
    records = load_index(repo, a.index)
    pins = {str(a.index.resolve()): sha(a.index)}
    pending = [((repo/corpus['manifest']).resolve(), corpus['manifest_sha256'])
               for corpus in read(a.index)['corpora']]
    seen = set()
    while pending:
        manifest, digest = pending.pop()
        assert sha(manifest) == digest
        if manifest in seen:
            continue
        seen.add(manifest)
        pins[str(manifest)] = digest
        for path, digest in read(manifest)['inputs'].items():
            if path.endswith('rollout.jsonl'):
                assert path not in pins or pins[path] == digest
                pins[path] = digest
            elif Path(path).name == 'manifest.json':
                # Derived corpora retain pinned parent teacher manifests;
                # capture manifests have another schema and are not teachers.
                assert sha(path) == digest
                parent = read(path)
                if parent.get('version') == VERSION and parent.get('state') == 'complete':
                    pending.append((Path(path).resolve(), digest))
    out.mkdir(parents=True)
    save(out/'protocol.json', dict(
        diagnostic_only=True, training_performed=False, promotion=False,
        source_index_sha256=sha(a.index), gammas=[.99, .995, .999, 1.0],
        margin=.05, label_changes_performed=False,
        interpretation='Discount sensitivity and aggregate native end-stat Pareto comparisons are descriptive. They do not establish an optimal objective. Earlier kills, time, ammunition and future safety can differ even when end stats match. Ties and ineligible cases remain present.',
        inputs=pins))
    cases = []
    counts = dict(choice0=0, choice1=0, tie=0, ineligible=0,
                  labelled_goal_differences=0, goal_disagreements=0,
                  labelled_death_differences=0, survival_disagreements=0,
                  end_stat_dominated_preferences=0, undiscounted_opposite_labels=0)
    for r in records:
        outcomes = [r['outcomes'][k] for k in ('choice0', 'choice1')]
        native = [x['native'] for x in outcomes]
        utilities = []
        for outcome in outcomes:
            path = Path(outcome['data'])/'rollout.jsonl'
            assert str(path) in pins and sha(path) == pins[str(path)]
            rr = [json.loads(line) for line in path.open(encoding='utf-8-sig')]
            rr = [v for v in rr if v['frame'] >= r['frame']]
            scores = {str(g): sum(g**(v['frame']-r['frame'])*v['reward'] for v in rr)
                      for g in (.99, .995, .999, 1.0)}
            assert abs(scores['0.99']-outcome['discounted_observed_return']) < 1e-9
            utilities.append(scores)
        preference = r['target']['preference']
        labelled = preference is not None
        eligible = r['target']['eligible_for_teacher_supervision']
        label_name = 'ineligible' if not eligible else 'tie' if not labelled else 'choice'+str(preference)
        counts[label_name] += 1
        goal_pref = int(native[1]['goal']) if native[0]['goal'] != native[1]['goal'] else None
        survival_pref = int(not native[1]['death']) if native[0]['death'] != native[1]['death'] else None
        vectors = [(int(v['goal']), -int(v['death']), v['kills'], v['dealt'], -v['received']) for v in native]
        end_pref = 0 if dominates(vectors[0], vectors[1]) else 1 if dominates(vectors[1], vectors[0]) else None
        deltas = {g: utilities[1][g]-utilities[0][g] for g in utilities[0]}
        sensitivity = {g: None if abs(v) < .05 else int(v > 0) for g, v in deltas.items()}
        flags = dict(
            goal_disagreement=labelled and goal_pref is not None and preference != goal_pref,
            survival_disagreement=labelled and survival_pref is not None and preference != survival_pref,
            end_stat_dominated_preference=labelled and end_pref is not None and preference != end_pref,
            undiscounted_opposite_label=labelled and sensitivity['1.0'] is not None and preference != sensitivity['1.0'])
        counts['labelled_goal_differences'] += labelled and goal_pref is not None
        counts['labelled_death_differences'] += labelled and survival_pref is not None
        for key, flag in flags.items():
            count_key = {'end_stat_dominated_preference': 'end_stat_dominated_preferences',
                         'undiscounted_opposite_label': 'undiscounted_opposite_labels'}.get(key, key+'s')
            counts[count_key] += int(flag)
        cases.append(dict(id=r['id'], map=r['map'], seed=r['seed'], frame=r['frame'],
                          eligible=eligible, preference=preference, goal_preference=goal_pref,
                          survival_preference=survival_pref, end_stat_pareto_preference=end_pref,
                          utility_deltas=deltas, discount_preferences=sensitivity, flags=flags,
                          native=native, observed_rows=[x['observed_return_rows'] for x in outcomes]))
    for path, digest in pins.items():
        assert sha(path) == digest
    save(out/'report.json', dict(state='complete', records=len(records), counts=counts,
        maps=len({r['map'] for r in records}), unique_source_worlds=len({(r['seed'],r['source_model_sha256']) for r in records}),
        cases=cases, diagnostic_only=True, label_changes_performed=False,
        training_performed=False, promotion=False))
    print(json.dumps(dict(records=len(records), counts=counts)))


if __name__ == '__main__':
    main()

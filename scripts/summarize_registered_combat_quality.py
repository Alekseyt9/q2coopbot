"""Aggregate paired native quality reports sharing one unchanged baseline."""
import argparse
import collections
import hashlib
import json
import pathlib


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def groups(pair):
    episode = pair['episode']
    yield 'map/' + pair['map']
    if '-group-parasite-infantry-gunner-weapons' in episode:
        composition = 'trio'
    elif '-group-parasite-infantry-weapons' in episode:
        composition = 'pair'
    else:
        return
    loadout = 'weapons-ssg' if episode.endswith('-weapons-ssg') else 'weapons'
    yield composition + '/' + loadout
    yield 'map/' + pair['map'] + '/' + composition + '/' + loadout
    if '-site-01-' in episode:
        yield 'site/first'
    elif '-site-02-' in episode:
        yield 'site/second'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison', action='append', required=True, help='AFTER_LABEL=paired_quality.json')
    parser.add_argument('--baseline-label', default='parent')
    parser.add_argument('--out', type=pathlib.Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    sources = {}
    totals = {}
    paired_goal_changes = {}
    strata = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    baseline = None
    for argument in args.comparison:
        label, filename = argument.split('=', 1)
        assert label and label != args.baseline_label and label not in sources
        path = pathlib.Path(filename).resolve()
        source_hash = sha(path)
        report = json.loads(path.read_text(encoding='utf-8-sig'))
        assert report['version'] == 'registered_combat_paired_quality_v1' and report['state'] == 'complete'
        pairs = report['pairs']
        old = {(p['episode'], p['seed'], p['map']):(p['metrics']['before'], p['before_capture_sha256']) for p in pairs}
        assert pairs and len(old) == len(pairs)
        first = baseline is None
        if first:
            baseline = old
            totals[args.baseline_label] = report['totals']['before']
        assert old == baseline, 'Baseline fixture/metrics/capture receipts differ'
        totals[label] = report['totals']['after']
        changes = collections.Counter()
        for pair in pairs:
            before_goal = bool(pair['metrics']['before']['goal'])
            after_goal = bool(pair['metrics']['after']['goal'])
            category = ('both_won' if before_goal else 'both_lost') if before_goal == after_goal else ('gained' if after_goal else 'lost')
            changes[category] += 1
        assert changes['gained'] - changes['lost'] == totals[label]['goal'] - totals[args.baseline_label]['goal']
        paired_goal_changes[label] = {key:changes[key] for key in ('gained','lost','both_won','both_lost')}
        for pair in pairs:
            for group in groups(pair):
                strata[group][label].update(dict(battles=1, **pair['metrics']['after']))
                if first:
                    strata[group][args.baseline_label].update(dict(battles=1, **pair['metrics']['before']))
        assert sha(path) == source_hash
        sources[label] = dict(path=str(path), sha256=source_hash)
    result = dict(version='registered_combat_quality_strata_v1', state='complete',
                  cases_per_variant=len(baseline), sources=sources, totals=totals, paired_goal_changes=paired_goal_changes, groups=strata,
                  scope='Descriptive strata of accepted paired native outcomes. Known parasite/infantry[/gunner] recipes classified from explicit recipe IDs; unknown compositions remain in map totals. No fitting or architecture promotion. Shorter episodes may reflect earlier deaths.')
    with args.out.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(json.dumps(totals, indent=2))


if __name__ == '__main__':
    main()

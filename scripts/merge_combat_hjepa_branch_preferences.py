"""Merge compatible native preference cohorts without renumbering identities."""
import argparse
import json
import pathlib
from train_combat_hjepa import sha
from train_combat_hjepa_branch_preferences import load_native


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=pathlib.Path, action='append', required=True)
    ap.add_argument('--split', choices=['train', 'validation'], required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    a = ap.parse_args()
    if a.out.exists() or pathlib.Path(str(a.out) + '.meta.json').exists():
        raise ValueError('fresh merged output required')
    rows, metas, seeds, extra = [], [], set(), {}
    for path in a.data:
        current = load_native(path, a.split)
        meta_path = pathlib.Path(str(path) + '.meta.json')
        meta = json.loads(meta_path.read_text())
        new_seeds = {r['seed'] for r in current}
        if seeds & new_seeds:
            raise ValueError('native preference seed overlap')
        if metas and (meta['parent_sha256'] != metas[0]['parent_sha256'] or
                      meta.get('target_mode', 'soft_score') != metas[0].get('target_mode', 'soft_score') or
                      meta['score'] != metas[0]['score']):
            raise ValueError('incompatible parent/target/outcome contracts')
        for key in ['branch_states', 'completion', 'parent_source']:
            p = pathlib.Path(meta[key]).resolve()
            extra[str(p)] = sha(p)
        extra[str(path.resolve())] = sha(path)
        extra[str(meta_path.resolve())] = sha(meta_path)
        extra.update(meta.get('additional_inputs_sha256', {}))
        for r in current:
            r.pop('teacher_weapon_probabilities', None)
        rows += current
        metas.append(meta)
        seeds |= new_seeds
    merged = dict(metas[0], rows=len(rows), additional_inputs_sha256=extra,
                  keep_wins=sum(m['keep_wins'] for m in metas), best_branch_wins=sum(m['best_branch_wins'] for m in metas),
                  rejected=[r for m in metas for r in m['rejected']])
    for field in ['sources_sha256', 'native_receipts_sha256']:
        merged[field] = {}
        for m in metas:
            for name, digest in m[field].items():
                if name in merged[field] and merged[field][name] != digest:
                    raise ValueError('conflicting source receipt')
                merged[field][name] = digest
    a.out.write_text(''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in rows))
    pathlib.Path(str(a.out) + '.meta.json').write_text(json.dumps(merged, indent=2))
    print(json.dumps(dict(rows=len(rows), seeds=len(seeds), split=a.split)))


if __name__ == '__main__':
    main()

"""Merge verified offline H-JEPA trajectories without crossing policy/life boundaries."""
import argparse
import json
import pathlib
from train_combat_hjepa import sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=pathlib.Path,action='append',required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    a=ap.parse_args()
    sources,parents,rows,seen={},{},[],set()
    objective=None
    for path in a.input:
        meta=json.loads(pathlib.Path(str(path)+'.meta.json').read_text())
        if meta['version']!='hjepa_data_v1' or meta['feature_version']!='combat_features_v6':raise ValueError('incompatible features')
        if meta.get('data_sha256')!=sha(path):raise ValueError('input data changed')
        if objective is not None and objective!=meta['objective_version']:raise ValueError('incompatible objectives')
        objective=meta['objective_version']
        parents[str(path)]=sha(path)
        for source,digest in meta['sources_sha256'].items():
            if sha(pathlib.Path(source))!=digest:raise ValueError('native source changed')
            if source in sources and sources[source]!=digest:raise ValueError('source conflict')
            sources[source]=digest
        group=[json.loads(line) for line in path.read_text().splitlines()]
        if len(group)!=meta['rows']:raise ValueError('row count mismatch')
        for row in group:
            key=(row['episode'],row['frame'])
            if key in seen:raise ValueError('duplicate trajectory/frame')
            seen.add(key);rows.append(row)
    with a.out.open('x',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    meta=dict(version='hjepa_data_v1',feature_version='combat_features_v6',rows=len(rows),sources_sha256=sources,
              parent_data_sha256=parents,data_sha256=sha(a.out),objective_version=objective,
              scope='Offline world trajectories only; policy/life namespaces retained. Not a PPO on-policy merge.')
    pathlib.Path(str(a.out)+'.meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(dict(rows=len(rows),objective=objective,parents=len(a.input))))


if __name__=='__main__':main()

"""Compose a matched comparison from two sealed comparisons sharing an anchor."""
import argparse
import pathlib

from process_combat_architecture_pool import read, save, sha
from report_registered_combat_comparison import paired_summary


def compose(left, right):
    """Endpoints are each 'after'; the same observed 'before' is the anchor."""
    def indexed(value):
        rows={}
        for pair in value['pairs']:
            key=(pair['episode'],pair['seed'],pair['map'])
            assert key not in rows,'Duplicate matched fixture'
            rows[key]=pair
        return rows
    a,b=indexed(left),indexed(right)
    assert a and a.keys()==b.keys(),'Endpoint fixtures differ'
    pairs=[]
    for key in sorted(a):
        x,y=a[key],b[key]
        assert x['before_capture_sha256']==y['before_capture_sha256'],'Anchor capture differs'
        assert x['metrics']['before']==y['metrics']['before'],'Anchor outcomes differ'
        pairs.append(dict(episode=key[0],seed=key[1],map=key[2],
            metrics=dict(before=x['metrics']['after'],after=y['metrics']['after']),
            before_capture_sha256=x['after_capture_sha256'],after_capture_sha256=y['after_capture_sha256'],
            anchor_capture_sha256=x['before_capture_sha256']))
    return pairs


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evaluation',type=pathlib.Path,required=True)
    ap.add_argument('--spec',type=pathlib.Path,required=True)
    for name in ('anchor','left','right'):ap.add_argument('--'+name,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    args=ap.parse_args();assert not args.out.exists()
    root=args.evaluation.resolve();seal_path=root/'complete.json';seal=read(seal_path)
    assert seal['state']=='complete' and not seal['training_performed'] and not seal['promotion']
    assert sha(args.spec)==seal['spec_sha256'];spec=read(args.spec)
    pool_path=root/'pool/report.json';assert sha(pool_path)==seal['pool_sha256']
    pool=read(pool_path);assert pool['state']=='complete' and pool['source_unchanged']
    assert pool['usable_captures']==len(pool['jobs'])==seal['native_cases']
    assert all(not j['error'] for j in pool['jobs'])
    labels=[v['label'] for v in spec['variants']]
    assert len(set((args.anchor,args.left,args.right)))==3
    assert len(labels)==len(set(labels))
    assert [{k:c[k] for k in ('name','before','after')} for c in seal['comparisons']]==spec['comparisons']
    pins={seal_path:sha(seal_path),args.spec:seal['spec_sha256'],pool_path:seal['pool_sha256']}
    endpoints=[]
    for label in (args.left,args.right):
        found=[c for c in seal['comparisons'] if c['before']==args.anchor and c['after']==label]
        assert len(found)==1,'A unique sealed common-anchor comparison is required'
        binding=found[0];path=root/('quality-'+binding['name']+'.json')
        assert sha(path)==binding['sha256'];pins[path]=binding['sha256'];value=read(path)
        assert value['state']=='complete'
        assert value['before_pool_sha256']==value['after_pool_sha256']==seal['pool_sha256']
        assert value['before_plan_index']==labels.index(args.anchor)
        assert value['after_plan_index']==labels.index(label)
        endpoints.append(value)
    pairs=compose(*endpoints)
    assert len(pairs)*len(labels)==seal['native_cases']
    groups={'all':paired_summary(pairs)}
    for group in ('ordinary','recovery'):
        rows=[p for p in pairs if p['episode'].endswith('-recovery')==(group=='recovery')]
        if rows:groups[group]=paired_summary(rows)
    for path,digest in pins.items():assert sha(path)==digest,str(path)
    scope=('Supplementary descriptive comparison of endpoints on identical held-out fixtures, '
        'composed from sealed same-anchor capture/outcome pairs. No new battles, inference or fitting. '
        'Exploratory family bootstrap, no multiplicity correction or independent training replicas. '
        'No promotion, campaign or coop acceptance.')
    save(args.out,dict(state='complete',version='registered_combat_cross_comparison_v1',
        anchor=args.anchor,before=args.left,after=args.right,pairs=pairs,groups=groups,
        inputs=[dict(path=str(p),sha256=d) for p,d in pins.items()],
        script_sha256=sha(pathlib.Path(__file__)),summary_helper_sha256=sha(pathlib.Path(__file__).with_name('report_registered_combat_comparison.py')),
        training_performed=False,promotion=False,scope=scope))
    print({g:s['paired_goal_changes'] for g,s in groups.items()},flush=True)


if __name__=='__main__':main()

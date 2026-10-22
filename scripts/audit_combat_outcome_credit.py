"""TRAIN reward/outcome ranking and actually normalized pre-fit PPO signal."""
import argparse
import ast
import json
from pathlib import Path
import statistics
from evaluate_registered_combat_models import read,save,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();assert not args.out.exists()
    root=args.corpus.resolve();closed=read(root/'complete.json');assert closed['state']=='complete'
    pool=read(root/'pool/report.json');assert sha(root/'pool/report.json')==closed['pool_sha256']
    data=root/'processing-train/merged';meta=read(data/'report.json')
    assert sha(data/'rollout.jsonl')==meta['rollout_sha256']
    rows=[json.loads(s) for s in (data/'rollout.jsonl').read_text().splitlines()]
    source=Path(__file__).with_name('ppo_combat.py');tree=ast.parse(source.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='advantages')
    namespace={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),namespace)
    advantages=namespace['advantages'];adv,_=advantages(rows,.99,.95);mc,_=advantages(rows,.99,1.)
    mean=statistics.mean(adv);std=statistics.pstdev(adv)
    cases=[];pins={str(root/'complete.json'):sha(root/'complete.json'),str(data/'rollout.jsonl'):meta['rollout_sha256']}
    for job in pool['jobs']:
        if job['plan_index']!=0:continue
        path=Path(job['root'])/'report.json';capture=read(path);pins[str(path)]=sha(path)
        result=capture['results'][0];assert result['capture_valid'] and result['seed']==job['seed']
        selected=[row for row in rows if row['seed']==job['seed']]
        outcome='win' if result.get('goal_stop') else 'death' if result.get('death_stop') else 'no_goal_or_death_marker'
        cases.append(dict(seed=job['seed'],outcome=outcome,rows=len(selected),
                          reward=sum(row['reward'] for row in selected),
                          discounted_return=sum(.99**i*row['reward'] for i,row in enumerate(selected))))
    index={(row['seed'],row['index']):i for i,row in enumerate(rows)};falls=[]
    path=root/'departure-audit.json';pins[str(path)]=sha(path)
    for case in read(path)['cases']:
        for event in case['events']:
            if not event['deep_drop']:continue
            i=index[case['seed'],event['index']]
            falls.append(dict(seed=case['seed'],index=event['index'],drop=event['drop'],
                              raw_gae=adv[i],normalized_gae=(adv[i]-mean)/(std+1e-8),lambda1_return=mc[i]))
    groups={}
    for name in sorted({case['outcome'] for case in cases}):
        selected=[case for case in cases if case['outcome']==name]
        groups[name]=dict(cases=len(selected),mean_reward=statistics.mean(c['reward'] for c in selected))
    save(args.out,dict(state='complete',split='TRAIN',cases=cases,groups=groups,deep_departures=falls,
         advantage_mean=mean,advantage_std=std,raw_positive_deep=sum(x['raw_gae']>0 for x in falls),
         normalized_positive_deep=sum(x['normalized_gae']>0 for x in falls),
         inputs=[dict(path=p,sha256=d) for p,d in pins.items()],ppo_source_sha256=sha(source),
         script_sha256=sha(__file__),training_performed=False,neural_computation=False,
         limitation='Descriptive recorded-outcome ranking. Positive raw GAE is not the normalized PPO gradient coefficient and does not establish that falling is rewarded.'))
    print(dict(groups=groups,raw_positive_deep=sum(x['raw_gae']>0 for x in falls),normalized_positive_deep=sum(x['normalized_gae']>0 for x in falls)))


if __name__=='__main__':main()

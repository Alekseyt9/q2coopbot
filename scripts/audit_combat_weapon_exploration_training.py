"""Count observed-state exploration opportunities in sealed train rows; no NN."""
import argparse,collections,json,pathlib
from process_combat_architecture_pool import read,save,sha
from combat_equipped_weapon_features import CATEGORIES


def row_counts(row):
    x=row['features'];assert len(x)==1147
    mask=x[833:845];equipped=x[1121:1133]
    assert all(v in (0,1) for v in mask+equipped) and mask[0]==1 and sum(equipped)==1
    weapon=row['sample'].get('weapon',0)
    assert type(weapon) is int and 0<=weapon<12 and mask[weapon]==1
    current=equipped.index(1);eligible=bool(current==11 and any(mask[1:5]))
    counts=collections.Counter(rows=1,eligible_rows=int(eligible),
        attack_samples=int(bool(row['sample']['attack'])),
        intervention_rows=int(bool(row.get('interventions'))))
    if eligible:
        counts['eligible_keep_samples']+=int(weapon==0)
        counts['eligible_supported_samples']+=int(weapon in range(1,5))
        counts['eligible_other_samples']+=int(weapon>=5)
        counts['eligible_attack_samples']+=int(bool(row['sample']['attack']))
    return counts,current,weapon


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--training',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True);args=ap.parse_args()
    assert not args.out.exists();root=args.training.resolve();seal_path=root/'complete.json'
    seal=read(seal_path);assert seal['state']=='complete' and seal['device']=='cuda'
    pool_path=root/'training/pool/report.json';pool=read(pool_path)
    assert pool['state']=='complete' and pool['source_unchanged']
    assert pool['usable_captures']==len(pool['jobs'])==seal['native_cases']==seal['cuda_cases']==336
    pins={seal_path:sha(seal_path),pool_path:sha(pool_path)}
    variants={}
    for update in seal['updates']:
        label=update['label'];processing=root/('training/processing-'+label)
        proc_path=processing/'report.json';proc=read(proc_path);pins[proc_path]=sha(proc_path)
        assert proc['state']=='complete' and proc['device']=='cuda' and proc['pool_sha256']==pins[pool_path]
        index=proc['selected_plan_index'];binding=pool['plans'][index]
        plan_path=pathlib.Path(binding['path']);assert sha(plan_path)==binding['sha256'];pins[plan_path]=binding['sha256']
        plan=read(plan_path);model_path=pathlib.Path(plan['model_path'])
        assert sha(model_path)==plan['model_sha256']==proc['model_sha256'];pins[model_path]=proc['model_sha256']
        model=read(model_path);assert model['feature_version']=='combat_features_v10'
        temperature=(model.get('weapon_exploration') or {}).get('temperature',1)
        jobs={j['job']:j for j in pool['jobs'] if j['plan_index']==index}
        cases=[]
        for entry in proc['cases']:
            job=jobs[entry['job']];assert not job['error'] and entry['split']=='train' and entry['seed']==job['seed']
            case=processing/f"case-{entry['job']:04d}"
            native_path=case/'native/report.json';cuda_path=case/'cuda/report.json'
            native,cuda=read(native_path),read(cuda_path)
            pins[native_path]=sha(native_path);pins[cuda_path]=sha(cuda_path)
            assert native['model_sha256']==cuda['model_sha256']==proc['model_sha256']
            assert native['native_rollout_sha256']==cuda['native_rollout_sha256']
            assert native['sequence_sha256']==cuda['sequence_sha256']
            stream_path=case/'native/native-rollout.jsonl'
            assert sha(stream_path)==native['native_rollout_sha256']
            counts=collections.Counter();equipment=collections.Counter();requests=collections.Counter();seen=set()
            with stream_path.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    row=json.loads(line);assert row['seed']==entry['seed']
                    assert (row['index'],row['frame']) not in seen;seen.add((row['index'],row['frame']))
                    c,current,request=row_counts(row);counts.update(c)
                    equipment[CATEGORIES[current]]+=1;requests[CATEGORIES[request] if request else 'keep']+=1
            assert counts['rows']==native['rows']==cuda['rows']
            assert sha(stream_path)==native['native_rollout_sha256']
            episode=plan['tasks'][job['task_index']]['episode']['id']
            cases.append(dict(job=entry['job'],seed=entry['seed'],episode=episode,counts=dict(counts),
                equipment=dict(equipment),requests=dict(requests),native_rollout_sha256=native['native_rollout_sha256']))
            if len(cases)%32==0:print(label,len(cases),flush=True)
        assert len(cases)==len(jobs)==112
        groups={}
        for name in ('all','ordinary','recovery'):
            selected=[c for c in cases if name=='all' or c['episode'].endswith('-recovery')==(name=='recovery')]
            counts=sum((collections.Counter(c['counts']) for c in selected),collections.Counter())
            groups[name]=dict(cases=len(selected),eligible_cases=sum(c['counts'].get('eligible_rows',0)>0 for c in selected),counts=dict(counts))
        assert groups['all']['counts']['rows']==proc['rows']==update['rows']
        variants[label]=dict(temperature=temperature,cases=cases,groups=groups)
    for p,d in pins.items():assert sha(p)==d,str(p)
    save(args.out,dict(state='complete',version='combat_weapon_exploration_training_observation_v1',
        variants=variants,inputs=[dict(path=str(p),sha256=d) for p,d in pins.items()],
        script_sha256=sha(pathlib.Path(__file__)),training_performed=False,neural_execution=False,promotion=False,
        scope='Only previously consumed eligible train transitions, with native/CUDA declared rollout and sequence hashes matched. State eligibility is Grenades equipped plus supported inventory mask; temp1 counts opportunities but applies no scaling. Samples are policy intent, not applied requests or observed response. Each branch visits its own trajectory; row fractions are descriptive, not causal effectiveness or held-out outcomes. No fitting, replay or relabeling.'))


if __name__=='__main__':main()

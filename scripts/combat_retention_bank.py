"""Pinned observation-only distillation bank, separate from PPO transitions."""
import argparse, collections, hashlib, json, math, pathlib

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))

def bucket(o,mode='count'):
    if mode not in ('count','composition'):raise ValueError('Unknown bank balance mode')
    enemies=o['enemies'];n=min(2,len(enemies))
    wall=any(p.get('standing_hull_clearance') is not None and p['standing_hull_clearance']<40 for p in o.get('local_geometry',{}).get('probes',[]))
    barrel=any(p['class']=='misc_explobox' and p['distance']<320 for p in o.get('visible_props',[]))
    projectile=bool(o.get('visible_projectiles'))
    group=f'enemies{n}' if mode=='count' else 'classes:'+('+'.join(sorted(e['class'] for e in enemies)) or 'none')
    return f'{group}/wall{int(wall)}/barrel{int(barrel)}/projectile{int(projectile)}'

def select(rows,limit,mode='count'):
    if mode not in ('count','composition'):raise ValueError('Unknown bank balance mode')
    groups=collections.defaultdict(list)
    for r in rows:groups[r['bucket']].append(r)
    selected=[];contexts=collections.Counter(key.split('/')[0] for key in groups)
    for key,group in sorted(groups.items()):
        group=sorted(group,key=lambda r:(r['seed'],r['index']))
        count=min(limit,len(group))
        for i in range(count):
            mass=1/len(groups) if mode=='count' else 1/(len(contexts)*contexts[key.split('/')[0]])
            selected.append({**group[i*len(group)//count],'weight':mass/count})
    return selected

def validate(bank,anchor_sha,feature_version,width,ppo_seeds=()):
    if bank.get('version')!='combat_retention_bank_v1' or bank.get('anchor_sha256')!=anchor_sha or bank.get('feature_version')!=feature_version:raise ValueError('Bank anchor/feature contract differs')
    train=set(bank['train_seeds']);validation=set(bank['validation_seeds']);forbidden=set(bank['forbidden_seeds'])
    if not train or not validation or train & validation or (train|validation)&(forbidden|set(ppo_seeds)):raise ValueError('Bank episode split or PPO/evaluation seeds overlap')
    for split,seeds in [('train',train),('validation',validation)]:
        rows=bank[split]
        if not rows or {r['seed'] for r in rows}-seeds:raise ValueError('Invalid bank split')
        identities=set()
        for r in rows:
            if set(r)!={'features','bucket','seed','index','weight'}:raise ValueError('Bank must contain observations only')
            key=(r['seed'],r['index'])
            if key in identities:raise ValueError('Duplicate bank observation')
            identities.add(key)
            if len(r['features'])!=width or not all(math.isfinite(v) for v in r['features']) or not math.isfinite(r['weight']) or r['weight']<=0:raise ValueError('Invalid bank feature/weight')
        if not math.isclose(sum(r['weight'] for r in rows),1.,abs_tol=1e-9):raise ValueError('Bank weights must sum to one')
    for path,digest in bank['source_sha256'].items():
        if sha(path)!=digest:raise ValueError(f'Changed bank input: {path}')
    return bank

def pin_bank(digest,weight,previous=None):
    if digest is None:
        if previous:raise ValueError('Resumed bank requires the pinned bank')
        return None
    if not math.isfinite(weight) or weight<=0:raise ValueError('Positive finite bank weight required')
    spec=dict(sha256=digest,weight=weight)
    if previous is not None and previous!=spec:raise ValueError('Bank or weight changed on resume')
    return spec

def rollout_seeds(seed,iterations,episodes_per_worker=1):
    if not 1<=iterations<=20 or not 1<=episodes_per_worker<=20 or seed<0 or seed+4*episodes_per_worker*iterations-1>2147483647:
        raise ValueError('Valid independent rollout seeds required')
    return list(range(seed,seed+4*episodes_per_worker*iterations))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--anchor-model',type=pathlib.Path,required=True);ap.add_argument('--train-data',type=pathlib.Path,action='append');ap.add_argument('--validation-data',type=pathlib.Path,action='append');ap.add_argument('--forbidden-seeds',type=pathlib.Path);ap.add_argument('--per-bucket',type=int,default=64);ap.add_argument('--out',type=pathlib.Path)
    ap.add_argument('--validate',type=pathlib.Path);ap.add_argument('--ppo-seed',type=int);ap.add_argument('--eval-seed',type=int);ap.add_argument('--iterations',type=int,default=4)
    ap.add_argument('--episodes-per-worker',type=int,choices=range(1,21),default=1)
    ap.add_argument('--eval-episodes-per-worker',type=int,choices=range(1,21),default=1)
    ap.add_argument('--balance',choices=('count','composition'),default='count');a=ap.parse_args()
    if a.validate:
        anchor=read(a.anchor_model)
        seeds=rollout_seeds(a.ppo_seed,a.iterations,a.episodes_per_worker) if a.ppo_seed is not None else []
        if a.eval_seed is not None:seeds+=rollout_seeds(a.eval_seed,1,a.eval_episodes_per_worker)
        validate(read(a.validate),sha(a.anchor_model),anchor['feature_version'],len(anchor['actor'][0]['weight'][0]),seeds);print('Bank validation passed');return
    if not a.train_data or not a.validation_data or not a.forbidden_seeds or not a.out:raise ValueError('Train/validation data, excluded seeds and output required')
    if a.per_bucket<1:raise ValueError('Positive bucket cap required')
    anchor=read(a.anchor_model);anchor_sha=sha(a.anchor_model);sources={str(a.anchor_model.resolve()):anchor_sha,str(a.forbidden_seeds.resolve()):sha(a.forbidden_seeds)}
    bank=dict(version='combat_retention_bank_v1',anchor_sha256=anchor_sha,feature_version=anchor['feature_version'],forbidden_seeds=read(a.forbidden_seeds),source_sha256=sources)
    if a.balance!='count':bank['balancing']=dict(mode=a.balance,weighting='equal_composition_then_context')
    coverage={}
    for split,folders in [('train',a.train_data),('validation',a.validation_data)]:
        rows=[];seeds=set()
        for folder in folders:
            meta=read(folder/'report.json')
            if meta['model_sha256']!=anchor_sha or meta['feature_version']!=anchor['feature_version']:raise ValueError('Bank data must come from the pinned parent')
            if sha(folder/'rollout.jsonl')!=meta['rollout_sha256']:raise ValueError('Changed exported bank data')
            for path,digest in {**meta['source_sha256'],str((folder/'report.json').resolve()):sha(folder/'report.json'),str((folder/'rollout.jsonl').resolve()):meta['rollout_sha256']}.items():
                if path in sources and sources[path]!=digest:raise ValueError('Conflicting bank receipt')
                sources[path]=digest
            receipts=[pathlib.Path(p) for p in meta['source_sha256'] if pathlib.Path(p).name=='steps.jsonl']
            observed={}
            for p in receipts:
                for line in p.read_text().splitlines():
                    s=json.loads(line);episode=int(s['episode'].split('-')[-1])
                    observed[(episode,s['index'])]=s['observation']
            data=[json.loads(line) for line in (folder/'rollout.jsonl').read_text().splitlines()]
            batch_seeds={r['seed'] for r in data}
            if seeds&batch_seeds:raise ValueError('Duplicate bank episode')
            seeds|=batch_seeds
            for r in data:rows.append(dict(features=r['features'],seed=r['seed'],index=r['index'],bucket=bucket(observed[(r['seed'],r['index'])],a.balance)))
        bank[split]=select(rows,a.per_bucket,a.balance);bank[f'{split}_seeds']=sorted(seeds)
        coverage[split]=dict(available=len(rows),selected=len(bank[split]),buckets=dict(collections.Counter(r['bucket'] for r in bank[split])))
    validate(bank,anchor_sha,anchor['feature_version'],len(anchor['actor'][0]['weight'][0]))
    with a.out.open('x',encoding='utf-8') as f:json.dump(bank,f,allow_nan=False)
    report=dict(bank_sha256=sha(a.out),coverage=coverage,scope='Only native-verified parent observations; deterministic bucket balance; validation episodes never enter training loss.')
    a.out.with_suffix('.report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()

"""Condition-only TRAIN subsets; original pool/plan binding stays authoritative."""
VERSION='combat_condition_balanced_subset_v1'
REQUEST_VERSION='combat_full_joint_subset_train_request_v2'


def select_seeds(plan,count):
    assert type(count) is int and count>0
    pool=[];seen=set()
    for task in plan['tasks']:
        assert task['split']=='train' and task['modes']==['learned']
        for instance in task['instances']:
            seed=instance['engine_seed']
            assert type(seed) is int and seed not in seen
            assert instance['split']=='train'
            seen.add(seed);pool.append((task['episode']['map'],task['episode']['id'],seed))
    assert count<=len(pool)
    maps={};episodes={};chosen=[]
    while len(chosen)<count:
        pair=min(pool,key=lambda p:(maps.get(p[0],0),episodes.get(p[1],0),p[2]))
        pool.remove(pair);chosen.append(pair[2])
        maps[pair[0]]=maps.get(pair[0],0)+1
        episodes[pair[1]]=episodes.get(pair[1],0)+1
    return sorted(chosen)


def validate(plan,manifest,plan_hash):
    assert set(manifest)=={'version','parent_plan_sha256','count','seeds','outcome_independent'}
    assert manifest['version']==VERSION and manifest['parent_plan_sha256']==plan_hash
    assert manifest['outcome_independent'] is True
    expected=select_seeds(plan,manifest['count'])
    assert manifest['seeds']==expected,'Subset must reproduce from conditions, not combat outcomes'
    return set(expected)

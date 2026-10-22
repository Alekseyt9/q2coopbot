"""Validate and concatenate immutable TRAIN replay shards on CUDA."""
import copy
import hashlib
import json
import pathlib
from process_combat_architecture_pool import read, sha


def policy_family(model):
    """Only conditional aim weights/variance may differ in this experiment."""
    fixed=copy.deepcopy(model)
    fixed.pop('value')
    fixed['attention'].pop('value')
    fixed['sampled_motion_aim'].pop('layers')
    fixed['log_std']=fixed['log_std'][:2]
    return hashlib.sha256(json.dumps(fixed,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def disjoint_keys(groups):
    seen=set()
    for keys in groups:
        local={(k['seed'],k['index']) for k in keys}
        assert len(local)==len(keys), 'Duplicate transition within replay shard'
        assert not local.intersection(seen), 'Duplicate transition across replay shards'
        seen.update(local)
    return seen


def load_catalog(paths, model, objective, pin):
    from train_combat_bc import torch
    assert torch.cuda.is_available()
    blocks=[];manifests=[];key_groups=[];offset=0
    family=policy_family(model)
    for path in paths:
        root=pathlib.Path(path).resolve();meta=read(pin(root/'complete.json'))
        assert meta['state']=='complete' and meta['device']=='cuda' and not meta['training_performed']
        assert meta['version']=='combat_causal_replay_v1' and meta['reward_sha256']==objective
        assert not meta['excluded_missing_successor']
        for item in meta['inputs']:pin(item['path'],item['sha256'])
        behavior=read(pin(pathlib.Path(meta['source'])/'weights.json',meta['source_weights_sha256']))
        assert policy_family(behavior)==family, 'Frozen encoder/policy factors differ'
        corpus=pathlib.Path(meta['corpus']);closed=read(corpus/'complete.json')
        protocol=read(pin(corpus/'protocol.json',closed['protocol_sha256']))
        assert protocol['train_only'] and all(a['split']=='train' for a in protocol['assignments'])
        seeds={s for a in protocol['assignments'] for s in a['seeds']}
        keys=read(pin(root/'keys.json',meta['keys_sha256']))
        assert all(k['seed'] in seeds for k in keys)
        data=torch.load(pin(root/'replay.pt',meta['replay_sha256']),map_location='cuda',weights_only=True)
        assert data['source_weights_sha256']==meta['source_weights_sha256'] and data['reward_sha256']==objective
        assert len(data['current'])==len(keys)==meta['rows']
        for value in data.values():
            if torch.is_tensor(value):assert value.is_cuda and bool(torch.isfinite(value).all())
        count=len(data['features']);assert len(data['raw'])==len(data['history'])==count
        assert bool(((data['current']>=0)&(data['current']<count)).all())
        assert bool(((data['successor']>=0)&(data['successor']<count)).all())
        data['current']=data['current']+offset;data['successor']=data['successor']+offset;offset+=count
        blocks.append(data);key_groups.append(keys)
        manifests.append(dict(path=str(root),complete_sha256=sha(root/'complete.json'),
            behavior_weights_sha256=meta['source_weights_sha256'],cases=meta['training_cases'],rows=meta['rows'],
            rollout_sha256=read(corpus/'processing-train/merged/report.json')['rollout_sha256']))
    disjoint_keys(key_groups)
    fields=('features','raw','history','current','successor','bootstrap','latent','condition','reward')
    result={name:torch.cat([b[name] for b in blocks],0) for name in fields}
    return result,manifests,family

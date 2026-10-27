"""Separate diagnostic-pair supervision contract. No ordinary PPO admission."""
import gzip
import json
import math
from pathlib import Path
from process_combat_architecture_pool import sha

VERSION='combat_causal_fire_pairs_v2'
LABEL_VERSION='combat_observed_suffix_preference_v1'


def preferred_attack(delta,margin):
    assert math.isfinite(delta) and math.isfinite(margin) and margin>0
    return 0 if delta>margin else 1 if delta < -margin else None


def load(path,verify_inputs=True):
    with gzip.open(path,'rt',encoding='utf-8') as f:data=json.load(f)
    validate(data)
    if verify_inputs:
        assert all(sha(Path(p))==digest for p,digest in data['inputs'].items()),'Pair input changed'
    return data


def validate(data):
    assert data['version']==VERSION and data['ordinary_ppo_eligible'] is False
    config=data['label_config']
    assert config['version']==LABEL_VERSION and config['bootstrap'] is False
    assert config['reward_version']=='combat_reward_v4' and 0<config['gamma']<=1 and config['margin']>0
    seen=set();splits=set()
    for case in data['cases']:
        assert case['seed'] not in seen;seen.add(case['seed'])
        assert case['split'] in ('train','diagnostic');splits.add(case['split'])
        assert set(case['ownership'])=={'source','release'}
        assert case['eligible_for_preference']==all(not frames for frames in case['ownership'].values())
        x=case['input'];assert set(x)=={'seed','index','frame','features','sample','sequence_prefix'}
        assert x['seed']==case['seed'] and len(x['features'])==data['feature_width']
        assert all(math.isfinite(v) for v in x['features'])
        assert set(x['sample'])=={'latent','target','aim_mode','vertical','weapon','memory'}
        assert len(x['sample']['latent'])==4 and all(math.isfinite(v) for v in x['sample']['latent'])
        prefix=x['sequence_prefix'];assert prefix and prefix[0]['memory']['reset'] is True
        last=None
        for row in prefix:
            assert set(row)=={'features','frame','index','memory','seed'}
            assert row['seed']==x['seed'] and row['frame']<=x['frame'] and row['index']<=x['index'],'Future context forbidden'
            assert len(row['features'])==data['feature_width'] and all(math.isfinite(v) for v in row['features'])
            if last is not None:
                assert row['index']>last['index'] and row['frame']>last['frame']
                assert row['memory']['reset'] or row['frame']==last['frame']+1
            assert set(row['memory'])<= {'actor','value','reset','position'}
            last=row
        assert last['index']==x['index'] and last['frame']==x['frame']
        assert last['features']==x['features'] and last['memory']==x['sample']['memory']
        y=case['target'];assert set(y)=={'source_return','release_return','delta','preferred_attack'}
        assert all(math.isfinite(y[k]) for k in ('source_return','release_return','delta'))
        assert abs(y['delta']-(y['release_return']-y['source_return']))<1e-9
        assert y['preferred_attack']==preferred_attack(y['delta'],config['margin'])
    assert splits=={'train','diagnostic'}

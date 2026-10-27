"""Metadata admission for paired continuation evidence, never ordinary PPO."""
import math
import re
from pathlib import Path
from statistics import mean,stdev

from combat_native_teacher_registry import INPUT_KEYS
from process_combat_architecture_pool import read,sha

VERSION='combat_counterfactual_continuations_v1'
INDEX_VERSION='combat_continuation_registry_v1'


def summary(streams):
    assert len(streams)>=3
    deltas=[s['outcomes']['choice1']['utility']-s['outcomes']['choice0']['utility'] for s in streams]
    eligible=[all(s['outcomes'][m]['rules_combat_rows']==0 for m in ('choice0','choice1')) for s in streams]
    active=[d for d,e in zip(deltas,eligible) if e]
    return dict(streams=len(streams),hybrid_streams=eligible.count(False),
        forks_observed_both=sum(s['suffix_seed'] is not None and all(o['fork_observed'] for o in s['outcomes'].values()) for s in streams),
        choice0=sum(d<=-.05 for d in active),tie=sum(abs(d)<.05 for d in active),choice1=sum(d>=.05 for d in active),
        all_streams_autonomous=all(eligible),mean_delta=mean(deltas) if all(eligible) else None,
        sample_stddev=stdev(deltas) if all(eligible) else None,
        observed_preference_flip=any(d<=-.05 for d in active) and any(d>=.05 for d in active))


def validate_record(record):
    assert record['version']==VERSION and record['diagnostic_only'] is True
    assert record['ordinary_ppo_eligible'] is False and record['training_target_emitted'] is False
    for name in ('id','source_model_sha256'):assert re.fullmatch('[0-9a-f]{64}',record[name])
    assert record['discount']==.99 and re.fullmatch('[0-9a-f]{64}',record['reward_config_sha256'])
    assert record['reward_version'] and record['combat_execution_contract']
    assert type(record['seed']) is int and 0<=record['seed']<=2147483647
    assert type(record['frame']) is int and record['frame']>0 and isinstance(record['map'],str) and record['map']
    x=record['input'];assert set(x)==INPUT_KEYS,'Future-dependent or unknown input field'
    for key,size in (('features',None),('source_latent',4),('pre_gate_context187',187)):
        values=x[key];assert isinstance(values,list) and values and (size is None or len(values)==size)
        assert all(type(v) in (int,float) and math.isfinite(v) for v in values)
    assert x['aim_mode']==0 and type(x['target']) is int and 1<=x['target']<=8
    assert type(x['vertical']) is int and 0<=x['vertical']<3
    assert type(x['weapon']) is int and 0<=x['weapon']<12 and isinstance(x['feature_version'],str)
    seen=set()
    for stream in record['streams']:
        seed=stream['suffix_seed']
        assert seed is None or type(seed) is int and 1<=seed<=2147483647
        assert seed not in seen,'Duplicate original or suffix RNG stream';seen.add(seed)
        assert stream['fork_frame']==(None if seed is None else record['frame']+1)
        assert set(stream['outcomes'])=={'choice0','choice1'}
        for outcome in stream['outcomes'].values():
            assert type(outcome['utility']) in (int,float) and math.isfinite(outcome['utility'])
            assert type(outcome['rules_combat_rows']) is int and outcome['rules_combat_rows']>=0
            assert type(outcome['goal']) is bool and type(outcome['death']) is bool
            assert outcome['fork_observed'] is None if seed is None else type(outcome['fork_observed']) is bool
            assert isinstance(outcome['data'],str) and outcome['data']
    assert None in seen,'Missing original continuation'
    assert record['summary']==summary(record['streams']),'Empirical summary disagrees with retained outcomes'


def load_manifest(path,expected_sha256=None):
    path=Path(path).resolve()
    if expected_sha256 is not None:assert sha(path)==expected_sha256
    manifest=read(path)
    assert manifest['version']==VERSION and manifest['state']=='complete'
    assert manifest['ordinary_ppo_eligible'] is False and manifest['training_targets_emitted'] is False
    records=[];seen=set()
    for entry in manifest['entries']:
        target=Path(entry['path']).resolve();assert target.is_relative_to(path.parent)
        assert sha(target)==entry['sha256'];record=read(target);validate_record(record)
        assert record['id']==entry['id'] and record['map']==entry['map'] and record['seed']==entry['seed']
        key=(record['source_model_sha256'],record['map'],record['seed'],record['frame'])
        assert key not in seen;seen.add(key);records.append(record)
    assert len(records)==manifest['contexts'] and records
    return records


def load_index(repo,path):
    repo=Path(repo).resolve();index=read(path)
    assert index['version']==INDEX_VERSION and index['ordinary_ppo_eligible'] is False
    assert index['training_targets_emitted'] is False and index['split_unit']=='map'
    records=[];seen=set()
    for corpus in index['corpora']:
        target=(repo/corpus['manifest']).resolve();assert target.is_relative_to(repo/'workspace/artifacts')
        for record in load_manifest(target,corpus['manifest_sha256']):
            assert record['id'] not in seen;seen.add(record['id']);records.append(record)
    assert records
    return records

"""Strict metadata admission for counterfactual supervision, never PPO rollout loading."""
import hashlib,json,math,re
from pathlib import Path

VERSION='combat_native_coupled_gate_teacher_v1'
INPUT_KEYS={'feature_version','features','pre_gate_context187','source_latent','target','aim_mode','vertical','weapon'}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_record(r):
    assert r['version']==VERSION and r['diagnostic_only'] is True and r['ordinary_ppo_eligible'] is False
    assert r['training_performed'] is False and r['coupled_fire_verified'] is True and r['same_continuation_policy'] is True
    assert re.fullmatch('[0-9a-f]{64}',r['id']) and re.fullmatch('[0-9a-f]{64}',r['source_model_sha256'])
    assert type(r['seed']) is int and 0<=r['seed']<=2147483647 and type(r['frame']) is int and r['frame']>0
    x=r['input'];assert set(x)==INPUT_KEYS,'Unexpected or future-dependent teacher input'
    for key,size in (('source_latent',4),('pre_gate_context187',187),('features',None)):
        assert isinstance(x[key],list) and x[key] and (size is None or len(x[key])==size)
        assert all(type(v) in (int,float) and math.isfinite(v) for v in x[key])
    assert type(x['target']) is int and 1<=x['target']<=8 and x['aim_mode']==0
    assert type(x['vertical']) is int and 0<=x['vertical']<3 and type(x['weapon']) is int and 0<=x['weapon']<12
    t=r['target'];assert type(t['eligible_for_teacher_supervision']) is bool
    assert math.isfinite(t['utility_delta']) and math.isfinite(t['tie_margin']) and t['tie_margin']>0
    before,after=(r['outcomes'][k] for k in ('choice0','choice1'))
    delta=after['discounted_observed_return']-before['discounted_observed_return']
    assert abs(delta-t['utility_delta'])<1e-9,'Teacher delta does not match paired outcomes'
    if t['eligible_for_teacher_supervision']:
        assert not t['ineligible_reasons']
        assert all(v['native']['rules_combat_rows']==0 for v in (before,after))
        expected=None if abs(delta)<t['tie_margin'] else int(delta>0)
    else:
        assert t['ineligible_reasons'];expected=None
    assert t['preference'] is None or type(t['preference']) is int
    assert t['preference']==expected,'Preference disagrees with declared utility/margin/eligibility'

def load_manifest(path,expected_sha256=None):
    path=Path(path).resolve()
    if expected_sha256 is not None:assert sha(path)==expected_sha256
    manifest=read(path);assert manifest['version']==VERSION and manifest['state']=='complete'
    assert manifest['ordinary_ppo_eligible'] is False and manifest['diagnostic_only'] is True and manifest['training_performed'] is False
    records=[];seen=set()
    for entry in manifest['entries']:
        record_path=Path(entry['path']).resolve();assert record_path.is_relative_to(path.parent),'Teacher record escapes its corpus directory'
        assert sha(record_path)==entry['sha256'];r=read(record_path);validate_record(r)
        assert r['id']==entry['id'] and r['seed']==entry['seed'] and r['map']==entry['map']
        assert r['target']['preference']==entry['preference'] and r['target']['eligible_for_teacher_supervision']==entry['teacher_eligible']
        key=(r['seed'],r['frame'],r['source_model_sha256']);assert key not in seen,'Duplicate counterfactual root'
        seen.add(key);records.append(r)
    assert records
    return records

def load_index(repo,path):
    repo=Path(repo).resolve();index=read(path)
    assert index['version']=='combat_counterfactual_teacher_index_v1' and index['ordinary_ppo_eligible'] is False
    records=[];seen=set()
    for corpus in index['corpora']:
        target=(repo/corpus['manifest']).resolve();assert target.is_relative_to(repo/'workspace/artifacts')
        assert corpus['schema']==VERSION and corpus['ordinary_ppo_eligible'] is False
        rr=load_manifest(target,corpus['manifest_sha256']);assert len(rr)==corpus['records']
        for r in rr:
            key=(r['seed'],r['frame'],r['source_model_sha256']);assert key not in seen,'Duplicate root across teacher corpora'
            seen.add(key);records.append(r)
    return records

"""Build observed-suffix labels from verified native action pairs, never PPO rows."""
import argparse
import copy
import gzip
import json
from pathlib import Path
from process_combat_architecture_pool import read,save,sha
from combat_causal_fire_pairs import VERSION,validate,preferred_attack


def lines(path):
    with path.open(encoding='utf-8-sig') as f:return [json.loads(s) for s in f if s.strip()]


def build(protocol_path,native_root,audit_root,config_path,out):
    assert not out.exists();out.mkdir(parents=True)
    protocol=read(protocol_path);pool=read(native_root/'report.json');audit=read(audit_root/'report.json')
    assert pool['state']=='complete' and pool['source_unchanged'] and pool['protocol_sha256']==sha(protocol_path)
    assert audit['state']=='passed' and all(audit['checks'].values()) and not audit['training_performed']
    assert read(audit_root/'complete.json')['report_sha256']==sha(audit_root/'report.json')
    assert all(sha(Path(p))==digest for p,digest in audit['inputs'].items())
    assert all(sha(Path(p))==digest for p,digest in protocol['inputs'].items())
    config=read(config_path);cases=[];inputs={}
    def pin(path):
        path=Path(path).resolve();inputs[str(path)]=sha(path);return path
    for p in (protocol_path,native_root/'report.json',audit_root/'report.json',audit_root/'complete.json',config_path,Path(__file__),Path(__file__).with_name('combat_causal_fire_pairs.py')):pin(p)
    model_hash=None;reward_hash=None;feature_width=None
    for c in protocol['cases']:
        model=Path(c['model']);mh=sha(pin(model))
        assert model_hash is None or mh==model_hash;model_hash=mh
        original=Path(c['original_trace']);original_case=original.parents[2]
        meta=read(pin(original_case/'cuda/report.json'));assert meta['numerical_verification']=='cuda_verified_v1' and meta['model_sha256']==mh
        rh=meta['reward_config_sha256'];assert reward_hash is None or rh==reward_hash;reward_hash=rh
        original_rows=lines(pin(original_case/'cuda/rollout.jsonl'))
        root=next(r for r in original_rows if r['frame']==c['frame'])
        context=lines(pin(original_case/'cuda/sequence.jsonl'))
        prefix=[copy.deepcopy(r) for r in context if r['index']<=root['index'] and r['frame']<=root['frame']]
        assert root['sample']['attack'] is True
        width=len(root['features']);assert feature_width is None or feature_width==width;feature_width=width
        sample={k:copy.deepcopy(root['sample'].get(k,0)) for k in ('latent','target','aim_mode','vertical','weapon','memory')}
        arm_returns={};ownership={}
        for label in ('source','release'):
            returns=[];arm_handoffs=[]
            for repeat in (0,1):
                job=next(j for j in pool['jobs'] if j['case_name']==c['name'] and j['label']==label and j['repeat']==repeat)
                assert not job['error'] and job['seed']==c['seed']
                cap=read(pin(Path(job['root'])/'report.json'));r=cap['results'][0]
                assert cap['capture_complete'] and cap['provenance_valid'] and r['capture_valid']
                base=audit_root/f"case-{job['job']:02d}";dm=read(pin(base/'cuda/report.json'))
                proof=read(pin(base/'cuda/cuda-verification.json'));assert proof['state']=='passed' and proof['device']=='cuda'
                if label=='release':
                    assert dm['diagnostic_only'] and not dm['ppo_training_ready'] and not proof['ordinary_training_eligible']
                reward=read(pin(base/'native/replay-0/reward-config.json'));assert reward==read(pin(Path(c['reward_config'])))
                assert reward['version']==config['reward_version'] and reward['aim_gamma']==config['gamma']
                assert dm['reward_config_sha256']==rh and dm['model_sha256']==mh
                steps=lines(pin(base/'native/replay-0/steps.jsonl'))
                scores={row['step']:row for row in lines(pin(base/'native/replay-0/rewards.jsonl'))}
                end=r['first_life']['end_frame'];total=0.;n=0;handoffs=[]
                boundary=r['goal_stop']['kill_frame'] if r['goal_stop'] else end
                for step in steps:
                    frame=step['observation']['identity']['frame']
                    if c['frame']<=frame<end:
                        if frame<boundary and step['owner']!='provider':handoffs.append(frame)
                        score=scores[step['index']];assert score['available']
                        total+=config['gamma']**(frame-c['frame'])*score['score'];n+=1
                assert n>0;returns.append(total);arm_handoffs.append(handoffs)
            assert abs(returns[0]-returns[1])<1e-8,'Cold reward suffix differs'
            assert arm_handoffs[0]==arm_handoffs[1],'Cold ownership differs'
            arm_returns[label]=returns[0]
            ownership[label]=arm_handoffs[0]
        delta=arm_returns['release']-arm_returns['source']
        cases.append(dict(seed=c['seed'],name=c['name'],split=c['preference_split'],observed_phase11_frames=c['observed_hold_frames'],
            ownership=ownership,eligible_for_preference=all(not frames for frames in ownership.values()),
            input=dict(seed=root['seed'],index=root['index'],frame=root['frame'],features=root['features'],sample=sample,sequence_prefix=prefix),
            target=dict(source_return=arm_returns['source'],release_return=arm_returns['release'],delta=delta,preferred_attack=preferred_attack(delta,config['margin']))))
    data=dict(version=VERSION,ordinary_ppo_eligible=False,source_model_sha256=model_hash,native_reward_config_sha256=reward_hash,
        feature_width=feature_width,label_config=config,cases=cases,inputs=inputs)
    validate(data)
    with gzip.open(out/'dataset.json.gz','wt',encoding='utf-8') as f:json.dump(data,f,separators=(',',':'))
    summary={s:dict(cases=sum(c['split']==s for c in cases),eligible=sum(c['split']==s and c['eligible_for_preference'] for c in cases),release=sum(c['split']==s and c['eligible_for_preference'] and c['target']['preferred_attack']==0 for c in cases),hold=sum(c['split']==s and c['eligible_for_preference'] and c['target']['preferred_attack']==1 for c in cases),ties=sum(c['split']==s and c['eligible_for_preference'] and c['target']['preferred_attack'] is None for c in cases)) for s in ('train','diagnostic')}
    save(out/'report.json',dict(state='passed',version=VERSION,source_model_sha256=model_hash,summary=summary,dataset_sha256=sha(out/'dataset.json.gz'),
        no_neural_inference=True,training_performed=False,scope='Observed discounted native reward suffix labels; no critic bootstrap or future input context. Whole-world preassigned engineering split, not final blind combat acceptance.'))
    print(summary)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','native','audit','config','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();build(a.protocol.resolve(),a.native.resolve(),a.audit.resolve(),a.config.resolve(),a.out.resolve())

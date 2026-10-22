"""Join mobile ground/control labels and explicit no-target fire releases.

All features remain canonical observed-only Go output. No neural execution.
"""
import argparse
import collections
import gzip
import json
import pathlib
from process_combat_architecture_pool import read,save,sha


def key(seed,identity):
    return (seed,)+tuple(identity[k] for k in ('map','connection','spawncount','actor','life','frame'))


def explicit_targetless_release(row,capture):
    """No prop/navigation/grenade aim may silently become a negative label."""
    o=capture['observation']
    return (row['native_matched'] and row['native_exclusive'] and 0<=o['observation_age_ms']<=300
        and capture.get('teacher_aim_source')=='none' and not capture.get('teacher_aim_entity')
        and not capture['proposed']['attack'] and not capture['applied']['attack']
        and not any(row['features'][426:466:5]))


def main():
    ap=argparse.ArgumentParser()
    for name in ('corpus','features','motion','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    args=ap.parse_args();assert not args.out.exists();pins={}
    def pin(path,expected=None):
        path=pathlib.Path(path).resolve();digest=sha(path);assert expected is None or expected==digest
        pins[str(path)]=digest;return path
    corpus=read(pin(args.corpus/'complete.json'));features=read(pin(args.features/'complete.json'));motion=read(pin(args.motion/'complete.json'))
    assert corpus['state']=='complete' and corpus['version']=='registered_rules_teacher_corpus_v1'
    assert features['version']=='registered_joint_teacher_features_v1' and features['state']=='complete'
    assert motion['version']=='registered_teacher_ground_motion_labels_v1' and motion['state']=='complete'
    assert pathlib.Path(features['corpus']).resolve()==pathlib.Path(motion['corpus']).resolve()==args.corpus.resolve()
    assert pathlib.Path(motion['features']).resolve()==args.features.resolve() and features['test_deferred'] and motion['test_deferred']
    for item in corpus['inputs']+features['inputs']+motion['inputs']:pin(item['path'],item['sha256'])
    raw={}
    for case in corpus['cases']:
        report=read(pathlib.Path(case['root'])/'report.json');worker=pathlib.Path(report['results'][0]['root'])
        with (worker/'bot.jsonl').open(encoding='utf-8-sig') as stream:
            for value in map(json.loads,stream):
                c=value['combat_policy'];k=key(case['seed'],c['observation']['identity']);assert k not in raw;raw[k]=c
    args.out.mkdir(parents=True);splits={};seen=set()
    for split,info in features['splits'].items():
        with gzip.open(pin(info['path'],info['sha256']),'rt',encoding='utf-8') as stream:rows=list(map(json.loads,stream))
        m=motion['splits'][split]
        with gzip.open(pin(m['path'],m['sha256']),'rt',encoding='utf-8') as stream:ground={key(r['seed'],r['identity']):r for r in map(json.loads,stream)}
        assert len(ground)==len(rows)
        seeds={r['seed'] for r in rows};assert not seen.intersection(seeds);seen.update(seeds)
        counts=collections.Counter();path=args.out/(split+'.jsonl.gz')
        with gzip.open(path,'wt',encoding='utf-8',compresslevel=6) as output:
            for r in rows:
                k=key(r['seed'],r['identity']);g=ground[k];c=raw[k]
                assert g['features']==r['features'] and c['applied']['identity']==r['identity']
                negative=bool(explicit_targetless_release(r,c))
                target=0 if negative else r['target'];target_mask=r['target_mask'] or negative
                attack=False if negative else r['attack'];attack_mask=r['attack_mask'] or negative
                if negative:assert not r['target_mask'] and not r['attack_mask']
                row=dict(seed=r['seed'],identity=r['identity'],features=r['features'],ground=g['label'],
                    decision=dict(target=target,target_mask=target_mask,attack=attack,attack_mask=attack_mask,
                        targetless_release=negative,reason='explicit_teacher_no_available_target_release' if negative else 'existing_canonical_teacher_masks'))
                counts['rows']+=1;counts['target_labels']+=target_mask;counts['attack_labels']+=attack_mask
                counts['attack_positive']+=attack_mask and attack;counts['attack_negative']+=attack_mask and not attack
                counts['targetless_release']+=negative;counts['ground_labels']+=g['label'] is not None
                if g['label']:counts['ground_'+g['label']['kind']]+=1
                output.write(json.dumps(row,allow_nan=False)+'\n')
        assert counts['targetless_release']>0 and counts['attack_positive']>0 and counts['ground_labels']>0
        splits[split]=dict(path=str(path.resolve()),sha256=sha(path),counts=dict(counts),seeds=sorted(seeds))
    for p,h in pins.items():assert sha(p)==h
    save(args.out/'complete.json',dict(state='complete',version='registered_joint_mobile_labels_v1',feature_version='combat_features_v10',
        test_deferred=True,corpus=str(args.corpus.resolve()),splits=splits,inputs=[dict(path=p,sha256=h) for p,h in pins.items()],
        script_sha256=sha(__file__),training_performed=False,neural_execution=False,promotion=False,weapon_supervision=False,vertical_supervision=False,
        scope='Canonical causal Go features and ground labels, joined by full seed/identity. Additional target0/firefalse only for native matched/exclusive, fresh explicit teacher aim none, both proposed/applied release, no observed available bbox target. Prop/navigation/grenade aim excluded. No runtime firing clamp, no optimality claim, no training or evaluation logs fitted.'))
    print(json.dumps(dict(state='complete',splits=splits)))


if __name__=='__main__':main()

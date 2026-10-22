"""Causal context + native-proven ground movement/aim teacher candidates.

No future state enters features. Observed displacement is a label-quality gate.
"""
import argparse
import gzip
import json
import math
import pathlib
from process_combat_architecture_pool import read,save,sha


def identity_key(seed,identity):
    return (seed,)+tuple(identity[k] for k in ('map','connection','spawncount','actor','life','frame'))


def main():
    ap=argparse.ArgumentParser()
    for name in ('corpus','features','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    args=ap.parse_args();assert not args.out.exists();pins={}
    def pin(path,expected=None):
        path=pathlib.Path(path).resolve();digest=sha(path);assert expected is None or expected==digest
        pins[str(path)]=digest;return path
    seal=read(pin(args.corpus/'complete.json'));meta=read(pin(args.features/'complete.json'))
    assert seal['state']=='complete' and seal['version']=='registered_rules_teacher_corpus_v1'
    assert meta['state']=='complete' and meta['version']=='registered_joint_teacher_features_v1'
    assert meta['test_deferred'] and pathlib.Path(meta['corpus']).resolve()==args.corpus.resolve()
    for binding in meta['inputs']+seal['inputs']:pin(binding['path'],binding['sha256'])
    raw={};following={}
    for case in seal['cases']:
        cap=read(pathlib.Path(case['root'])/'report.json');worker=pathlib.Path(cap['results'][0]['root'])
        with (worker/'bot.jsonl').open(encoding='utf-8-sig') as f:rows=[r['combat_policy'] for r in map(json.loads,f)]
        for c in rows:
            k=identity_key(case['seed'],c['observation']['identity']);assert k not in raw;raw[k]=c
        for c,n in zip(rows,rows[1:]):
            expected=dict(c['observation']['identity'],frame=c['observation']['identity']['frame']+1)
            if n['observation']['identity']==expected:following[identity_key(case['seed'],c['observation']['identity'])]=n
    args.out.mkdir(parents=True);splits={};seen=set()
    for split,info in meta['splits'].items():
        with gzip.open(pin(info['path'],info['sha256']),'rt',encoding='utf-8') as stream:rows=list(map(json.loads,stream))
        seeds={r['seed'] for r in rows};assert not seen.intersection(seeds);seen.update(seeds)
        counts=dict(rows=len(rows),moving=0,spacing_hold=0,fine=0,coarse=0);reject={}
        path=args.out/(split+'.jsonl.gz')
        with gzip.open(path,'wt',encoding='utf-8',compresslevel=6) as output:
            for row in rows:
                k=identity_key(row['seed'],row['identity']);c=raw[k];o=c['observation'];a=c['applied'];next_row=following.get(k)
                assert a['identity']==row['identity']
                label=None;reason='context_or_unproven'
                if row['native_matched'] and row['native_exclusive'] and row['target_mask'] and next_row is not None:
                    n=next_row['observation'];moving=math.hypot(a['forward'],a['side'])>.1
                    delta=math.hypot(n['position'][0]-o['position'][0],n['position'][1]-o['position'][1])
                    reason='not_supported_ground_motion'
                    if o['on_ground'] and n['health']>0 and a['vertical']=='release':
                        reason='blocked_or_ambiguous_motion'
                        useful_moving=moving and delta>2 and not c.get('move_limit_reason')
                        useful_hold=not moving and c.get('move_limit_reason')=='combat_spacing_hold'
                        angles=[a['yaw_delta_degrees'],a['pitch_delta_degrees']]
                        assert all(math.isfinite(v) for v in angles+[a['forward'],a['side']])
                        assert max(abs(a['forward']),abs(a['side']))<=1.
                        if useful_moving or useful_hold:
                            mode=int(max(map(abs,angles))<14.85);scale=15. if mode else 180.
                            reason='angle_outside_open_tanh_support'
                            if max(map(abs,angles))<.99*scale:
                                label=dict(movement=[a['forward'],a['side']],aim_degrees=angles,mode=mode,
                                    target=row['target'],kind='moving' if useful_moving else 'spacing_hold',
                                    observed_next_displacement=delta)
                                counts[label['kind']]+=1;counts['fine' if mode else 'coarse']+=1
                if label is None:reject[reason]=reject.get(reason,0)+1
                output.write(json.dumps(dict(seed=row['seed'],identity=row['identity'],features=row['features'],label=label),allow_nan=False)+'\n')
        assert counts['moving']>0 and counts['spacing_hold']>0,('Missing balanced motion support',split,counts)
        splits[split]=dict(path=str(path.resolve()),sha256=sha(path),counts=counts,rejections=reject)
    for path,digest in pins.items():assert sha(path)==digest
    save(args.out/'complete.json',dict(state='complete',version='registered_teacher_ground_motion_labels_v1',feature_version='combat_features_v10',
        corpus=str(args.corpus.resolve()),features=str(args.features.resolve()),test_deferred=True,splits=splits,
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),training_performed=False,
        neural_execution=False,promotion=False,vertical_supervision=False,
        scope='Observed causal Go features with past target intent; explicit available teacher target, native matched/exclusive adjacent grounded actions. Movement labels need displacement>2 and no move limit; stationary labels require explicit combat_spacing_hold. Future displacement only filters labels, never features. Teacher yaw/pitch/movement are coordinated executed commands, not optimal actions; no jump/crouch competence. All rejected rows retained as temporal context.'))
    print(json.dumps(dict(state='complete',splits=splits)))


if __name__=='__main__':main()

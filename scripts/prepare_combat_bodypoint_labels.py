"""Paired upper/center geometric labels from TRAIN corpora, with common support.

No policy inference, training, visibility oracle, or evaluation data mining.
The native bbox center is a candidate teacher, not a guaranteed visible point.
"""
import argparse
import copy
import gzip
import io
import json
import math
import pathlib
from process_combat_architecture_pool import read, save, sha


def wrap(degrees):
    return (degrees+180) % 360-180


def center_label(row):
    result=copy.deepcopy(row);label=result['offline_label'];pose=label['native_pose']
    for key in ('origin','mins','maxs'):
        assert len(pose[key])==3 and all(math.isfinite(v) for v in pose[key])
    assert all(a<b for a,b in zip(pose['mins'],pose['maxs']))
    point=[o+(a+b)/2 for o,a,b in zip(pose['origin'],pose['mins'],pose['maxs'])]
    direction=[p-m for p,m in zip(point,label['native_muzzle'])]
    norm=math.sqrt(sum(v*v for v in direction));assert norm>1e-6
    pitch=-math.degrees(math.atan2(direction[2],math.hypot(*direction[:2])))
    yaw=math.degrees(math.atan2(direction[1],direction[0]))
    desired=[pitch-label['native_recoil'][0],wrap(yaw-label['native_recoil'][1])]
    change=[wrap(a-b) for a,b in zip(desired,label['desired_view_degrees'])]
    for key in ('desired_command_delta_degrees','correction_from_executed_view_degrees'):
        label[key]=[wrap(a+b) for a,b in zip(label[key],change)]
    aim=label['native_aim'];aim_norm=math.sqrt(sum(v*v for v in aim));assert aim_norm>1e-6
    cosine=sum(a*b for a,b in zip(aim,direction))/(aim_norm*norm)
    label['native_bodypoint_error_degrees']=math.degrees(math.acos(max(-1,min(1,cosine))))
    label['source_upper_observed_bodypoint_error_degrees']=label.pop('observed_bodypoint_error_degrees')
    # No reconstructed client bbox is available in this derived-label format.
    # Do not mislabel the inherited upper-point diagnostic as center error.
    label['observed_bodypoint_error_degrees']=None
    label.update(desired_point=point,desired_view_degrees=desired,distance=norm,
        body_point='native_bbox_center',point_visibility_verified=False)
    result['version']='combat_aim_feedback_bodypoint_pair_v1'
    return result


def usable(row):
    label=row['offline_label'];scale=15. if row['sample'].get('aim_mode',0) else 180.
    desired=label['desired_command_delta_degrees']
    assert len(desired)==2 and all(math.isfinite(v) for v in desired)
    return not (label['muzzle_blocked'] or label['water'] or max(map(abs,desired))>=scale*.99)


def write_gzip(path,rows):
    with path.open('wb') as stream:
        with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as compressed:
            with io.TextIOWrapper(compressed,encoding='utf-8',newline='\n') as output:
                for row in rows:output.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True)
    args=ap.parse_args();assert not args.out.exists()
    pins={}
    def pin(path,expected=None):
        path=pathlib.Path(path);digest=sha(path);assert expected is None or expected==digest
        pins[str(path.resolve())]=digest;return path
    meta=read(pin(args.source/'report.json'));assert meta['state']=='complete' and not meta['training_performed']
    assert meta['version']=='combat_aim_feedback_labels_v2', 'Input must be original upper-point labels'
    for item in meta['inputs']:pin(item['path'],item['sha256'])
    corpus=pathlib.Path(meta['corpus']);protocol=read(pin(corpus/'protocol.json'))
    assert not protocol['diagnostic_intervention'] and set(meta['splits'])=={'train','development'}
    args.out.mkdir(parents=True)
    splits={'upper':{},'center':{}};counts={};seen=set()
    for split,binding in meta['splits'].items():
        with gzip.open(pin(binding['output'],binding['sha256']),'rt',encoding='utf-8') as stream:original=list(map(json.loads,stream))
        assert all(row['split']==split for row in original)
        seeds={row['seed'] for row in original};assert not seen.intersection(seeds);seen.update(seeds)
        upper=[];center=[];changed=[];upper_only=center_only=0
        for row in original:
            candidate=center_label(row);a,b=usable(row),usable(candidate)
            upper_only+=a and not b;center_only+=b and not a
            if not (a and b):continue
            upper.append(row);center.append(candidate)
            changed.append(abs(wrap(candidate['offline_label']['desired_command_delta_degrees'][0]-row['offline_label']['desired_command_delta_degrees'][0])))
        assert upper and {(r['seed'],r['index']) for r in upper}=={(r['seed'],r['index']) for r in center}
        counts[split]=dict(input_rows=len(original),common_rows=len(upper),excluded=len(original)-len(upper),
            upper_only_usable=upper_only,center_only_usable=center_only,
            mean_pitch_label_change_degrees=sum(changed)/len(changed),max_pitch_label_change_degrees=max(changed))
        for name,rows in [('upper',upper),('center',center)]:
            root=args.out/name;root.mkdir(exist_ok=True);destination=root/(split+'.jsonl.gz')
            write_gzip(destination,rows)
            splits[name][split]=dict(cases=len({r['seed'] for r in rows}),seeds=sorted({r['seed'] for r in rows}),
                counts=dict(labelled_shots=len(rows)),output=str(destination.resolve()),sha256=sha(destination))
    for p,h in pins.items():assert sha(p)==h
    inputs=[dict(path=p,sha256=h) for p,h in pins.items()]
    for name in splits:
        save(args.out/name/'report.json',dict(state='complete',version='combat_aim_feedback_bodypoint_pair_v1',
            corpus=str(corpus.resolve()),source_labels=str(args.source.resolve()),body_point=name,
            common_support_only=True,splits=splits[name],inputs=inputs,script_sha256=sha(__file__),
            training_performed=False,promotion=False,
            scope='Identical source histories and usable examples in both teacher branches. Native geometry labels only; center visibility and optimality are unverified. Observed center error is absent, inherited upper diagnostic separately named. Train/development only; no evaluation captures.'))
    save(args.out/'complete.json',dict(state='complete',version='combat_bodypoint_label_preparation_v1',
        splits=counts,reports=[dict(path=str((args.out/name/'report.json').resolve()),sha256=sha(args.out/name/'report.json')) for name in splits],
        inputs=inputs,script_sha256=sha(__file__),training_performed=False,promotion=False))
    print(json.dumps(counts))


if __name__=='__main__':main()

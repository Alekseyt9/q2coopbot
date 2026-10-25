"""Select fresh contact-loss TRAIN roots from verified source captures; scalar metadata only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha


def camera_relative_mean_shift(relative, yaw_degrees, magnitude=1.5):
    dx,dy=relative[:2];length=math.hypot(dx,dy)
    if not math.isfinite(length) or length<16:raise ValueError('Remembered direction too short or nonfinite')
    yaw=math.radians(yaw_degrees)
    # Native AngleVectors right=(sin(yaw),-cos(yaw)); positive sidemove
    # uses that right vector. Observation feature tails can use LEFT-positive
    # coordinates and must not be copied directly into command side means.
    return [magnitude*(dx*math.cos(yaw)+dy*math.sin(yaw))/length,
            magnitude*(dx*math.sin(yaw)-dy*math.cos(yaw))/length]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--screen',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();root=a.screen.resolve();out=a.out.resolve();assert not out.exists()
    spec,pool,seal=[read(root/p) for p in ('spec.json','pool/report.json','complete.json')]
    assert spec['version']=='combat_ready_weapon_rules_train_screen_v1'
    assert seal['native_cases']==32 and seal['cuda_cases']==16 and pool['source_unchanged']
    assert sha(root/'spec.json')==seal['spec_sha256'] and sha(root/'pool/report.json')==seal['pool_sha256']
    candidates=[];inputs={str(p):sha(p) for p in (root/'spec.json',root/'complete.json',root/'pool/report.json',Path(__file__).resolve())}
    for job in pool['jobs']:
        v=spec['variants'][job['plan_index']]
        if v['mode']!='learned':continue
        assert sha(v['model'])==spec['model_sha256']
        directory=root/f"case-{v['plan_index']:02d}";meta=read(directory/'cuda/report.json');proof=read(directory/'cuda/cuda-verification.json')
        assert proof['state']=='passed' and proof['device']=='cuda'
        assert meta['model_sha256']==spec['model_sha256'] and sha(directory/'cuda/cuda-verification.json')==meta['cuda_verification_sha256']
        native=next((directory/'native').glob('replay-*/steps.jsonl'));roll=directory/'cuda/rollout.jsonl'
        assert sha(native)==meta['source_sha256'][str(native.resolve())] and sha(roll)==meta['rollout_sha256']
        for p in (native,roll,directory/'cuda/report.json'):inputs[str(p)]=sha(p)
        with native.open(encoding='utf-8-sig') as stream:ss=[json.loads(line) for line in stream]
        with roll.open(encoding='utf-8-sig') as stream:rows={x['frame']:x for x in map(json.loads,stream)}
        for prev,s in zip(ss,ss[1:]):
            o=s['observation'];f=o['identity']['frame'];fresh=[m for m in o.get('remembered_threats') or [] if m['age_frames']<=2]
            if not (s['owner']=='provider' and o['identity']['life']==1 and f in rows and f==prev['observation']['identity']['frame']+1
                    and prev['observation']['enemies'] and not o['enemies'] and fresh and o['health']>0):continue
            target=min(fresh,key=lambda m:(m['age_frames'],m['id']));dx,dy=target['last_observed_relative'][:2];length=math.hypot(dx,dy)
            if length<16:continue
            toward=camera_relative_mean_shift(target['last_observed_relative'],o['view_angles'][1])
            candidates.append(dict(map=v['map'],episode=v['episode'],seed=v['seed'],loadout=v['label'],frame=f,
                source_plan=str(root/f"plan-{v['plan_index']:02d}.json"),source_case=str(directory),instance=v['instance'],
                observed_remembered_target=target,reference_view_yaw_degrees=o['view_angles'][1],
                candidate_mean_shift_toward=toward,candidate_mean_shift_away=[-x for x in toward],
                features_sha256=hashlib.sha256(json.dumps(rows[f]['features'],separators=(',',':')).encode()).hexdigest()))
    selected=[min((x for x in candidates if x['map']==name),key=lambda x:(x['seed'],x['frame'],x['loadout'])) for name in sorted({x['map'] for x in candidates})]
    out.mkdir(parents=True)
    result=dict(version='combat_fresh_contact_loss_train_roots_v2_right_positive',source_model=spec['variants'][0]['model'],
        source_weights_sha256=spec['model_sha256'],cases=selected,inputs=inputs,
        candidate_counts={name:sum(x['map']==name for x in candidates) for name in ('base3','city1','jail1','ware2')},
        selection='Only observed current/past state; earliest(seed,frame,loadout) permap with visible-to-occluded transition and fresh remembered target. No future outcomes used to rank roots.',
        neural_execution=False,new_native_games=0,training_performed=False,
        scope='Camera-relative constant mean shifts are candidate diagnostics, not a navigation oracle or supervised optimal-action labels. Only maps with actual eligible witnesses are selected.')
    save(out/'selection.json',result);save(out/'complete.json',dict(state='complete',selection_sha256=sha(out/'selection.json')))
    print([(x['map'],x['loadout'],x['frame']) for x in selected])


if __name__=='__main__':main()

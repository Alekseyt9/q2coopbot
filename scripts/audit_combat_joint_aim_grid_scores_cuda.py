"""Compare frozen predictor/head scores with accepted one-state TRAIN outcomes."""
import argparse
import json
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha
from ppo_combat import torch, training_devices
from combat_joint_aim_policy import JointAim, events
from combat_aim_candidates import choose, frame_uniforms


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('probe','proof','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    seal=read(pin(a.proof/'complete.json'));proof=read(pin(a.proof/'report.json',seal['report_sha256']))
    assert proof['state']=='complete' and proof['accepted_categories']>=2 and not proof['training_performed']
    closed=read(pin(a.probe/'complete.json'));spec=read(pin(a.probe/'spec.json',closed['spec_sha256']))
    prep=read(pin(Path(spec['prepared'])/'report.json',spec['prepared_report_sha256']))
    source=Path(prep['source']);model=read(pin(source/'weights.json',prep['source_weights_sha256']))
    meta=read(pin(a.probe/'case-00/cuda/report.json'))
    rows=list(map(json.loads,pin(a.probe/'case-00/cuda/rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig')))
    row=next(r for r in rows if r['frame']==146);assert row['seed']==1252025109
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    joint=JointAim(model).requires_grad_(False);x=torch.tensor([row['features']],device='cuda',dtype=torch.float32)
    cat=lambda n:torch.tensor([row['sample'].get(n,0)],device='cuda',dtype=torch.long)
    z,_,timing,active=events([row],x,cat('target'),cat('aim_mode'),cat('vertical'),cat('weapon'))
    captured=[];handle=joint.head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
    with torch.inference_mode():
        logits,eligible=joint(x,z,cat('target'),cat('aim_mode'),cat('vertical'),cat('weapon'),timing)
        assert bool(eligible[0]) and torch.equal(active,eligible)
        p=logits.double().softmax(-1)[0];hints=captured[0][0,-27:-2].double();assert hints.shape==(25,)
        u=frame_uniforms(row['sample']['sampling_seed'],146,1)
        picked=int(choose(logits.double(),torch.tensor([u[0]],device='cuda'),torch.tensor([u[1]],device='cuda'),active)[0])
        assert picked==15
        accepted=[r for r in proof['results'] if r['common_state_verified'] and len(r['shots'])==1]
        positive=[r['category'] for r in accepted if r['shots'][0]['outcome']=='live_monster_damage']
        negative=[r['category'] for r in accepted if r['shots'][0]['outcome']=='solid_contact']
        assert positive and negative and len(positive)+len(negative)==len(accepted)
        a_scores=hints[positive,None];b_scores=hints[negative][None,:]
        auc=float(((a_scores>b_scores).double()+.5*(a_scores==b_scores).double()).mean())
        hit_mass=float(p[positive].sum());tail_hit_mass=hit_mass/float(p[1:].sum())
        order=torch.argsort(hints,descending=True).tolist()
    handle.remove()
    records=[dict(category=k,head_probability=float(p[k]),predicted_hit_probability=float(hints[k].sigmoid()),
        native_outcome=next((r['shots'][0]['outcome'] for r in accepted if r['category']==k),None),accepted=k in positive+negative)
        for k in range(25)]
    for path,h in pins.items():assert sha(Path(path))==h,path
    a.out.mkdir(parents=True);report=dict(state='complete',version='combat_joint_aim_grid_scores_v1',device='cuda',
        picked_category=picked,baseline_probability=float(p[0]),positive_categories=positive,negative_categories=negative,
        hit_probability_mass=hit_mass,conditional_nonzero_hit_mass=tail_hit_mass,predictor_one_state_auc=auc,
        predictor_category_order=order,records=records,inputs=pins,script_sha256=sha(__file__),
        training_performed=False,validation_used=False,new_games=False,promotion=False,
        scope='Single targeted TRAIN state, fixed seeded spread. Frozen scores compared with accepted native alternatives only. No calibration/generalization claim, fitting or conversion of these replays to on-policy PPO.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(json.dumps({k:report[k] for k in ('picked_category','baseline_probability','positive_categories','hit_probability_mass','conditional_nonzero_hit_mass','predictor_one_state_auc','predictor_category_order')}))


if __name__=='__main__':main()

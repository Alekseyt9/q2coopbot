"""Migrate a stochastic combat policy to explicit augmented aim events on CUDA."""
import argparse
import copy
import json
import math
from pathlib import Path
from evaluate_registered_combat_models import read,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','predictor','prototype','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--baseline-probability',type=float,default=.5)
    ap.add_argument('--credit-audit',type=Path)
    a=ap.parse_args();assert not a.out.exists();pins={}
    assert math.isfinite(a.baseline_probability) and 0<a.baseline_probability<1
    def pin(path,expected=None):
        digest=sha(path);assert expected is None or digest==expected,str(path)
        pins[str(path.resolve())]=digest;return path
    seal=read(pin(a.source/'complete.json'))
    for name,key in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):pin(a.source/name,seal[key])
    ps=read(pin(a.predictor/'complete.json'));predictor=read(pin(a.predictor/'predictor.json',ps['predictor_sha256']))
    prototype=read(pin(a.prototype/'complete.json'));pin(a.prototype/'initial-head.pt',prototype['checkpoint_sha256'])
    pin(a.prototype/'report.json',prototype['report_sha256'])
    if a.baseline_probability!=.5:
        assert a.credit_audit,'Non-default initialization requires completed TRAIN credit audit'
        credit_seal=read(pin(a.credit_audit/'complete.json'))
        credit=read(pin(a.credit_audit/'report.json',credit_seal['report_sha256']))
        assert credit['state']=='complete' and credit['totals']['rows']==4061
        assert not credit['unassigned_native_effects'] and all(h['confirmed_goal'] for h in credit['handoffs'])
        assert credit['spacing_telescoping_max_error']<1e-8
    for name in ('combat_joint_aim_policy.py','combat_aim_candidates.py'):pin(Path(__file__).with_name(name))
    from ppo_combat import torch,training_devices,layers
    from ppo_recurrent import durable_json,durable_write
    from migrate_combat_navigation_cuda import branch
    from fork_conditional_weapon_exploration_cuda import equal_tree
    from combat_aim_candidates import VERSION,AimCandidateHead,OFFSETS
    from combat_joint_aim_policy import validate,base_model,JointAim
    from combat_precision_head import validate_precision_model
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    source=read(a.source/'weights.json');validate_precision_model(source)
    cp=torch.load(a.source/'checkpoint.pt',map_location='cuda',weights_only=True)
    assert cp['weights_sha256']==seal['weights_sha256']
    for key in ('actor','value'):assert equal_tree(branch(source,key).state_dict(),cp[key])
    initial=torch.load(a.prototype/'initial-head.pt',map_location='cuda',weights_only=True)
    assert initial['version']==VERSION and not initial['runtime_integrated']
    head=AimCandidateHead();head.load_state_dict(initial['model'])
    with torch.no_grad():
        head.layers[-1].bias[0]=math.log(24*a.baseline_probability/(1-a.baseline_probability))
        probabilities=head(torch.zeros((1,1359),device='cuda')).softmax(-1)[0]
        assert abs(float(probabilities[0])-a.baseline_probability)<1e-6
        assert bool(torch.isfinite(probabilities).all()) and bool((probabilities>0).all())
    model=copy.deepcopy(source);model.update(action_distribution=VERSION,
        aim_candidate_head=dict(version=VERSION,offset_degrees=list(OFFSETS),sampler='frame_hash_categorical_v1',layers=layers(head.layers),predictor=predictor))
    validate(model);assert base_model(model)==source
    try:validate_precision_model(model)
    except ValueError as exc:assert 'dedicated versioned scorer' in str(exc)
    else:raise AssertionError('Legacy scorer accepted augmented policy')
    decoded=JointAim(model)
    assert equal_tree(head.state_dict(),decoded.head.state_dict())
    opt=torch.optim.Adam(head.parameters(),lr=3e-4)
    a.out.mkdir(parents=True)
    durable_json(a.out/'weights.json',model)
    # CPU deserialization preserves saved CPU RNG bytes. All tensor comparisons
    # and initialization above run on CUDA; no old optimizer group is rewritten.
    fork=torch.load(a.source/'checkpoint.pt',map_location='cpu',weights_only=True)
    fork.update(weights_sha256=sha(a.out/'weights.json'),action_distribution=VERSION,
                aim_candidate_head=head.state_dict(),aim_candidate_optimizer=opt.state_dict())
    durable_write(a.out/'checkpoint.pt',lambda f:torch.save(fork,f))
    loaded=torch.load(a.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    for key in cp:
        if key!='weights_sha256':assert equal_tree(cp[key],loaded[key]),key
    assert equal_tree(head.state_dict(),loaded['aim_candidate_head'])
    report=dict(state='complete',version='combat_joint_aim_initialization_v1',device='cuda',action_distribution=VERSION,
        head_parameters=sum(p.numel() for p in head.parameters()),source=str(a.source.resolve()),source_weights_sha256=seal['weights_sha256'],
        baseline_probability=a.baseline_probability,actual_baseline_probability=float(probabilities[0]),
        nonbaseline_probability=1-a.baseline_probability,credit_audit=str(a.credit_audit.resolve()) if a.credit_audit else None,
        original_actor_value_std_adam_rng_history_unchanged=True,separate_head_optimizer_empty=True,legacy_scorer_rejected=True,
        inputs=pins,script_sha256=sha(__file__),training_performed=False,runtime_compatibility_verified=False,promotion=False,
        scope='New on-policy augmented distribution, not an inference-only modifier. Explicit original latent/category event. Initial weights only; native/CUDA compatibility required before TRAIN collection.')
    for path,digest in pins.items():assert sha(Path(path))==digest
    durable_json(a.out/'report.json',report)
    durable_json(a.out/'complete.json',dict(version='combat_update_complete_v1',weights_sha256=sha(a.out/'weights.json'),checkpoint_sha256=sha(a.out/'checkpoint.pt'),report_sha256=sha(a.out/'report.json')))
    print(json.dumps({k:v for k,v in report.items() if k!='inputs'}))


if __name__=='__main__':main()

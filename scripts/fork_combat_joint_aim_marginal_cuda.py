"""Diagnostic constant-marginal head; preserve all source actor/value state."""
import argparse
import copy
from pathlib import Path
from evaluate_registered_combat_models import read, sha
from ppo_combat import torch, training_devices, layers
from ppo_recurrent import durable_json, durable_write
from combat_joint_aim_policy import JointAim, validate
from fork_conditional_weapon_exploration_cuda import equal_tree
from train_combat_joint_aim_cuda import cuda_tree


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','audit','protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--baseline-probability',type=float,default=None,
        help='Diagnostic constant P(category 0), with uniform positive tail; no training or mask')
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();digest=sha(p);assert expected is None or digest==expected,str(p);pins[str(p)]=digest;return p
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    source=a.source.resolve();seal=read(pin(source/'complete.json'))
    for name,key in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):pin(source/name,seal[key])
    audit_seal=read(pin(a.audit/'complete.json'));audit=read(pin(a.audit/'report.json',audit_seal['report_sha256']))
    assert audit['state']=='complete' and audit['device']=='cuda' and not audit['training_performed'] and not audit['validation_used']
    assert audit['model_sha256']==seal['weights_sha256'] and audit['calibration_label']=='round4'
    pin(a.protocol)
    model=read(source/'weights.json');joint=JointAim(model).requires_grad_(False)
    cp=torch.load(source/'checkpoint.pt',map_location='cuda',weights_only=True)
    assert cp['weights_sha256']==seal['weights_sha256'] and equal_tree(cp['aim_candidate_head'],joint.head.state_dict())
    q=torch.tensor(audit['calibrated_marginal_probabilities'],device='cuda',dtype=torch.float64)
    if a.baseline_probability is not None:
        assert 0<a.baseline_probability<1
        q.fill_((1-a.baseline_probability)/24);q[0]=a.baseline_probability
    assert q.shape==(25,) and bool((q>0).all()) and abs(float(q.sum())-1)<1e-10
    with torch.no_grad():joint.head.layers[-1].weight.zero_();joint.head.layers[-1].bias.copy_(q.log().float())
    updated=copy.deepcopy(model);updated['aim_candidate_head']['layers']=layers(joint.head.layers);validate(updated)
    old=copy.deepcopy(model);restored=copy.deepcopy(updated)
    old['aim_candidate_head'].pop('layers');restored['aim_candidate_head'].pop('layers');assert old==restored
    assert updated['aim_candidate_head']['layers'][0]==model['aim_candidate_head']['layers'][0]
    exported=JointAim(updated).requires_grad_(False);assert equal_tree(exported.head.state_dict(),joint.head.state_dict())
    with torch.no_grad():
        test=torch.zeros((2,1359),device='cuda');test[1].fill_(1)
        predicted=exported.head(test).double().softmax(-1)
        assert torch.equal(predicted[0],predicted[1]);error=float((predicted-q).abs().max());assert error<1e-6
    head_opt=torch.optim.Adam(exported.head.parameters(),lr=.0003)
    head_opt.load_state_dict(dict(state={},param_groups=copy.deepcopy(cp['aim_candidate_optimizer']['param_groups'])))
    fork=copy.deepcopy(cp)
    fork.update(aim_candidate_head=exported.head.state_dict(),aim_candidate_optimizer=head_opt.state_dict(),
        actor_training={**cp['actor_training'],'scope':'joint_aim_constant_marginal_probe_v1'},
        diagnostic_joint_aim_fork=dict(version='joint_aim_constant_marginal_probe_v1',source_weights_sha256=seal['weights_sha256'],
            audit_report_sha256=audit_seal['report_sha256'],protocol_sha256=sha(a.protocol),head_optimizer_reset=True,training_performed=False))
    a.out.mkdir(parents=True);durable_json(a.out/'weights.json',updated);fork['weights_sha256']=sha(a.out/'weights.json')
    durable_write(a.out/'checkpoint.pt',lambda f:torch.save(fork,f))
    loaded=torch.load(a.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    changed={'weights_sha256','aim_candidate_head','aim_candidate_optimizer','actor_training'}
    for key in cp:
        if key not in changed:assert equal_tree(cp[key],cuda_tree(loaded[key])),key
    assert not loaded['aim_candidate_optimizer']['state']
    assert equal_tree(loaded['aim_candidate_head'],exported.head.state_dict())
    for path,digest in pins.items():assert sha(Path(path))==digest,str(path)
    report=dict(state='complete',version='joint_aim_constant_marginal_probe_v1',device='cuda',source=str(source),
        source_weights_sha256=seal['weights_sha256'],actor_value_std_and_their_adam_rng_history_exact=True,
        changed_model_fields=['aim_candidate_head.layers.1.weight','aim_candidate_head.layers.1.bias'],head_optimizer_reset=True,
        constant_probabilities=predicted[0].tolist(),constant_probability_max_error=error,diagnostic_only=True,
        constant_distribution_mode='train_marginal' if a.baseline_probability is None else 'explicit_baseline_control',
        requested_baseline_probability=a.baseline_probability,
        training_performed=False,promotion=False,native_runtime_verified=False,inputs=pins,script_sha256=sha(__file__),
        scope='Diagnostic constant categorical distribution. TRAIN marginal or explicitly specified baseline control, no outcome fitting. Correct stochastic distribution, not a mask. Source actor/fire unchanged; native CUDA acceptance required. Budget trainer rejects diagnostic checkpoint scope.')
    durable_json(a.out/'report.json',report);durable_json(a.out/'complete.json',dict(state='complete',
        weights_sha256=sha(a.out/'weights.json'),checkpoint_sha256=sha(a.out/'checkpoint.pt'),report_sha256=sha(a.out/'report.json')))
    print('Diagnostic marginal fork sealed on CUDA; no training')


if __name__=='__main__':main()

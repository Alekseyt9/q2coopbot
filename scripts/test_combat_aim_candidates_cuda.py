"""CUDA distribution/latent contract for the proposed joint aim head."""
import argparse
import copy
import json
import math
from pathlib import Path
from ppo_combat import torch,training_devices
from combat_aim_candidates import AimCandidateHead,inputs,eligible,choose,log_probability,entropy,apply_offsets,joint_log_probability,frame_uniforms
from build_combat_predictive_aim_dataset_cuda import candidate_for
from evaluate_registered_combat_models import sha,save


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();a.out.mkdir(parents=True)
    training_devices();torch.set_default_device('cuda');torch.manual_seed(20261023)
    head=AimCandidateHead().double()
    f=torch.zeros((1,1147),dtype=torch.float64,device='cuda',requires_grad=True)
    c=torch.zeros((1,185),dtype=torch.float64,device='cuda');c[:,77]=1;c[:,99]=1;c[:,147]=1
    hints=torch.zeros((1,25),dtype=torch.float64,device='cuda',requires_grad=True)
    t=torch.zeros((1,2),dtype=torch.float64,device='cuda')
    x=inputs(f,c,hints,t);active=eligible(c);assert bool(active.all())
    logits=head(x);p=logits.softmax(-1).detach()
    # Constructor creates FP32 parameters before double(), so log(24) roundoff
    # is allowed at normal numerical precision, not treated as behavior gain.
    assert abs(float(p[0,0])-.5)<1e-6
    assert torch.allclose(p[0,1:],torch.full((24,),1/48,dtype=torch.float64,device='cuda'),atol=1e-7,rtol=1e-6)
    n=20000;uniforms=[frame_uniforms(20261023,i,1) for i in range(1,n+1)]
    u=torch.tensor(uniforms,dtype=torch.float64,device='cuda')
    choices=choose(logits.expand(n,-1),u[:,0],u[:,1],active.expand(n))
    expected=torch.tensor([candidate_for(20261023,i,1) for i in range(1,n+1)],dtype=torch.long,device='cuda')
    assert torch.equal(choices,expected), 'Initial sampling differs from sealed exploration'
    assert len(choices.unique())==25
    # Nonuniform categorical sampling needs continuous jitter inside the old
    # offset bins, rather than keeping a fixed uniform24distribution forever.
    torch.manual_seed(20261023);arbitrary=torch.linspace(-2,2,25,dtype=torch.float64,device='cuda')[None]
    draws=200000;random=torch.rand((draws,2),dtype=torch.float64,device='cuda')
    sample=choose(arbitrary.expand(draws,-1),random[:,0],random[:,1],torch.ones(draws,dtype=torch.bool,device='cuda'))
    observed=torch.bincount(sample,minlength=25).double()/draws
    assert float((observed-arbitrary.softmax(-1)[0]).abs().max())<.004
    inactive=torch.zeros(1,dtype=torch.bool,device='cuda')
    assert int(choose(logits,u[:1,0],u[:1,1],inactive)[0])==0
    assert float(log_probability(logits,torch.zeros(1,dtype=torch.long,device='cuda'),inactive)[0].detach())==0
    assert float(entropy(logits,inactive)[0].detach())==0
    raw=torch.tensor([[.1,-.2,.1,-.1],[.2,.3,10.,-10.],[.4,-.5,.2,.3]],dtype=torch.float64,device='cuda')
    mode=torch.tensor([1,1,0],dtype=torch.long,device='cuda');choice=torch.tensor([1,0,24],dtype=torch.long,device='cuda')
    transformed=apply_offsets(raw,mode,choice)
    assert torch.equal(transformed[:,:2],raw[:,:2]) and torch.equal(transformed[1],raw[1])
    assert bool(torch.isfinite(transformed).all())
    scale=torch.tensor([15.,15.,180.],dtype=torch.float64,device='cuda')
    assert abs(float(transformed[0,3].tanh()*scale[0]-raw[0,3].tanh()*scale[0])+5)<1e-6
    # The Gaussian belongs to ORIGINAL latent; using transformed latent is a
    # different density and would corrupt PPO, especially near clipping edges.
    normal=torch.distributions.Normal(raw,torch.full_like(raw,.02))
    source=normal.log_prob(raw).sum(-1).detach();wrong=normal.log_prob(transformed).sum(-1)
    assert float((wrong-source).abs().max())>10
    l=logits.expand(3,-1);on=active.expand(3);before=joint_log_probability(source,l,choice,on).detach()
    final_fire=torch.nn.functional.logsigmoid(transformed[:,3]*3)
    raw_fire=torch.nn.functional.logsigmoid(raw[:,3]*3)
    assert float((final_fire-raw_fire).abs().max())>.1
    assert torch.allclose(joint_log_probability(source+final_fire,l,choice,on)-log_probability(l,choice,on),source+final_fire,atol=1e-8,rtol=1e-7)
    train=copy.deepcopy(head);opt=torch.optim.Adam(train.parameters(),lr=.001)
    logits2=train(x.expand(3,-1));joint=joint_log_probability(source,logits2,choice,on)
    adv=torch.tensor([1.,-.5,1.],dtype=torch.float64,device='cuda')
    loss=-((joint-before).exp()*adv).mean();opt.zero_grad();loss.backward()
    assert f.grad is None and hints.grad is None
    assert float(train.layers[-1].bias.grad.abs().sum())>0
    opt.step();after_logits=train(x.expand(3,-1));after=joint_log_probability(source,after_logits,choice,on)
    ratio=(after-before).exp()
    categorical_ratio=(log_probability(after_logits,choice,on)-log_probability(l,choice,on)).exp()
    assert torch.allclose(ratio,categorical_ratio,atol=1e-8,rtol=1e-7)
    assert float(-(ratio*adv).mean().detach())<float(loss.detach())
    cp=a.out/'initial-head.pt';torch.save(dict(version='combat_joint_aim_candidates_v1',model=head.state_dict(),runtime_integrated=False),cp)
    report=dict(state='complete',version='combat_aim_candidate_cuda_prototype_v1',device='cuda',parameters=sum(v.numel() for v in head.parameters()),
        initial_assignment_frames=n,initial_all25categories_seen=True,nonuniform_sampling_draws=draws,
        original_latent_retained=True,joint_ratio_reduces_to_categorical_when_source_frozen=True,
        conditional_fire_must_use_final_aim=True,checkpoint_sha256=sha(cp),script_sha256=sha(__file__),helper_sha256=sha(Path(__file__).with_name('combat_aim_candidates.py')),
        unavailable_branch_logprob_entropy_zero=True,source_and_hint_pathwise_gradients_detached=True,
        toy_gradient_step_only=True,combat_policy_training=False,runtime_integrated=False,promotion=False,
        scope='Prototype distribution/transform math only. One toy optimizer step on a cloned head is not combat training. Initial head is not installed into source actor or accepted by runtime/PPO processing yet.')
    (a.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json'),checkpoint_sha256=sha(cp),runtime_integrated=False,combat_policy_training=False))
    print(json.dumps(report))


if __name__=='__main__':main()

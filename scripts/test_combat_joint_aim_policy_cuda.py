"""Dedicated joint scorer tests, with nonzero action-conditioned fire on CUDA."""
import argparse
import copy
import json
from pathlib import Path
from ppo_combat import torch,nn,training_devices
from combat_joint_aim_policy import JointAim,base_model,events,probabilities
from combat_aim_candidates import apply_offsets,log_probability,entropy
from combat_precision_head import probabilities as original,validate_precision_model
from combat_conditional_fire import logits as fire_logits
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists()
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=read(a.model);joint=JointAim(model)
    assert not base_model(model).get('action_distribution')
    try:validate_precision_model(model)
    except ValueError:pass
    else:raise AssertionError('Legacy scorer accepted joint policy')
    n=3;x=torch.zeros((n,1147),device='cuda');x[:,833]=1;x[:,426]=1
    x[:,7]=1;x[:,9]=1;x[:,73]=1;x[:,75]=.5
    x[:,502:506]=torch.tensor([1.,.25,-.375,.5],device='cuda');x[:,1125]=1
    target=torch.tensor([1,1,0],device='cuda');mode=torch.ones(n,dtype=torch.long,device='cuda')
    vertical=torch.zeros(n,dtype=torch.long,device='cuda');weapon=vertical.clone()
    source=torch.tensor([[.1,-.2,.02,-.03],[.2,-.1,-.01,.02],[.3,.1,.01,.01]],device='cuda')
    choice=torch.tensor([1,5,0],device='cuda');timing=torch.tensor([[.1,1.]]*n,device='cuda')
    final64=apply_offsets(source.double(),mode,choice);final=final64.float()
    raw=torch.zeros((n,81),device='cuda');std=torch.full((4,),-4.,device='cuda');attack=torch.ones(n,device='cuda')
    fire=nn.Linear(181,1,bias=False,device='cuda')
    with torch.no_grad():fire.weight.zero_();fire.weight[0,164]=30;fire.weight[0,165]=-20
    fire.requires_grad_(False)
    lp,ent,logits,active=probabilities(joint,raw,std,source,final,attack,vertical,weapon,x,target,mode,choice,timing,fire_branch=fire)
    assert active.tolist()==[True,True,False]
    old,_=original(raw,std,source,attack,vertical,weapon,x,target,mode,fire_branch=fire)
    cat=log_probability(logits,choice,active)
    before=torch.distributions.Bernoulli(logits=fire_logits(fire,x,raw,target,mode,source,vertical,weapon))
    after=torch.distributions.Bernoulli(logits=fire_logits(fire,x,raw,target,mode,final,vertical,weapon))
    difference=after.log_prob(attack)-before.log_prob(attack)
    assert float(difference.abs().max())>.1,'Test did not exercise nonzero conditional fire'
    assert torch.allclose(lp,old+difference+cat,atol=1e-4,rtol=1e-6)
    wrong,_=original(raw,std,final,attack,vertical,weapon,x,target,mode,fire_branch=fire)
    assert float((wrong+cat-lp).abs().max().detach())>10,'Wrong transformed Gaussian not detected'
    assert float(cat[-1].detach())==float(entropy(logits,active)[-1].detach())==0
    rows=[]
    for i in range(n):
        s=dict(latent=final64[i].tolist())
        if bool(active[i]):s['aim_candidate']=dict(source_latent=source[i].double().tolist(),candidate=int(choice[i]),candidates=25,log_probability=float(cat[i].detach()),timing=timing[i].double().tolist())
        rows.append(dict(sample=s))
    z,k,t,on=events(rows,x,target,mode,vertical,weapon)
    assert torch.equal(z,source) and torch.equal(k,choice) and torch.equal(on,active)
    for mutate in ('missing','transform','category','nonfinite'):
        bad=copy.deepcopy(rows)
        if mutate=='missing':bad[0]['sample'].pop('aim_candidate')
        elif mutate=='transform':bad[0]['sample']['latent'][0]+=.2
        elif mutate=='category':bad[0]['sample']['aim_candidate']['candidate']=25
        else:bad[0]['sample']['aim_candidate']['timing'][0]=float('nan')
        try:events(bad,x,target,mode,vertical,weapon)
        except (AssertionError,ValueError):pass
        else:raise AssertionError('Bad event accepted: '+mutate)
    joint.zero_grad();(-lp[:2].mean()).backward()
    assert joint.head.layers[-1].bias.grad is not None and float(joint.head.layers[-1].bias.grad.abs().sum())>0
    assert all(p.grad is None for p in joint.hints.parameters())
    a.out.mkdir(parents=True)
    report=dict(state='complete',device='cuda',model_sha256=sha(a.model),script_sha256=sha(__file__),helper_sha256=sha(Path(__file__).with_name('combat_joint_aim_policy.py')),
        nonzero_final_fire_test=True,wrong_transformed_gaussian_rejected=True,malformed_events_rejected=4,
        candidate_gradient_nonzero=True,predictor_gradients_absent=True,inactive_contribution_zero=True,
        training_performed=False,native_compatibility=False,scope='Synthetic CUDA joint likelihood and event contract only, not combat quality.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(json.dumps(report))


if __name__=='__main__':main()

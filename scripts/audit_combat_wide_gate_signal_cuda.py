"""Read-only CUDA attribution of gate learning on already-consumed TRAIN data."""
import argparse,json,math
from pathlib import Path
from collections import defaultdict
from ppo_combat import torch,training_devices,advantages
from combat_wide_aim import WideAim,events
from train_combat_full_joint_ppo_cuda import admit
from evaluate_registered_combat_models import read,save,sha

def stats(x):
    if not x.numel():return dict(count=0)
    x=x.double()
    return dict(count=x.numel(),mean=float(x.mean()),std=float(x.std(unbiased=False)),minimum=float(x.min()),maximum=float(x.max()))

def audit(root,pins):
    protocol=read(root/'protocol.json');source=Path(protocol['source']);source_model=read(source/'weights.json');fitted=read(root/'fit/weights.json');config=protocol['config']
    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path)
        assert expected is None or digest==expected,str(path)
        pins[str(path)]=digest;return path
    request=read(pin(root/'train-request.json'))
    rows,_,_,seeds,_=admit(request,sha(pin(source/'weights.json')),[],[],pin)
    pin(root/'fit/weights.json');pin(root/'fit/report.json')
    x=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float64)
    cat=lambda k:torch.tensor([r['sample'].get(k,0) for r in rows],device='cuda',dtype=torch.long)
    target,mode,vertical,weapon=(cat(k) for k in ('target','aim_mode','vertical','weapon'))
    data=events(source_model,rows,x,target,mode,vertical,weapon);active=data['active'];choice=data['choice'].double()
    with torch.no_grad():
        before=WideAim(source_model['wide_aim']).double();after=WideAim(fitted['wide_aim']).double()
        old_logits,old_active=before(x,data['source'],target,mode,vertical,weapon)
        new_logits,new_active=after(x,data['source'],target,mode,vertical,weapon)
        assert torch.equal(active,old_active) and torch.equal(active,new_active)
        recorded=torch.tensor([r['sample'].get('wide_aim',{}).get('logit',0.) for r in rows],device='cuda',dtype=torch.float64)
        error=float((recorded[active]-old_logits[active]).abs().max());assert error<1e-8
        p0,p1=old_logits.sigmoid(),new_logits.sigmoid()
        adv,ret=advantages(rows,config['gamma'],config['lambda'])
        a=torch.tensor(adv,device='cuda',dtype=torch.float32);z=((a-a.mean())/(a.std(unbiased=False)+1e-8)).double()
        # At the behavior policy: derivative of the gate's unclipped surrogate
        # with respect to each logit, before multiplication by network Jacobian.
        score=z*(choice-p0)
        kl=p0*(p0.log()-p1.log())+(1-p0)*((1-p0).log()-(1-p1).log())
        old_lp=torch.distributions.Bernoulli(logits=old_logits).log_prob(choice)
        new_lp=torch.distributions.Bernoulli(logits=new_logits).log_prob(choice)
        ratio=(new_lp-old_lp).exp()
        new_choices=(data['uniforms']<p1).long()
        eligible=torch.where(active)[0].tolist();per_seed=defaultdict(list);records=[]
        for i in eligible:
            r=rows[i];record=dict(seed=r['seed'],frame=r['frame'],choice=int(choice[i]),p_before=float(p0[i]),p_after=float(p1[i]),uniform=float(data['uniforms'][i]),raw_advantage=float(a[i]),normalized_advantage=float(z[i]),bias_score_gradient=float(score[i]),reward=r['reward'],value=r['sample']['value'],next_value=r['next_value'],terminal=r['terminal'],counterfactual_choice_after_fit=int(new_choices[i]))
            records.append(record);per_seed[r['seed']].append(record)
        groups={}
        for k in (0,1):
            mask=active&(data['choice']==k)
            groups[str(k)]=dict(sample_attack_requests=sum(bool(rows[i]['sample']['attack']) for i in torch.where(mask)[0].tolist()),raw_advantage=stats(a[mask]),normalized_advantage=stats(z[mask]),bias_score=stats(score[mask]),probability_before=stats(p0[mask]),probability_after=stats(p1[mask]))
        seed_summary=[dict(seed=seed,events=len(rr),selected=sum(r['choice'] for r in rr),signed_bias_score=sum(r['bias_score_gradient'] for r in rr),absolute_bias_score=sum(abs(r['bias_score_gradient']) for r in rr)) for seed,rr in per_seed.items()]
        seed_summary.sort(key=lambda r:r['absolute_bias_score'],reverse=True)
        result=dict(rows=len(rows),seeds=len(seeds),active_events=int(active.sum()),selected_events=int((active&(choice==1)).sum()),native_logit_max_error=error,active_fraction=float(active.double().mean()),global_advantage=stats(a),active_advantage=stats(a[active]),choice_groups=groups,bias_score_sum=float(score[active].sum()),gate_score_dilution=float(active.double().mean()),active_bernoulli_kl=float(kl[active].mean()),global_bernoulli_kl=float(torch.where(active,kl,0.).mean()),active_ratio=stats(ratio[active]),active_clipped_fraction=float((ratio[active].sub(1).abs()>config['clip']).double().mean()),counterfactual_train_choice_changes=int(((new_choices!=data['choice'])&active).sum()),per_seed=seed_summary,events=records,
            scope='Consumed TRAIN attribution only. New probabilities evaluated on original observed states; counterfactual choice flips are NOT native trajectories or quality results. Bias score is not a causal estimate of aim value. No fitting, no optimizer mutation, no training export.')
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--batch',type=Path,action='append',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert not a.out.exists()
    training_devices();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pins={};results={str(p):audit(p,pins) for p in a.batch}
    for p,h in pins.items():assert sha(p)==h
    a.out.mkdir(parents=True)
    save(a.out/'report.json',dict(state='complete',device='cuda',diagnostic_only=True,training_performed=False,quality_evaluated=False,promotion=False,inputs=pins,batches=results))
    save(a.out/'complete.json',dict(state='complete',diagnostic_only=True,report_sha256=sha(a.out/'report.json'),training_performed=False,promotion=False))
    for name,r in results.items():print(name,json.dumps({k:r[k] for k in ('rows','active_events','selected_events','active_fraction','bias_score_sum','active_bernoulli_kl','global_bernoulli_kl','active_clipped_fraction','counterfactual_train_choice_changes')}))

if __name__=='__main__':main()

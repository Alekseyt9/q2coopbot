"""Fixed500 pair-ranking comparison using causal geometry deltas, CUDA only."""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
from evaluate_registered_combat_models import read,save,sha
from ppo_combat import torch,nn,training_devices,layers
from ppo_recurrent import durable_write
from combat_joint_aim_policy import JointAim
from combat_aim_candidates import apply_offsets,OFFSETS
from combat_action_effect import inputs as motor_inputs,projected_geometry
from probe_combat_shared_candidate_scorers_cuda import Shared
from fork_conditional_weapon_exploration_cuda import equal_tree


class Geometry(nn.Module):
    def __init__(self,beta,linear):
        super().__init__();self.linear=linear
        self.network=nn.Linear(31,1,bias=False,device='cuda') if linear else nn.Sequential(nn.Linear(31,8,device='cuda'),nn.ReLU(),nn.Linear(8,1,device='cuda'))
        with torch.no_grad():
            if linear:self.network.weight.zero_()
            else:self.network[-1].weight.zero_();self.network[-1].bias.zero_()
        self.beta=nn.Parameter(torch.tensor(beta,device='cuda'));mask=torch.zeros(25,device='cuda');mask[0]=1;self.register_buffer('baseline',mask)
    def forward(self,flat,delta,hints):return self.network(delta).squeeze(-1).clamp(-20,20)+self.beta*self.baseline
    def project(self):
        with torch.no_grad():self.beta.clamp_(-20,20)
    def optimizer(self):return torch.optim.Adam([dict(params=self.network.parameters(),lr=.01 if self.linear else .003,weight_decay=.0001),dict(params=[self.beta],lr=.05,weight_decay=0)])
    def export(self):
        r=dict(kind='geometry_candidate_delta31_linear_v1' if self.linear else 'geometry_candidate_delta31_mlp_v1',beta=float(self.beta.detach()),score_clip=20,beta_bounds=[-20,20])
        if self.linear:r['weight']=self.network.weight.detach()[0].tolist()
        else:r['layers']=layers(self.network)
        return r


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('source','dataset','protocol','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    seal=read(pin(a.source/'complete.json'))
    for n,k in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):pin(a.source/n,seal[k])
    model=read(a.source/'weights.json');ds=read(pin(a.dataset/'complete.json'));meta=read(pin(a.dataset/'report.json',ds['report_sha256']))
    assert meta['version']=='combat_native_immediate_pair_dataset_v1' and meta['fit_admission_passed'] and meta['source_weights_sha256']==seal['weights_sha256']
    rows=list(map(json.loads,pin(a.dataset/'contexts.jsonl',ds['labels_sha256']).open(encoding='utf-8-sig')));pin(a.protocol)
    assert len(rows)==meta['contexts'];train=[i for i,r in enumerate(rows) if r['split']=='train'];dev=[i for i,r in enumerate(rows) if r['split']=='development']
    fresh=[i for i in dev if rows[i]['cohort']=='policy-v2'];assert fresh and not {rows[i]['seed'] for i in train}&{rows[i]['seed'] for i in dev}
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):source=JointAim(model).requires_grad_(False)
    source_before=copy.deepcopy(source.state_dict());count=len(rows)
    x=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float32)
    z=torch.tensor([r['sample']['aim_candidate']['source_latent'] for r in rows],device='cuda',dtype=torch.float32)
    cat=lambda k:torch.tensor([r['sample'].get(k,0) for r in rows],device='cuda',dtype=torch.long)
    timing=torch.tensor([r['sample']['aim_candidate']['timing'] for r in rows],device='cuda',dtype=torch.float32)
    target,mode,vertical,weapon=(cat(k) for k in ('target','aim_mode','vertical','weapon'))
    captured=[];hook=source.head.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach()))
    with torch.inference_mode():old_logits,active=source(x,z,target,mode,vertical,weapon,timing)
    hook.remove();assert bool(active.all());flat=captured[0].clone();hints=flat[:,-27:-2].clone();old=old_logits.softmax(-1).clone()
    c=source.action_context(x,z,target,mode,vertical,weapon);expanded=c[:,None,:].expand(-1,25,-1).clone()
    ids=torch.arange(25,device='cuda').expand(count,-1);modes=mode[:,None].expand(-1,25).reshape(-1)
    final=apply_offsets(z[:,None,:].expand(-1,25,-1).reshape(-1,4).double(),modes,ids.reshape(-1));scale=torch.where(modes==1,15./180.,1.).double()
    expanded[:,:,164:166]=(final[:,2:].tanh()*scale[:,None]).reshape(count,25,2).float()
    offsets=torch.tensor(OFFSETS,device='cuda');deltas=torch.stack((offsets[ids//5],offsets[ids%5]),-1)
    predictor=model['aim_candidate_head']['predictor'];mean=torch.tensor(predictor['mean'][:185],device='cuda');std=torch.tensor(predictor['std'][:185],device='cuda')
    probability=hints.sigmoid()
    raw=torch.cat((((expanded-mean)/std).clamp(-20,20),deltas/5,probability[:,:,None],probability[:,:1,None].expand(-1,25,-1),(probability-probability[:,:1])[:,:,None],(ids==0).float()[:,:,None]),-1).detach().clone()
    with torch.no_grad():
        causal=expanded.reshape(-1,185).double();t=timing[:,None,:].expand(-1,25,-1).reshape(-1,2).double()
        motor=source.hints.motor((motor_inputs(causal,t)-source.hints.motor_mean)/source.hints.motor_std)
        geometry=projected_geometry(causal,motor,False).reshape(count,25,29)
        checked=(((torch.cat((causal,geometry.reshape(-1,29)),-1)-source.hints.shot_mean)/source.hints.shot_std)*source.hints.shot_weight).sum(-1)+source.hints.shot_bias
        assert torch.equal(checked.reshape(count,25).float(),hints),'Geometry differs from frozen hint forward'
        relative=((geometry-geometry[:,:1])/source.hints.shot_std[185:214]).clamp(-20,20).float()
        compact=torch.cat((relative,(probability-probability[:,:1])[:,:,None],(deltas.square().sum(-1)/25)[:,:,None]),-1).detach().clone()
    assert compact.shape==(count,25,31) and raw.shape==(count,25,191) and bool(torch.isfinite(compact).all())
    known=torch.zeros((count,25),device='cuda',dtype=torch.bool);hp=torch.zeros((count,25),device='cuda')
    for i,r in enumerate(rows):
        for v in r['alternatives']:known[i,v['category']]=True;hp[i,v['category']]=v['health_damage']
    pairs=known[:,:,None]&known[:,None,:]&(hp[:,:,None]>hp[:,None,:]);pair_count=pairs.sum((-1,-2))
    tr=torch.tensor(train,device='cuda',dtype=torch.long);rank_train=tr[pair_count[tr]>0];assert len(rank_train)>=4
    positive=known&(hp>hp[:,:1]);harmful=known&(hp<hp[:,:1])
    groups={'train':train,'development':dev,'fresh_development':fresh,'legacy_development':[i for i in dev if i not in fresh]}
    def metrics(prob,idx,logits=None):
        p=prob.double();v=dict(contexts=len(idx),useful_mass=float((p*positive).sum(-1)[idx].mean()),harmful_mass=float((p*harmful).sum(-1)[idx].mean()),
            known_mass=float((p*known).sum(-1)[idx].mean()),baseline_probability=float(p[idx,0].mean()),
            unmeasured_argmax_contexts=int((~known.gather(1,p.argmax(-1)[:,None])[:,0])[idx].sum()))
        if logits is not None:
            valid=[i for i in idx if int(pair_count[i])>0];margin=logits[:,:,None]-logits[:,None,:]
            v['pair_contexts']=len(valid);v['pair_accuracy']=float((((margin>0)*pairs).sum((-1,-2))/pair_count.clamp_min(1))[valid].mean()) if valid else None
        return v
    source_metrics={n:metrics(old,i,old_logits) for n,i in groups.items()}
    p0=float(old[tr,0].mean());beta=math.log(24*p0/(1-p0));a.out.mkdir(parents=True);reports={}
    for name in ('linear31','geometry8','raw191'):
        def build():
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(20261023);return Shared(beta) if name=='raw191' else Geometry(beta,name=='linear31')
        module=build();optimizer=module.optimizer();inputs=raw if name=='raw191' else compact
        for step in range(500):
            logits=module(flat,inputs,hints);logp=logits.log_softmax(-1)
            loss_per=(nn.functional.softplus(logits[:,None,:]-logits[:,:,None])*pairs).sum((-1,-2))/pair_count.clamp_min(1)
            rank_loss=loss_per[rank_train].mean();kl=(old[tr]*(old_logits[tr].log_softmax(-1)-logp[tr])).sum(-1).mean();loss=rank_loss+.05*kl
            assert bool(torch.isfinite(loss));optimizer.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(module.parameters(),.5)
            assert bool(torch.isfinite(norm));optimizer.step();module.project()
        with torch.no_grad():logits=module(flat,inputs,hints);prob=logits.softmax(-1);q=prob[tr].double().mean(0);q=q/q.sum()
        assert bool(torch.isfinite(prob).all()) and bool((prob>0).all()) and equal_tree(source_before,source.state_dict())
        statistics={n:metrics(prob,i,logits) for n,i in groups.items()};controls={n:metrics(q[None,:].expand_as(prob),i) for n,i in groups.items()}
        d,f=statistics['development'],statistics['fresh_development'];s,fs=source_metrics['development'],source_metrics['fresh_development'];co,fc=controls['development'],controls['fresh_development']
        checks=dict(useful_source=d['useful_mass']>=s['useful_mass']+.05,useful_constant=d['useful_mass']>=co['useful_mass']+.02,harmful=d['harmful_mass']<=s['harmful_mass']+.05,
                    fresh_useful_source=f['useful_mass']>=fs['useful_mass']+.02,fresh_useful_constant=f['useful_mass']>=fc['useful_mass'],fresh_harmful=f['harmful_mass']<=fs['harmful_mass']+.05)
        directory=a.out/name;directory.mkdir();durable_write(directory/'head.pt',lambda stream:torch.save(dict(module=module.state_dict(),optimizer=optimizer.state_dict(),steps=500),stream))
        restored=build();restored.load_state_dict(torch.load(directory/'head.pt',map_location='cuda',weights_only=True)['module'])
        with torch.no_grad():assert torch.equal(restored(flat,inputs,hints),logits)
        save(directory/'prototype.json',dict(kind='combat_native_candidate_score_prototype_v1',variant=name,head=module.export(),native_runtime_compatible=False,
             source_weights_sha256=seal['weights_sha256'],dataset_sha256=ds['labels_sha256']))
        save(directory/'contexts.json',[dict(seed=r['seed'],frame=r['frame'],split=r['split'],cohort=r['cohort'],probabilities=prob[i].tolist(),constant_probabilities=q.tolist(),known_actions=r['alternatives']) for i,r in enumerate(rows)])
        report=dict(state='complete',variant=name,device='cuda',steps=500,parameters=sum(p.numel() for p in module.parameters()),statistics=statistics,constant=controls,source=source_metrics,
                    final_rank_loss=float(rank_loss.detach()),final_kl=float(kl.detach()),checks=checks,gate=all(checks.values()),roundtrip_verified=True,native_runtime_compatible=False,training_performed=True,promotion=False)
        save(directory/'report.json',report);save(directory/'complete.json',dict(state='complete',report_sha256=sha(directory/'report.json'),prototype_sha256=sha(directory/'prototype.json'),checkpoint_sha256=sha(directory/'head.pt')));reports[name]=report
    for p,h in pins.items():assert sha(p)==h,p
    eligible=sorted((n for n,r in reports.items() if r['gate']),key=lambda n:(-(reports[n]['statistics']['development']['useful_mass']-reports[n]['constant']['development']['useful_mass']),reports[n]['statistics']['development']['harmful_mass'],n))
    frozen=a.out/'sources';frozen.mkdir()
    for name in (Path(__file__).name,'combat_joint_aim_policy.py','combat_action_effect.py','combat_action_geometry.py','probe_combat_shared_candidate_scorers_cuda.py'):shutil.copy2(Path(__file__).with_name(name),frozen/name)
    save(a.out/'report.json',dict(state='complete',version='combat_geometry_pair_rank_comparison_v1',device='cuda',variants=reports,eligible=eligible,selected_for_runtime=eligible[0] if eligible else None,
         inputs=pins,script_sha256=sha(__file__),training_performed=True,native_games=0,promotion=False,scope='Fixed500 prototypes on reused immediate TRAIN labels. No expected return, playable new policy, or native quality claim.'))
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(json.dumps(dict(eligible=eligible,variants={n:dict(parameters=r['parameters'],development=r['statistics']['development'],constant=r['constant']['development'],fresh=r['statistics']['fresh_development'],gate=r['gate']) for n,r in reports.items()}),indent=2))


if __name__=='__main__':main()

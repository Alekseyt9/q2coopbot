"""Fixed500 CUDA comparison of flat, calibrated and action-shared aim heads."""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import time
from evaluate_registered_combat_models import read,save,sha
from ppo_combat import torch,nn,training_devices,layers
from ppo_recurrent import durable_write
from combat_joint_aim_policy import JointAim
from combat_aim_candidates import apply_offsets,OFFSETS
from fork_conditional_weapon_exploration_cuda import equal_tree


class Flat(nn.Module):
    def __init__(self,source):
        super().__init__();self.network=copy.deepcopy(source.head).requires_grad_(True)
    def forward(self,flat,shared,hints):return self.network(flat)
    def project(self):pass
    def optimizer(self):return torch.optim.Adam(self.parameters(),lr=.0003,weight_decay=.0001)
    def export(self):return dict(kind='flat1359x64x25',layers=layers(self.network.layers))


class Calibrator(nn.Module):
    def __init__(self,beta):
        super().__init__();self.alpha=nn.Parameter(torch.zeros((),device='cuda'));self.beta=nn.Parameter(torch.tensor(beta,device='cuda'))
        mask=torch.zeros(25,device='cuda');mask[0]=1;self.register_buffer('baseline',mask)
    def forward(self,flat,shared,hints):
        p=hints.sigmoid();return self.alpha*(p-p[:,:1])+self.beta*self.baseline
    def project(self):
        with torch.no_grad():self.alpha.clamp_(0,20);self.beta.clamp_(-20,20)
    def optimizer(self):return torch.optim.Adam(self.parameters(),lr=.05)
    def export(self):return dict(kind='hint_probability_calibration_v1',alpha=float(self.alpha.detach()),beta=float(self.beta.detach()),alpha_bounds=[0,20],beta_bounds=[-20,20])


class Shared(nn.Module):
    def __init__(self,beta):
        super().__init__();self.network=nn.Sequential(nn.Linear(191,16,device='cuda'),nn.ReLU(),nn.Linear(16,1,device='cuda'))
        with torch.no_grad():self.network[-1].weight.zero_();self.network[-1].bias.zero_()
        self.beta=nn.Parameter(torch.tensor(beta,device='cuda'));mask=torch.zeros(25,device='cuda');mask[0]=1;self.register_buffer('baseline',mask)
    def forward(self,flat,shared,hints):
        score=self.network(shared).squeeze(-1).clamp(-20,20);return score+self.beta*self.baseline
    def project(self):
        with torch.no_grad():self.beta.clamp_(-20,20)
    def optimizer(self):return torch.optim.Adam([dict(params=self.network.parameters(),lr=.003,weight_decay=.0001),dict(params=[self.beta],lr=.05,weight_decay=0)])
    def export(self):return dict(kind='shared_candidate_causal191_v1',layers=layers(self.network),beta=float(self.beta.detach()),score_clip=20,beta_bounds=[-20,20])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('source','dataset','protocol','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    seal=read(pin(a.source/'complete.json'))
    for n,k in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):pin(a.source/n,seal[k])
    model=read(a.source/'weights.json');ds=read(pin(a.dataset/'complete.json'));meta=read(pin(a.dataset/'report.json',ds['report_sha256']))
    assert meta['fit_admission_passed'] and meta['contexts']>=17 and not meta['training_performed']
    rows=list(map(json.loads,pin(a.dataset/'contexts.jsonl',ds['labels_sha256']).open(encoding='utf-8-sig')));pin(a.protocol)
    train=[i for i,r in enumerate(rows) if r['split']=='train'];dev=[i for i,r in enumerate(rows) if r['split']=='development']
    count=len(rows);late=meta['version']=='combat_native_aim_pair_dataset_v3_late'
    assert count==meta['contexts'] and train and dev and not {rows[i]['seed'] for i in train}&{rows[i]['seed'] for i in dev}
    if not late:assert count==17 and len(train)==12 and len(dev)==5
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):source=JointAim(model).requires_grad_(False)
    source_before=copy.deepcopy(source.state_dict())
    x=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float32)
    cat=lambda n:torch.tensor([r['sample'].get(n,0) for r in rows],device='cuda',dtype=torch.long)
    z=torch.tensor([r['sample'].get('aim_candidate',{}).get('source_latent',r['sample']['latent']) for r in rows],device='cuda',dtype=torch.float32)
    timing=torch.tensor([r['sample']['aim_candidate']['timing'] for r in rows],device='cuda',dtype=torch.float32)
    target,mode,vertical,weapon=(cat(k) for k in ('target','aim_mode','vertical','weapon'))
    captured=[];handle=source.head.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach()))
    with torch.inference_mode():old_logits,active=source(x,z,target,mode,vertical,weapon,timing)
    handle.remove();assert bool(active.all());flat=captured[0].clone();hints=flat[:,-27:-2].clone();old_p=old_logits.softmax(-1).clone()
    c=source.action_context(x,z,target,mode,vertical,weapon);expanded=c[:,None,:].expand(-1,25,-1).clone()
    ids=torch.arange(25,device='cuda').expand(count,-1);modes=mode[:,None].expand(-1,25).reshape(-1)
    final=apply_offsets(z[:,None,:].expand(-1,25,-1).reshape(-1,4).double(),modes,ids.reshape(-1))
    scale=torch.where(modes==1,15./180.,1.).double()
    expanded[:,:,164:166]=(final[:,2:].tanh()*scale[:,None]).reshape(count,25,2).float()
    p=hints.sigmoid();offsets=torch.tensor(OFFSETS,device='cuda');deltas=torch.stack((offsets[ids//5],offsets[ids%5]),-1)/5
    predictor=model['aim_candidate_head']['predictor'];mean=torch.tensor(predictor['mean'][:185],device='cuda');std=torch.tensor(predictor['std'][:185],device='cuda')
    normalized=((expanded-mean)/std).clamp(-20,20)
    shared=torch.cat((normalized,deltas,p[:,:,None],p[:,:1,None].expand(-1,25,-1),(p-p[:,:1])[:,:,None],(ids==0).float()[:,:,None]),-1).detach().clone()
    assert shared.shape==(count,25,191) and bool(torch.isfinite(shared).all())
    train_idx=torch.tensor(train,device='cuda',dtype=torch.long);dev_idx=torch.tensor(dev,device='cuda',dtype=torch.long)
    known=torch.zeros((count,25),device='cuda',dtype=torch.bool);health=torch.zeros((count,25),device='cuda')
    for i,r in enumerate(rows):
        for v in r['alternatives']:known[i,v['category']]=True;health[i,v['category']]=v['health_damage']
    base_hp=health[:,0];positive=known&(health>base_hp[:,None]);harmful=known&(health<base_hp[:,None])
    preference=torch.zeros((len(train),25),device='cuda')
    for ti,i in enumerate(train):
        if bool(positive[i].any()):best=known[i]&(health[i]==health[i,known[i]].max());preference[ti,best]=1/int(best.sum())
        else:preference[ti,0]=1
    teacher=.1*old_p[train_idx]+.9*preference;assert bool((teacher>0).all())
    p0=float(old_p[train_idx,0].mean());beta=math.log(24*p0/(1-p0))
    def metrics(prob,indices):
        prob=prob.double();coverage=(prob*known).sum(-1)
        return dict(contexts=len(indices),known_mass=float(coverage[indices].mean()),
            useful_mass=float((prob*positive).sum(-1)[indices].mean()),harmful_mass=float((prob*harmful).sum(-1)[indices].mean()),
            conditional_known_hp=float(((prob*known*health).sum(-1)/coverage.clamp_min(1e-20))[indices].mean()),
            baseline_probability=float(prob[indices,0].mean()),unmeasured_argmax_contexts=int((~known.gather(1,prob.argmax(-1)[:,None])[:,0])[indices].sum()))
    groups={'train':train,'development':dev}
    if late:
        groups['fresh_development']=[i for i in dev if rows[i].get('cohort')=='late-v1']
        groups['parent_development']=[i for i in dev if rows[i].get('cohort')!='late-v1']
        assert groups['fresh_development'] and len(groups['parent_development'])==5
    source_metrics={s:metrics(old_p,idx) for s,idx in groups.items()}
    a.out.mkdir(parents=True);save(a.out/'protocol.json',dict(version='combat_shared_candidate_comparison_v1',protocol_sha256=sha(a.protocol),
        source_weights_sha256=seal['weights_sha256'],dataset_sha256=ds['labels_sha256'],steps=500,initialization_seed=20261023,
        normalization=dict(mean=mean.tolist(),std=std.tolist(),causal_clamp=20,offset_scale=5,input_width=191),inputs=pins))
    reports={}
    try:
        for name in ('flat','calibrator','shared'):
            save(a.out/'progress.json',dict(stage='cuda-fit',variant=name,steps=500))
            def build():
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(20261023)
                    return Flat(source) if name=='flat' else Calibrator(beta) if name=='calibrator' else Shared(beta)
            module=build();optimizer=module.optimizer();initial=copy.deepcopy(module.state_dict());started=time.perf_counter()
            with torch.no_grad():initial_loss=float(-(teacher*module(flat[train_idx],shared[train_idx],hints[train_idx]).log_softmax(-1)).sum(-1).mean())
            norms=[]
            for step in range(500):
                logits=module(flat[train_idx],shared[train_idx],hints[train_idx]);loss=-(teacher*logits.log_softmax(-1)).sum(-1).mean()
                assert bool(torch.isfinite(loss));optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=nn.utils.clip_grad_norm_(module.parameters(),.5);assert bool(torch.isfinite(norm));norms.append(float(norm));optimizer.step();module.project()
            torch.cuda.synchronize();seconds=time.perf_counter()-started
            with torch.no_grad():
                logits=module(flat,shared,hints);probs=logits.softmax(-1);assert bool(torch.isfinite(logits).all()) and bool((probs>0).all())
                final_loss=float(-(teacher*logits[train_idx].log_softmax(-1)).sum(-1).mean())
                q=probs[train_idx].double().mean(0);q=q/q.sum();constant=q[None,:].expand_as(probs)
            assert not equal_tree(initial,module.state_dict()) and equal_tree(source_before,source.state_dict())
            statistics={s:metrics(probs,idx) for s,idx in groups.items()}
            controls={s:metrics(constant,idx) for s,idx in groups.items()}
            d=statistics['development'];reference=source_metrics['development'];control=controls['development']
            gain=dict(useful_vs_source=d['useful_mass']-reference['useful_mass'],useful_vs_constant=d['useful_mass']-control['useful_mass'],harmful_vs_source=d['harmful_mass']-reference['harmful_mass'])
            gate=gain['useful_vs_source']>=.05 and gain['useful_vs_constant']>=.02 and gain['harmful_vs_source']<=.05
            fresh_guard=True
            if late:
                f,fs,fc=(g['fresh_development'] for g in (statistics,source_metrics,controls))
                fresh_guard=f['useful_mass']>=fs['useful_mass']+.02 and f['useful_mass']>=fc['useful_mass'] and f['harmful_mass']<=fs['harmful_mass']+.05
                gate=gate and fresh_guard
            directory=a.out/name;directory.mkdir()
            durable_write(directory/'head.pt',lambda f:torch.save(dict(module=module.state_dict(),optimizer=optimizer.state_dict(),steps=500),f))
            restored=build();loaded=torch.load(directory/'head.pt',map_location='cuda',weights_only=True);restored.load_state_dict(loaded['module'])
            with torch.no_grad():assert torch.equal(restored(flat,shared,hints),logits)
            prototype=dict(kind='combat_native_candidate_score_prototype_v1',variant=name,head=module.export(),
                native_runtime_compatible=False,source_weights_sha256=seal['weights_sha256'],dataset_sha256=ds['labels_sha256'])
            save(directory/'prototype.json',prototype)
            per_context=[dict(context_id=r['context_id'],seed=r['seed'],frame=r['frame'],geometry=r['geometry'],split=r['split'],
                source_probabilities=old_p[i].tolist(),probabilities=probs[i].tolist(),constant_probabilities=q.tolist(),
                known_actions=r['alternatives'],argmax=int(probs[i].argmax())) for i,r in enumerate(rows)]
            save(directory/'contexts.json',per_context)
            report=dict(state='complete',version='combat_shared_candidate_comparison_v1',variant=name,device='cuda',steps=500,
                parameters=sum(p.numel() for p in module.parameters()),initial_loss=initial_loss,final_loss=final_loss,fit_seconds=seconds,
                fresh_guard=bool(fresh_guard),dataset_version=meta['version'],
                gradient_norm_max=max(norms),statistics=statistics,constant=controls,source=source_metrics,gate=bool(gate),gate_deltas=gain,
                native_runtime_compatible=False,training_performed=True,promotion=False,roundtrip_verified=True)
            save(directory/'report.json',report);save(directory/'complete.json',dict(state='complete',report_sha256=sha(directory/'report.json'),prototype_sha256=sha(directory/'prototype.json'),checkpoint_sha256=sha(directory/'head.pt')))
            reports[name]=report
        for p,h in pins.items():assert sha(Path(p))==h,p
        eligible=sorted((n for n,r in reports.items() if r['gate']),key=lambda n:(-reports[n]['gate_deltas']['useful_vs_constant'],reports[n]['statistics']['development']['harmful_mass'],n))
        snapshots=a.out/'sources';snapshots.mkdir()
        for path in (Path(__file__),Path(__file__).with_name('combat_joint_aim_policy.py'),Path(__file__).with_name('combat_conditional_fire.py'),Path(__file__).with_name('combat_action_effect.py')):shutil.copy2(path,snapshots/path.name)
        save(a.out/'report.json',dict(state='complete',version='combat_shared_candidate_comparison_v1',device='cuda',variants=reports,
            eligible=eligible,selected_for_runtime=eligible[0] if eligible else None,inputs=pins,script_sha256=sha(__file__),training_performed=True,promotion=False,
            native_games_run=0,scope='Fixed500 prototypes on targeted TRAIN development split. No playable new PPO model or native quality claim. Source tensors/files unchanged.'))
        save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
        save(a.out/'progress.json',dict(stage='complete',eligible=eligible))
        print(json.dumps(dict(eligible=eligible,variants={n:{k:r[k] for k in ('parameters','initial_loss','final_loss','statistics','constant','gate','gate_deltas')} for n,r in reports.items()}),indent=2))
    except BaseException as e:
        save(a.out/'progress.json',dict(stage='failed',error=str(e),exception=type(e).__name__));raise


if __name__=='__main__':main()

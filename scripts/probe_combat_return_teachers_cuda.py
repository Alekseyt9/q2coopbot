"""Matched immediate-hit versus finite observed-return teachers, CUDA only."""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
from evaluate_registered_combat_models import read,save,sha
from ppo_combat import torch,nn,training_devices
from ppo_recurrent import durable_write
from combat_joint_aim_policy import JointAim
from combat_aim_candidates import apply_offsets,OFFSETS
from probe_combat_shared_candidate_scorers_cuda import Shared
from fork_conditional_weapon_exploration_cuda import equal_tree


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('source','dataset','returns','protocol','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    seal=read(pin(a.source/'complete.json'))
    for n,k in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):pin(a.source/n,seal[k])
    model=read(a.source/'weights.json');ds=read(pin(a.dataset/'complete.json'))
    meta=read(pin(a.dataset/'report.json',ds['report_sha256']));assert meta['fit_admission_passed']
    policy=meta['version']=='combat_native_aim_pair_dataset_v4_policy'
    rows=[r for r in map(json.loads,pin(a.dataset/'contexts.jsonl',ds['labels_sha256']).open(encoding='utf-8-sig')) if r.get('cohort')==('policy-v2' if policy else 'late-v1')]
    rs=read(pin(a.returns/'complete.json'));returns=read(pin(a.returns/'report.json',rs['report_sha256']))
    assert returns['version']=='combat_one_frame_observed_returns_v1' and not returns['excluded']
    # Revalidate the sealed return audit's native inputs, not only its summary.
    for p,h in returns['inputs'].items():pin(p,h)
    pin(a.protocol);bykey={(r['seed'],r['frame']):r for r in returns['contexts']}
    assert len(rows)==len(bykey)
    if policy:
        assert len(rows)==meta['contexts'] and meta['source_weights_sha256']==seal['weights_sha256']
        assert meta['fit_admissions']['immediate'] and ds['returns_report_sha256']==rs['report_sha256']
    else:assert len(rows)==26
    for r in rows:
        rr=bykey[r['seed'],r['frame']];assert rr['split']==r['split']
        assert {v['category']:v['health_damage'] for v in r['alternatives']}=={v['category']:v['immediate_hp'] for v in rr['samples']}
    train=[i for i,r in enumerate(rows) if r['split']=='train'];dev=[i for i,r in enumerate(rows) if r['split']=='development']
    assert not {rows[i]['seed'] for i in train}&{rows[i]['seed'] for i in dev}
    if policy:assert len(train)>=16 and len(dev)>=6 and len({r['seed'] for r in rows})==len(rows)
    else:assert len(train)==14 and len(dev)==12
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):source=JointAim(model).requires_grad_(False)
    before=copy.deepcopy(source.state_dict());count=len(rows)
    x=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float32)
    cat=lambda k:torch.tensor([r['sample'].get(k,0) for r in rows],device='cuda',dtype=torch.long)
    z=torch.tensor([r['sample']['aim_candidate']['source_latent'] for r in rows],device='cuda',dtype=torch.float32)
    timing=torch.tensor([r['sample']['aim_candidate']['timing'] for r in rows],device='cuda',dtype=torch.float32)
    target,mode,vertical,weapon=(cat(k) for k in ('target','aim_mode','vertical','weapon'))
    captured=[];hook=source.head.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach()))
    with torch.inference_mode():old_logits,active=source(x,z,target,mode,vertical,weapon,timing)
    hook.remove();assert bool(active.all());flat=captured[0].clone();hints=flat[:,-27:-2].clone();old=old_logits.softmax(-1).clone()
    c=source.action_context(x,z,target,mode,vertical,weapon);expanded=c[:,None,:].expand(-1,25,-1).clone()
    ids=torch.arange(25,device='cuda').expand(count,-1);modes=mode[:,None].expand(-1,25).reshape(-1)
    final=apply_offsets(z[:,None,:].expand(-1,25,-1).reshape(-1,4).double(),modes,ids.reshape(-1))
    scale=torch.where(modes==1,15./180.,1.).double()
    expanded[:,:,164:166]=(final[:,2:].tanh()*scale[:,None]).reshape(count,25,2).float()
    predictor=model['aim_candidate_head']['predictor'];mean=torch.tensor(predictor['mean'][:185],device='cuda');std=torch.tensor(predictor['std'][:185],device='cuda')
    p=hints.sigmoid();offsets=torch.tensor(OFFSETS,device='cuda');deltas=torch.stack((offsets[ids//5],offsets[ids%5]),-1)/5
    shared=torch.cat((((expanded-mean)/std).clamp(-20,20),deltas,p[:,:,None],p[:,:1,None].expand(-1,25,-1),(p-p[:,:1])[:,:,None],(ids==0).float()[:,:,None]),-1).detach().clone()
    assert shared.shape==(count,25,191) and bool(torch.isfinite(shared).all())
    known=torch.zeros((count,25),device='cuda',dtype=torch.bool)
    hp=torch.zeros((count,25),device='cuda',dtype=torch.float64);sparse=hp.clone();full=hp.clone()
    for i,r in enumerate(rows):
        for v in bykey[r['seed'],r['frame']]['samples']:
            k=v['category'];known[i,k]=True;hp[i,k]=v['immediate_hp'];sparse[i,k]=v['sparse_delta'];full[i,k]=v['full_delta']
    assert bool(known[:,0].all()) and bool((sparse[:,0]==0).all()) and bool((full[:,0]==0).all())
    positive=known&(sparse>=1);negative=known&(sparse<=-1)
    hp_positive=known&(hp>hp[:,:1]);hp_negative=known&(hp<hp[:,:1])
    tr=torch.tensor(train,device='cuda',dtype=torch.long);dv=torch.tensor(dev,device='cuda',dtype=torch.long)
    def metrics(prob,idx):
        prob=prob.double()
        return dict(contexts=len(idx),known_mass=float((prob*known).sum(-1)[idx].mean()),
            sparse_useful_mass=float((prob*positive).sum(-1)[idx].mean()),sparse_harmful_mass=float((prob*negative).sum(-1)[idx].mean()),
            known_sparse_delta=float((prob*sparse*known).sum(-1)[idx].mean()),known_full_delta=float((prob*full*known).sum(-1)[idx].mean()),
            immediate_useful_mass=float((prob*hp_positive).sum(-1)[idx].mean()),immediate_harmful_mass=float((prob*hp_negative).sum(-1)[idx].mean()),
            baseline_probability=float(prob[idx,0].mean()),unmeasured_argmax_contexts=int((~known.gather(1,prob.argmax(-1)[:,None])[:,0])[idx].sum()))
    groups={'train':tr,'development':dv}
    if policy:
        for mapname in sorted({r['geometry'].split('-site-')[0] for r in rows}):
            ids_map=[i for i in dev if rows[i]['geometry'].split('-site-')[0]==mapname]
            if ids_map:groups['development_'+mapname]=torch.tensor(ids_map,device='cuda',dtype=torch.long)
    source_metrics={s:metrics(old,i) for s,i in groups.items()}
    beta=math.log(24*float(old[tr,0].mean())/(1-float(old[tr,0].mean())))
    def build():
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(20261023);return Shared(beta)
    a.out.mkdir(parents=True);reports={};initial=None
    fit_names=['immediate']
    if not policy or meta['fit_admissions']['sparse_return']:fit_names.append('sparse_return')
    for name in fit_names:
        module=build();optimizer=module.optimizer()
        if initial is None:initial=copy.deepcopy(module.state_dict())
        else:assert equal_tree(initial,module.state_dict())
        preference=torch.zeros((len(train),25),device='cuda');values=hp if name=='immediate' else sparse
        preferred=[]
        for ti,i in enumerate(train):
            best_value=values[i,known[i]].max()
            improve=bool(best_value>hp[i,0]) if name=='immediate' else bool(best_value>=1)
            if improve:
                best=known[i]&((values[i]-best_value).abs()<=1e-6);preference[ti,best]=1/int(best.sum())
            else:preference[ti,0]=1
            preferred.append(preference[ti].tolist())
        teacher=.1*old[tr]+.9*preference;assert bool((teacher>0).all())
        for step in range(500):
            logits=module(flat,shared,hints);loss=-(teacher*logits[tr].log_softmax(-1)).sum(-1).mean()
            assert bool(torch.isfinite(loss));optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=nn.utils.clip_grad_norm_(module.parameters(),.5);assert bool(torch.isfinite(norm));optimizer.step();module.project()
        with torch.no_grad():
            logits=module(flat,shared,hints);prob=logits.softmax(-1);q=prob[tr].double().mean(0);q=q/q.sum()
        assert bool(torch.isfinite(prob).all()) and bool((prob>0).all()) and equal_tree(before,source.state_dict())
        statistics={s:metrics(prob,i) for s,i in groups.items()};controls={s:metrics(q[None,:].expand_as(prob),i) for s,i in groups.items()}
        out=a.out/name;out.mkdir()
        durable_write(out/'head.pt',lambda f:torch.save(dict(module=module.state_dict(),optimizer=optimizer.state_dict(),steps=500),f))
        restored=build();restored.load_state_dict(torch.load(out/'head.pt',map_location='cuda',weights_only=True)['module'])
        with torch.no_grad():assert torch.equal(restored(flat,shared,hints),logits)
        save(out/'prototype.json',dict(kind='combat_native_candidate_score_prototype_v1',variant='shared_'+name,head=module.export(),
             native_runtime_compatible=False,source_weights_sha256=seal['weights_sha256'],dataset_sha256=ds['labels_sha256'],return_labels_sha256=rs['report_sha256']))
        save(out/'contexts.json',[dict(seed=r['seed'],frame=r['frame'],split=r['split'],probabilities=prob[i].tolist(),constant_probabilities=q.tolist(),
             source_probabilities=old[i].tolist(),known_actions=bykey[r['seed'],r['frame']]['samples']) for i,r in enumerate(rows)])
        report=dict(state='complete',variant='shared_'+name,device='cuda',steps=500,parameters=sum(p.numel() for p in module.parameters()),
                    statistics=statistics,constant=controls,source=source_metrics,final_loss=float(loss.detach()),train_preference=preferred,
                    native_runtime_compatible=False,training_performed=True,promotion=False,roundtrip_verified=True)
        reports[name]=report
    reference=source_metrics['development'];matched=reports['immediate']['statistics']['development']
    immediate_control=reports['immediate']['constant']['development']
    immediate_checks=dict(useful_source=matched['immediate_useful_mass']>=reference['immediate_useful_mass']+.05,
                          useful_constant=matched['immediate_useful_mass']>=immediate_control['immediate_useful_mass']+.02,
                          harmful=matched['immediate_harmful_mass']<=reference['immediate_harmful_mass']+.05)
    checks={};control=None
    if 'sparse_return' in reports:
        d=reports['sparse_return']['statistics']['development'];control=reports['sparse_return']['constant']['development']
        checks=dict(useful_source=d['sparse_useful_mass']>=reference['sparse_useful_mass']+.05,
                useful_constant=d['sparse_useful_mass']>=control['sparse_useful_mass']+.02,
                useful_matched=d['sparse_useful_mass']>=matched['sparse_useful_mass']+.02,
                harmful=d['sparse_harmful_mass']<=reference['sparse_harmful_mass']+.05,
                return_constant=d['known_sparse_delta']>=control['known_sparse_delta']+.10,
                return_matched=d['known_sparse_delta']>=matched['known_sparse_delta']+.10)
    sparse_gate=bool(checks) and all(checks.values());immediate_gate=policy and all(immediate_checks.values())
    selected='sparse_return' if sparse_gate else ('immediate' if immediate_gate else None)
    for name,r in reports.items():
        r.update(gate=sparse_gate if name=='sparse_return' else immediate_gate,gate_checks=checks if name=='sparse_return' else immediate_checks)
        out=a.out/name;save(out/'report.json',r);save(out/'complete.json',dict(state='complete',report_sha256=sha(out/'report.json'),prototype_sha256=sha(out/'prototype.json'),checkpoint_sha256=sha(out/'head.pt')))
    for p,h in pins.items():assert sha(p)==h,p
    snapshots=a.out/'sources';snapshots.mkdir()
    for name in (Path(__file__).name,'probe_combat_shared_candidate_scorers_cuda.py','combat_joint_aim_policy.py','combat_conditional_fire.py','combat_action_effect.py'):
        shutil.copy2(Path(__file__).with_name(name),snapshots/name)
    save(a.out/'report.json',dict(state='complete',version='combat_shared_return_teacher_comparison_v1',device='cuda',contexts=count,
         train_contexts=len(train),development_contexts=len(dev),variants=reports,checks=checks,immediate_checks=immediate_checks,gate_passed=selected is not None,selected_for_runtime=selected,inputs=pins,
         script_sha256=sha(__file__),training_performed=True,promotion=False,native_games=0,
         scope='Matched fixed500 prototypes on finite single-rollout TRAIN acquisition labels. Unknown actions lack return labels; no expected-Q or native quality claim.'))
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(json.dumps(dict(gate_passed=selected is not None,selected_for_runtime=selected,checks=checks,immediate_checks=immediate_checks,development={n:r['statistics']['development'] for n,r in reports.items()},constant=control,source=reference),indent=2))


if __name__=='__main__':main()

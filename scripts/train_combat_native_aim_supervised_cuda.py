"""Fixed-budget head-only supervised fit on sealed native counterfactual labels."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
from evaluate_registered_combat_models import read, save, sha
from ppo_combat import torch, nn, training_devices, layers
from ppo_recurrent import durable_write
from combat_joint_aim_policy import JointAim, validate
from fork_conditional_weapon_exploration_cuda import equal_tree
from train_combat_joint_aim_cuda import cuda_tree


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('source','baseline-model','dataset','protocol','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    source=a.source.resolve();seal=read(pin(source/'complete.json'))
    for n,k in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):pin(source/n,seal[k])
    model=read(source/'weights.json');validate(model)
    bs=read(pin(a.baseline_model/'complete.json'));baseline=read(pin(a.baseline_model/'weights.json',bs['weights_sha256']))
    before=copy.deepcopy(model);after=copy.deepcopy(baseline)
    before['aim_candidate_head']['layers']=None;after['aim_candidate_head']['layers']=None;assert before==after
    ds=read(pin(a.dataset/'complete.json'));data=read(pin(a.dataset/'report.json',ds['report_sha256']))
    assert data['version']=='combat_native_aim_pair_dataset_v2' and data['fit_admission_passed'] and not data['training_performed']
    assert data['source_weights_sha256']==bs['weights_sha256']
    rows=list(map(json.loads,pin(a.dataset/'contexts.jsonl',ds['labels_sha256']).open(encoding='utf-8-sig')))
    assert len(rows)==17 and len({(r['seed'],r['frame']) for r in rows})==17
    train=[i for i,r in enumerate(rows) if r['split']=='train'];dev=[i for i,r in enumerate(rows) if r['split']=='development']
    assert len(train)==12 and len(dev)==5 and not {rows[i]['seed'] for i in train}&{rows[i]['seed'] for i in dev}
    improvement=lambda r:any(v['health_damage']>r['baseline_health_damage'] for v in r['alternatives'])
    assert len({rows[i]['seed'] for i in train if improvement(rows[i])})>=2 and any(improvement(rows[i]) for i in dev)
    assert all(len(r['alternatives'])>=3 for r in rows);pin(a.protocol)
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cp=torch.load(source/'checkpoint.pt',map_location='cuda',weights_only=True);assert cp['weights_sha256']==seal['weights_sha256']
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):joint=JointAim(model).requires_grad_(False)
    assert equal_tree(cp['aim_candidate_head'],joint.head.state_dict())
    initial_head=copy.deepcopy(joint.head.state_dict());hints_before=copy.deepcopy(joint.hints.state_dict())
    x=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float32)
    cat=lambda n:torch.tensor([r['sample'].get(n,0) for r in rows],device='cuda',dtype=torch.long)
    z=torch.tensor([r['sample'].get('aim_candidate',{}).get('source_latent',r['sample']['latent']) for r in rows],device='cuda',dtype=torch.float32)
    timing=torch.tensor([r['sample']['aim_candidate']['timing'] for r in rows],device='cuda',dtype=torch.float32)
    args=(x,z,cat('target'),cat('aim_mode'),cat('vertical'),cat('weapon'),timing)
    captured=[];handle=joint.head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
    with torch.inference_mode():initial_logits,active=joint(*args)
    handle.remove();assert bool(active.all());inputs=captured[0].detach().clone();old_p=initial_logits.softmax(-1).detach().clone()
    known=torch.zeros((17,25),device='cuda',dtype=torch.bool);health=torch.zeros((17,25),device='cuda')
    for i,r in enumerate(rows):
        for v in r['alternatives']:known[i,v['category']]=True;health[i,v['category']]=v['health_damage']
    baseline_hp=health[:,0];positive=known&(health>baseline_hp[:,None]);harmful=known&(health<baseline_hp[:,None])
    train_indices=torch.tensor(train,device='cuda',dtype=torch.long);dev_indices=torch.tensor(dev,device='cuda',dtype=torch.long)
    preferences=torch.zeros((len(train),25),device='cuda')
    for ti,i in enumerate(train):
        if bool(positive[i].any()):
            best=known[i]&(health[i]==health[i,known[i]].max());preferences[ti,best]=1/int(best.sum())
        else:preferences[ti,0]=1
    targets=.1*old_p[train_indices]+.9*preferences;assert bool((targets>0).all())
    joint.head.requires_grad_(True);optimizer=torch.optim.Adam(joint.head.parameters(),lr=.0003,weight_decay=.0001)
    a.out.mkdir(parents=True);save(a.out/'progress.json',dict(stage='gpu-fit',steps=200,train_contexts=12,development_contexts=5))
    initial_loss=float(-(targets*initial_logits[train_indices].log_softmax(-1)).sum(-1).mean())
    losses=[];norms=[];started=time.perf_counter()
    for step in range(200):
        logits=joint.head(inputs[train_indices]);loss=-(targets*logits.log_softmax(-1)).sum(-1).mean()
        assert bool(torch.isfinite(loss));optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=nn.utils.clip_grad_norm_(joint.head.parameters(),.5);assert bool(torch.isfinite(norm));optimizer.step()
        losses.append(float(loss.detach()));norms.append(float(norm))
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    assert not equal_tree(initial_head,joint.head.state_dict()) and equal_tree(hints_before,joint.hints.state_dict())
    assert all(p.grad is None for p in joint.hints.parameters())
    with torch.inference_mode():
        final_logits=joint.head(inputs);final_p=final_logits.softmax(-1)
        mean_p=final_p[train_indices].double().mean(0);assert bool((mean_p>0).all())
        final_loss=float(-(targets*final_logits[train_indices].log_softmax(-1)).sum(-1).mean())
    updated=copy.deepcopy(model);updated['aim_candidate_head']['layers']=layers(joint.head.layers);validate(updated)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):decoded=JointAim(updated).requires_grad_(False)
    with torch.inference_mode():
        decoded_logits,decoded_active=decoded(*args)
        assert torch.equal(decoded_active,active) and torch.equal(decoded_logits,final_logits)
    constant=copy.deepcopy(updated)
    constant['aim_candidate_head']['layers'][-1]['weight']=[[0.]*64 for _ in range(25)]
    constant['aim_candidate_head']['layers'][-1]['bias']=mean_p.log().float().tolist();validate(constant)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):control=JointAim(constant).requires_grad_(False)
    with torch.inference_mode():control_logits,control_active=control(*args);control_p=control_logits.softmax(-1)
    assert torch.equal(control_active,active) and torch.equal(control_p,control_p[:1].expand_as(control_p))
    assert float((control_p[0].double()-mean_p).abs().max())<1e-6
    def metrics(p,indices):
        p=p.double();known_mass=(p*known).sum(-1);valid=p*known
        return dict(contexts=len(indices),known_action_probability_mean=float(known_mass[indices].mean()),
            improving_action_probability_mean=float((p*positive).sum(-1)[indices].mean()),
            harmful_action_probability_mean=float((p*harmful).sum(-1)[indices].mean()),
            conditional_known_health_damage_mean=float(((valid*health).sum(-1)/known_mass.clamp_min(1e-20))[indices].mean()),
            baseline_probability_mean=float(p[indices,0].mean()),
            unmeasured_argmax_contexts=int((~known.gather(1,p.argmax(-1)[:,None])[:,0])[indices].sum()))
    statistics={name:{split:metrics(p,indices) for split,indices in (('train',train),('development',dev))}
                for name,p in (('source',old_p),('trained',final_p),('constant',control_p))}
    sd,td,cd=(statistics[n]['development'] for n in ('source','trained','constant'))
    gains=dict(improving_vs_source=td['improving_action_probability_mean']-sd['improving_action_probability_mean'],
        improving_vs_constant=td['improving_action_probability_mean']-cd['improving_action_probability_mean'],
        harmful_vs_source=td['harmful_action_probability_mean']-sd['harmful_action_probability_mean'])
    passed=gains['improving_vs_source']>=.05 and gains['improving_vs_constant']>=.02 and gains['harmful_vs_source']<=.05
    for name,m,module,opt in (('trained',updated,joint,optimizer),('constant',constant,control,None)):
        restored=copy.deepcopy(m);restored['aim_candidate_head']['layers']=model['aim_candidate_head']['layers'];assert restored==model
        directory=a.out/name;directory.mkdir();save(directory/'weights.json',m)
        head_opt=torch.optim.Adam(module.head.parameters(),lr=.0003) if opt is None else opt
        fork=copy.deepcopy(cp);fork.update(weights_sha256=sha(directory/'weights.json'),aim_candidate_head=module.head.state_dict(),
            aim_candidate_optimizer=head_opt.state_dict(),actor_training={**cp['actor_training'],'scope':'combat_native_aim_supervised_v2_'+name},
            native_aim_supervised=dict(version='combat_native_aim_supervised_v2',source_weights_sha256=seal['weights_sha256'],
                dataset_sha256=ds['labels_sha256'],protocol_sha256=sha(a.protocol),head_steps=200 if opt else 0,head_optimizer_reset=True))
        durable_write(directory/'checkpoint.pt',lambda f:torch.save(fork,f));loaded=torch.load(directory/'checkpoint.pt',map_location='cuda',weights_only=True)
        for key in cp:
            if key not in ('weights_sha256','aim_candidate_head','aim_candidate_optimizer','actor_training'):assert equal_tree(cp[key],cuda_tree(loaded[key])),key
        assert equal_tree(module.head.state_dict(),loaded['aim_candidate_head'])
        save(directory/'report.json',dict(state='complete',version='combat_native_aim_supervised_v2',variant=name,device='cuda',
            training_performed=opt is not None,head_steps=200 if opt else 0,source=str(source),dataset=str(a.dataset.resolve()),
            source_weights_sha256=seal['weights_sha256'],all_non_head_neural_and_optimizer_rng_history_exact=True,
            ppo_counters_and_consumed_rollouts_unchanged=True,head_optimizer_reset=True,statistics=statistics[name],promotion=False))
        save(directory/'complete.json',dict(state='complete',weights_sha256=sha(directory/'weights.json'),checkpoint_sha256=sha(directory/'checkpoint.pt'),report_sha256=sha(directory/'report.json')))
    frozen=a.out/'trainer-sources';frozen.mkdir()
    for path in (Path(__file__),Path(__file__).with_name('combat_joint_aim_policy.py')):shutil.copy2(path,frozen/path.name)
    for p,h in pins.items():assert sha(Path(p))==h,p
    report=dict(state='complete',version='combat_native_aim_supervised_v2',device='cuda',training_performed=True,head_steps=200,
        head_parameters=sum(p.numel() for p in joint.head.parameters()),train_contexts=12,development_contexts=5,
        initial_loss=initial_loss,final_loss=final_loss,gradient_norm_max=max(norms),fit_seconds=seconds,statistics=statistics,
        gate=bool(passed),gate_deltas=gains,inputs=pins,script_sha256=sha(__file__),promotion=False,
        scope='Fixed endpoint, head-only supervised CUDA fit. Partial native action labels; unknown actions not labelled. Development evaluated after all200steps. Constant distribution calibrated on TRAIN only. No native quality verification or automatic promotion.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json'),
        variants={n:sha(a.out/n/'complete.json') for n in ('trained','constant')}))
    save(a.out/'progress.json',dict(stage='complete',gate=bool(passed)))
    print(json.dumps({k:report[k] for k in ('head_steps','initial_loss','final_loss','fit_seconds','statistics','gate','gate_deltas')},indent=2))


if __name__=='__main__':main()

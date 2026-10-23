"""TRAIN-only head decomposition: contextual choices versus marginal noise rate."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from evaluate_registered_combat_models import read, save, sha
from train_combat_bc import torch, training_devices
from combat_joint_aim_policy import JointAim, events
from combat_aim_candidates import VERSION, choose, frame_uniforms, log_probability


def stats(v):
    assert v.is_cuda and bool(torch.isfinite(v).all())
    q=torch.quantile(v,torch.tensor([.1,.5,.9],device='cuda',dtype=v.dtype))
    return dict(mean=float(v.mean()),std=float(v.std(unbiased=False)),min=float(v.min()),p10=float(q[0]),median=float(q[1]),p90=float(q[2]),max=float(v.max()))


def kl(p,q):
    return (p*(p.log()-q.log())).sum(-1)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('model','baseline','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--corpus',action='append',required=True,help='label=sealed TRAIN corpus')
    ap.add_argument('--calibration-label',required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def pin(p,expected=None):
        p=Path(p).resolve();digest=sha(p);assert expected is None or digest==expected,str(p);pins[str(p)]=digest;return p
    def model(root):
        root=Path(root).resolve();seal=read(pin(root/'complete.json'))
        m=read(pin(root/'weights.json',seal['weights_sha256']));r=read(pin(root/'report.json',seal['report_sha256']))
        assert r['device']=='cuda' and m['action_distribution']==VERSION
        return m,seal['weights_sha256']
    final_model,final_sha=model(a.model);baseline_model,baseline_sha=model(a.baseline)
    final=JointAim(final_model).requires_grad_(False);baseline=JointAim(baseline_model).requires_grad_(False)
    assert final_model['aim_candidate_head']['predictor']==baseline_model['aim_candidate_head']['predictor']
    corpora=dict(v.split('=',1) for v in a.corpus)
    assert len(corpora)==len(a.corpus) and a.calibration_label in corpora
    # Calibrate constants on the declared last TRAIN corpus, then freeze them
    # for other historical TRAIN contexts. No validation enters this script.
    order=[a.calibration_label]+[k for k in corpora if k!=a.calibration_label]
    a.out.mkdir(parents=True)
    save(a.out/'protocol.json',dict(version='combat_joint_aim_head_rate_audit_v1',state='prepared',
        model=str(a.model.resolve()),model_sha256=final_sha,baseline_sha256=baseline_sha,corpora=corpora,
        calibration_label=a.calibration_label,training_performed=False,new_games=False,device='cuda',
        controls=['initial_constant','same_baseline_probability_uniform_tail','same_marginal_probabilities'],
        scope='Fixed recorded observations/source draws, categorical head only. No actor/fire regeneration, fitting, counterfactual engine outcomes or quality claim.'))
    reports={};calibrated=None;initial_constant=None
    try:
        with torch.inference_mode():
            for label in order:
                save(a.out/'progress.json',dict(stage='cuda-analysis',label=label))
                corpus=Path(corpora[label]).resolve();closed=read(pin(corpus/'complete.json'))
                protocol=read(pin(corpus/'protocol.json',closed['protocol_sha256']))
                assert closed['state']=='complete' and protocol['train_only'] and len(protocol['assignments'])==1
                assignment=protocol['assignments'][0];assert assignment['split']=='train'
                plan=read(pin(assignment['plan'],assignment['plan_sha256']))
                assert all(t['split']=='train' for t in plan['tasks'])
                data=corpus/'processing-train/merged';meta=read(pin(data/'report.json'))
                assert meta['numerical_verification']=='cuda_verified_v1' and meta['action_distribution']==VERSION
                rows=list(map(json.loads,pin(data/'rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig')))
                assert len(rows)==meta['rows'] and {r['seed'] for r in rows}==set(assignment['seeds'])
                behavior_model,behavior_sha=model(protocol['source'])
                assert behavior_sha==meta['model_sha256']==plan['model_sha256']
                assert behavior_model['aim_candidate_head']['predictor']==final_model['aim_candidate_head']['predictor']
                behavior=JointAim(behavior_model).requires_grad_(False)
                x=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float32)
                cat=lambda key:torch.tensor([r['sample'].get(key,0) for r in rows],device='cuda',dtype=torch.long)
                target,mode,vertical,weapon=(cat(k) for k in ('target','aim_mode','vertical','weapon'))
                z,old_choice,timing,active=events(rows,x,target,mode,vertical,weapon)
                args=(x,z,target,mode,vertical,weapon,timing)
                old_logits,old_active=behavior(*args);logits,current_active=final(*args);base_logits,base_active=baseline(*args)
                assert torch.equal(active,old_active) and torch.equal(active,current_active) and torch.equal(active,base_active) and bool(active.any())
                uniforms=torch.tensor([frame_uniforms(r['sample']['sampling_seed'],r['frame'],1) for r in rows],device='cuda',dtype=torch.float64)
                pick=lambda l:choose(l,uniforms[:,0],uniforms[:,1],active)
                assert torch.equal(pick(old_logits.double()),old_choice),'Recorded behavior category differs'
                expected=torch.tensor([(r['sample'].get('aim_candidate') or {}).get('log_probability',0.) for r in rows],device='cuda',dtype=torch.float64)
                old_lp_error=float((log_probability(old_logits.double(),old_choice,active)-expected).abs().max())
                assert old_lp_error<.0001
                p=logits.double().softmax(-1);p0=p[active,0];qbar=p[active].mean(0)
                basep=base_logits.double().softmax(-1)
                assert float((basep-basep[0]).abs().max())<1e-12,'Initial negative control is not constant'
                if calibrated is None:
                    calibrated=qbar.clone();initial_constant=basep[0].clone()
                assert float((initial_constant-basep[0]).abs().max())<1e-12
                local_rate=torch.full((25,),float((1-qbar[0])/24),device='cuda',dtype=torch.float64);local_rate[0]=qbar[0]
                constant_rate=torch.full((25,),float((1-calibrated[0])/24),device='cuda',dtype=torch.float64);constant_rate[0]=calibrated[0]
                controls=dict(initial_constant=initial_constant,same_baseline_probability_uniform_tail=constant_rate,same_marginal_probabilities=calibrated)
                total=kl(p[active],initial_constant).mean();context=kl(p[active],qbar).mean();global_kl=kl(qbar,initial_constant)
                rate=kl(local_rate,initial_constant);tail=kl(qbar,local_rate)
                decomposition_error=max(float(abs(total-context-global_kl)),float(abs(global_kl-rate-tail)))
                assert decomposition_error<1e-9
                final_choice=pick(logits.double());choices={k:pick(q.log()[None,:].expand(len(rows),-1)) for k,q in controls.items()}
                comparison={}
                for name,q in controls.items():
                    c=choices[name];both=active&(c!=0)&(final_choice!=0)
                    comparison[name]=dict(mean_kl_final_to_control=float(kl(p[active],q).mean()),
                        counterfactual_nonzero_choices=int(((c!=0)&active).sum()),
                        differing_categories=int(((c!=final_choice)&active).sum()),
                        differing_gate_decisions=int((((c!=0)!=(final_choice!=0))&active).sum()),
                        different_tail_given_both_nonzero=int(((c!=final_choice)&both).sum()))
                per_case=[]
                for seed in sorted({r['seed'] for r in rows}):
                    mask=torch.tensor([r['seed']==seed for r in rows],device='cuda',dtype=torch.bool)&active
                    per_case.append(dict(seed=seed,eligible=int(mask.sum()),recorded_behavior_nonzero=int(((old_choice!=0)&mask).sum()),
                        final_head_counterfactual_nonzero=int(((final_choice!=0)&mask).sum()),
                        rate_control_different_categories=int(((final_choice!=choices['same_baseline_probability_uniform_tail'])&mask).sum()),
                        marginal_control_different_categories=int(((final_choice!=choices['same_marginal_probabilities'])&mask).sum())))
                shares=dict(context=float(context/total),global_rate=float(rate/total),global_tail_preferences=float(tail/total)) if float(total)>1e-12 else None
                reports[label]=dict(rows=len(rows),eligible=int(active.sum()),baseline_probability=stats(p0),
                    mean_tail_probabilities=(qbar[1:]/qbar[1:].sum()).tolist(),
                    kl_decomposition=dict(total_final_to_initial=float(total),context=float(context),global_drift=float(global_kl),
                        global_rate=float(rate),global_tail_preferences=float(tail),fractions=shares,max_identity_error=decomposition_error),
                    recorded_behavior_nonzero=int(((old_choice!=0)&active).sum()),
                    final_head_counterfactual_nonzero=int(((final_choice!=0)&active).sum()),controls=comparison,cases=per_case,
                    behavior_category_lp_max_error=old_lp_error,behavior_categories_verified=True)
                print(label,json.dumps({k:v for k,v in reports[label].items() if k not in ('cases','mean_tail_probabilities')}),flush=True)
                del behavior,rows,x,z,p,old_logits,logits,base_logits
        for path,digest in pins.items():assert sha(Path(path))==digest,'Input changed: '+path
        frozen=a.out/'sources';frozen.mkdir();helpers={};script_root=Path(__file__).resolve().parent
        files={Path(__file__).resolve()}
        for m in list(sys.modules.values()):
            name=getattr(m,'__file__',None)
            if name and Path(name).resolve().parent==script_root and Path(name).suffix=='.py':files.add(Path(name).resolve())
        for p in sorted(files):shutil.copy2(p,frozen/p.name);helpers[p.name]=sha(frozen/p.name)
        report=dict(state='complete',version='combat_joint_aim_head_rate_audit_v1',device='cuda',
            model_sha256=final_sha,calibration_label=a.calibration_label,calibrated_marginal_probabilities=calibrated.tolist(),
            datasets=reports,inputs=pins,helpers=helpers,protocol_sha256=sha(a.out/'protocol.json'),
            training_performed=False,new_games=False,validation_used=False,promotion=False,quality_verified=False,
            scope='Descriptive categorical distributions on fixed TRAIN histories/source draws. Context KL measures dependence, not useful information or causal improvement. Hash category comparisons are counterfactual labels only; actor/fire/physics were not rerun. Earlier corpora are TRAIN history, not held-out validation.')
        save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
        save(a.out/'progress.json',dict(stage='complete',datasets=list(reports)))
    except BaseException as e:
        save(a.out/'progress.json',dict(stage='failed',exception=type(e).__name__,error=str(e)));raise


if __name__=='__main__':main()

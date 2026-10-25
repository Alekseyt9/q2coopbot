"""Verify current-v12 continuation/window contracts on real TRAIN histories using CUDA."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from ppo_combat import torch,training_devices
from ppo_recurrent import prepare_context,sequence
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import JointAim,base_model,events,probabilities
from combat_precision_head import selected_means
from combat_movement_probe import WINDOW_VERSION,shifts,validate as validate_movement
from combat_continuation_probe import VERSION,validate as validate_continuation,effective_seed


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    selected=read(pin(a.selection));assert selected['version']=='combat_fresh_contact_loss_train_roots_v2_right_positive'
    assert len(selected['cases'])==3 and {x['map'] for x in selected['cases']}=={'base3','jail1','ware2'}
    for p,h in selected['inputs'].items():pin(p,h)
    for p in Path(__file__).parent.glob('*.py'):pin(p)
    model=read(pin(selected['source_model'],selected['source_weights_sha256']))
    assert model['feature_version']=='combat_features_v12_projectile_types_neutral'
    assert not any(k.startswith('diagnostic_') for k in model)
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    actor=branch(base_model(model,allow_shared=True),'actor').requires_grad_(False)
    joint=JointAim(model,allow_shared=True).requires_grad_(False)
    actor_state=copy.deepcopy(actor.state_dict());joint_state=copy.deepcopy(joint.state_dict())
    equal=lambda aa,bb:aa.keys()==bb.keys() and all(torch.equal(aa[k],bb[k]) and bb[k].is_cuda for k in aa)
    a.out.mkdir(parents=True);variants=[];checks=[];used_seeds=set()
    for ci,case in enumerate(selected['cases']):
        directory=Path(case['source_case'])/'cuda';meta=read(pin(directory/'report.json'))
        rows=[json.loads(s) for s in pin(directory/'rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig')]
        history=[json.loads(s) for s in pin(directory/'sequence.jsonl',meta['sequence_sha256']).open(encoding='utf-8-sig')]
        root=next(r for r in rows if r['frame']==case['frame']);assert hashlib.sha256(json.dumps(root['features'],separators=(',',':')).encode()).hexdigest()==case['features_sha256']
        context=prepare_context(history,rows,'cuda');frame=case['frame']
        frames=torch.tensor([r['frame'] for r in rows],device='cuda',dtype=torch.long)
        x=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float32)
        cat=lambda k:torch.tensor([r['sample'].get(k,0) for r in rows],device='cuda',dtype=torch.long)
        target,mode,vertical,weapon=(cat(k) for k in ('target','aim_mode','vertical','weapon'))
        final=torch.tensor([r['sample']['latent'] for r in rows],device='cuda');attack=torch.tensor([float(r['sample']['attack']) for r in rows],device='cuda')
        std=torch.tensor(model['log_std'],device='cuda');original,choice,timing,_=events(rows,x,target,mode,vertical,weapon)
        with torch.inference_mode():
            raw,memory_error=sequence(actor,context,32,True,'actor');assert memory_error<2e-5
            means=selected_means(raw,target,mode,x,original[:,:2],actor.sampled_motion_aim_branch,vertical,actor.motor_branch)
            args=(std,original,final,attack,vertical,weapon,x,target,mode,choice,timing,actor.sampled_motion_aim_branch,actor.motor_branch,actor.conditional_fire_branch)
            base=probabilities(joint,raw,*args);lp_error=float((base[0]-torch.tensor([r['sample']['log_probability'] for r in rows],device='cuda')).abs().max());assert lp_error<.003,lp_error
        for replicate in range(2):
            seed=int.from_bytes(hashlib.sha256(f'contact-loss-policy-future-v1:{case["seed"]}:{frame}:{replicate}'.encode()).digest()[:4],'little')&0x7fffffff
            assert seed not in used_seeds;used_seeds.add(seed)
            for label,delta in [('control',None),('zero',[0.,0.]),('toward',case['candidate_mean_shift_toward']),('away',case['candidate_mean_shift_away'])]:
                changed=copy.deepcopy(model);changed['diagnostic_continuation_probe']=dict(version=VERSION,fork_frame=frame+1,seed=seed)
                if delta is not None:changed['diagnostic_movement_probe']=dict(version=WINDOW_VERSION,frame=frame,duration_frames=10,mean_shift=delta)
                validate_continuation(changed);validate_movement(changed)
                try:base_model(changed,allow_shared=True)
                except AssertionError:pass
                else:raise AssertionError('Diagnostic admitted to ordinary scoring')
                clean=base_model(changed,allow_probe=True,allow_shared=True)
                assert clean==base_model(model,allow_shared=True)
                decoded=branch(clean,'actor').requires_grad_(False);decoded_joint=JointAim(changed,allow_probe=True,allow_shared=True).requires_grad_(False)
                assert equal(actor_state,decoded.state_dict()) and equal(joint_state,decoded_joint.state_dict())
                assert effective_seed(changed,dict(frame=frame,sample=dict(sampling_seed=case['seed'])))==case['seed']
                assert effective_seed(changed,dict(frame=frame+1,sample=dict(sampling_seed=case['seed'])))==seed
                with torch.inference_mode():
                    shift=shifts(changed,frames);scored=probabilities(decoded_joint,raw,*args,movement_shift=shift)
                    analytic=(shift*(original[:,:2]-means[:,:2])/std[:2].exp().square()-.5*shift.square()/std[:2].exp().square()).sum(-1)
                    error=float(((scored[0]-base[0])-analytic).abs().max());assert error<.001,error
                    outside=(frames<frame)|(frames>=frame+10);assert torch.equal(scored[0][outside],base[0][outside])
                    if label in ('control','zero'):assert torch.equal(scored[0],base[0])
                i=len(variants);path=a.out/f'model-{i:02d}.json';save(path,changed)
                variants.append(dict(index=i,context_index=ci,context=case,replicate=replicate,label=label,fork_seed=seed,
                    model=str(path.resolve()),model_sha256=sha(path),movement_shift=delta,gaussian_formula_error=error))
        checks.append(dict(map=case['map'],seed=case['seed'],root_frame=frame,rows=len(rows),actor_memory_error=memory_error,source_lp_error=lp_error))
    assert equal(actor_state,actor.state_dict()) and equal(joint_state,joint.state_dict())
    assert len(variants)==24 and all(sha(p)==h for p,h in pins.items())
    report=dict(version='combat_contact_loss_probe_preparation_v1',state='complete',device='cuda',variants=variants,checks=checks,inputs=pins,
        source_model=selected['source_model'],source_weights_sha256=selected['source_weights_sha256'],selection_sha256=sha(a.selection),
        neural_parameters_exact=True,zero_likelihood_exact=True,outside_window_exact=True,ordinary_scoring_rejected=True,
        native_budget=24,cuda_budget=24,pool=8,timescale=2,training_performed=False,promotion=False,
        scope='Numerical contract on recorded real source histories, not fresh forked trajectories. Forked/zero native identity and draw receipts still require live runs. Motion-conditioned aim/fire remain active; native engine RNG unchanged.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(variants=24,device='cuda',checks=checks))


if __name__=='__main__':main()

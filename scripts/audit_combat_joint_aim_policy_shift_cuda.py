"""Read-only fixed-history policy drift after the coupled PPO experiment."""
import argparse
import json
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from ppo_combat import torch,training_devices
from ppo_recurrent import prepare_context,sequence
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model,JointAim,events
from combat_precision_head import selected_means,mode_distribution,probabilities
from combat_conditional_fire import logits as fire_logits
from combat_target_head import distribution as target_distribution
from combat_weapon_head import distribution as weapon_distribution,feature_mask


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','initial','head_only','coupled','corpus','evaluation','out'):ap.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path);assert expected is None or digest==expected,str(path)
        pins[str(path)]=digest;return path
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    roots=dict(source=a.source,initial=a.initial,head_only=a.head_only,base_actor=a.coupled/'base-actor-only',joint=a.coupled/'joint')
    models={};actors={};stds={};heads={};hashes={}
    for label,root in roots.items():
        closed=read(pin(root/'complete.json'));model=read(pin(root/'weights.json',closed['weights_sha256']))
        models[label]=model;hashes[label]=closed['weights_sha256']
        actors[label]=branch(base_model(model) if model.get('action_distribution') else model,'actor').requires_grad_(False)
        stds[label]=torch.tensor(model['log_std'],device='cuda')
        if model.get('action_distribution'):heads[label]=JointAim(model).requires_grad_(False)
    closed=read(pin(a.evaluation/'complete.json'));assert closed['state']=='complete' and closed['native_cases']==48 and closed['cuda_cases']==40
    datasets=[('TRAIN_initial',a.corpus/'processing-train/merged','initial'),
        ('validation_source',a.evaluation/'processing-source/merged','source'),
        ('validation_initial',a.evaluation/'processing-initial/merged','initial'),
        ('validation_joint',a.evaluation/'processing-joint/merged','joint')]
    def stats(v):
        assert v.is_cuda and bool(torch.isfinite(v).all())
        q=torch.quantile(v,torch.tensor([.5,.9,.99],device='cuda'))
        return dict(mean=float(v.mean()),median=float(q[0]),p90=float(q[1]),p99=float(q[2]),max=float(v.max()))
    def means(actor,raw,target,mode,x,z,vertical):
        return selected_means(raw,target,mode,x,z[:,:2],actor.sampled_motion_aim_branch,vertical,actor.motor_branch)
    reports={}
    with torch.no_grad():
        for name,data,behavior in datasets:
            meta=read(pin(data/'report.json'));assert meta['model_sha256']==hashes[behavior] and meta['numerical_verification']=='cuda_verified_v1'
            for f,key in (('rollout.jsonl','rollout_sha256'),('sequence.jsonl','sequence_sha256')):pin(data/f,meta[key])
            rows=[json.loads(line) for line in (data/'rollout.jsonl').read_text().splitlines()]
            context=[json.loads(line) for line in (data/'sequence.jsonl').read_text().splitlines()]
            prepared=prepare_context(context,rows,'cuda');x=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float32)
            cat=lambda key:torch.tensor([r['sample'].get(key,0) for r in rows],device='cuda',dtype=torch.long)
            target,mode,vertical,weapon=(cat(k) for k in ('target','aim_mode','vertical','weapon'))
            final=torch.tensor([r['sample']['latent'] for r in rows],device='cuda',dtype=torch.float32)
            attack=torch.tensor([float(r['sample']['attack']) for r in rows],device='cuda')
            augmented=bool(meta.get('action_distribution'))
            if augmented:z,choice,timing,active=events(rows,x,target,mode,vertical,weapon)
            else:z=final;active=None
            reference_raw,ref_memory_error=sequence(actors['source'],prepared,32,behavior!='joint','actor')
            assert behavior=='joint' or ref_memory_error<2e-5
            ref_means=means(actors['source'],reference_raw,target,mode,x,z,vertical)
            fire=lambda actor,raw:torch.distributions.Bernoulli(logits=fire_logits(actor.conditional_fire_branch,x,raw,target,mode,final,vertical,weapon))
            old=dict(normal=torch.distributions.Normal(ref_means,stds['source'].exp()),
                target=target_distribution(reference_raw[:,:45],x),mode=mode_distribution(reference_raw,target),
                weapon=weapon_distribution(reference_raw[:,8:20],feature_mask(x[:,:845])),
                vertical=torch.distributions.Categorical(logits=reference_raw[:,5:8]),fire=fire(actors['source'],reference_raw))
            results={}
            if behavior!='joint':
                actor=actors['source']
                lp,_=probabilities(reference_raw,stds['source'],z,attack,vertical,weapon,x,target,mode,
                    actor.sampled_motion_aim_branch,actor.motor_branch,actor.conditional_fire_branch)
                # Source/initial fire output is structurally zero, so final aim
                # does not alter this baseline likelihood. Initial adds category.
                assert bool((actor.conditional_fire_branch[-1].weight==0).all()) and bool((actor.conditional_fire_branch[-1].bias==0).all())
                recorded_cat=torch.tensor([r['sample']['aim_candidate']['log_probability'] if r['sample'].get('aim_candidate') else 0. for r in rows],device='cuda')
                old_error=float((lp+recorded_cat-torch.tensor([r['sample']['log_probability'] for r in rows],device='cuda')).abs().max())
                assert old_error<.003
            else:old_error=None
            for label in ('head_only','base_actor','joint'):
                actor=actors[label];raw,_=sequence(actor,prepared,32,False,'actor');mu=means(actor,raw,target,mode,x,z,vertical)
                new=dict(normal=torch.distributions.Normal(mu,stds[label].exp()),target=target_distribution(raw[:,:45],x),
                    mode=mode_distribution(raw,target),weapon=weapon_distribution(raw[:,8:20],feature_mask(x[:,:845])),
                    vertical=torch.distributions.Categorical(logits=raw[:,5:8]),fire=fire(actor,raw))
                parts={key:torch.distributions.kl_divergence(old[key],new[key]) for key in old}
                parts['normal']=parts['normal'].sum(-1)
                total=sum(parts.values()).clamp_min(0)
                scale=torch.where(mode==1,15.,180.)
                drift=((mu[:,2:].tanh()-ref_means[:,2:].tanh())*scale[:,None]).square().sum(-1).sqrt()
                if label=='head_only':
                    assert torch.equal(raw,reference_raw) and float(total.max())<1e-6
                result=dict(selected_factor_kl=stats(total),component_mean_kl={key:float(v.mean()) for key,v in parts.items()},
                    mean_aim_angle_drift_degrees=stats(drift),fraction_kl_above_training_budget=float((total>.005).float().mean()))
                if augmented and bool(active.any()):
                    before,on=heads['initial'](x,z,target,mode,vertical,weapon,timing)
                    after,new_on=heads[label](x,z,target,mode,vertical,weapon,timing)
                    assert torch.equal(active,on) and torch.equal(on,new_on)
                    oldlog,newlog=before.log_softmax(-1),after.log_softmax(-1)
                    ck=(oldlog.exp()*(oldlog-newlog)).sum(-1).clamp_min(0)
                    result['candidate_conditional_kl']=stats(ck[active])
                    result['baseline_probability']=float(newlog[active,0].exp().mean())
                results[label]=result
            reports[name]=dict(rows=len(rows),behavior_model=behavior,reference_source_replay_error=old_error,variants=results)
            print(name,{label:round(result['selected_factor_kl']['mean'],6) for label,result in results.items()},flush=True)
    for path,digest in pins.items():assert sha(Path(path))==digest
    a.out.mkdir(parents=True)
    report=dict(state='complete',version='combat_joint_aim_fixed_history_shift_v1',device='cuda',training_performed=False,new_games=False,
        datasets=reports,inputs=pins,script_sha256=sha(__file__),promotion=False,
        scope='Read-only diagnosis. Gaussian/mode/fire are conditioned on recorded behavior movement/target/mode/pose/action, not exact full-policy KL integrated over every conditional action. Histories stay fixed; no counterfactual game outcomes or cause-of-loss claim. Validation data never fitted.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))


if __name__=='__main__':main()

"""CUDA contract audit, not training data or a combat quality experiment."""
import argparse
import copy
import io
import hashlib
import json
from pathlib import Path
from ppo_combat import torch,training_devices
from ppo_recurrent import prepare_context,sequence
from combat_full_joint_ppo import FullJointScope,equal_tree
from combat_joint_aim_policy import events,probabilities,base_model
from combat_wide_aim import events as wide_events,geometry,transform,terms,VERSION,SALT
from combat_aim_candidates import frame_uniforms
from process_combat_architecture_pool import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('source','migration','request','out'):ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists()
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    source=read(a.source/'weights.json');cp=torch.load(a.source/'checkpoint.pt',map_location='cuda',weights_only=True)
    old=FullJointScope(source,cp['config']);old.restore(cp,sha(a.source/'weights.json'))
    ds={n:read(a.migration/n/'weights.json') for n in ('disabled','enabled')}
    config=read(a.migration/'enabled/config.json')
    disabled=FullJointScope(ds['disabled'],config);enabled=FullJointScope(ds['enabled'],config)
    for m in ds.values():
        assert base_model(m,allow_shared=True,allow_wide=True)==base_model(source,allow_shared=True)
        try:base_model(m,allow_shared=True)
        except AssertionError:pass
        else:raise AssertionError('Legacy scorer admitted wide event')
    rows=[];history=[];pins={};wanted={1844000142,1865000142,1892000142}
    request=read(a.request);assert request['split']=='train'
    import json
    for c in request['cases']:
        if c['seed'] not in wanted:continue
        data=Path(c['data']);meta=read(data/'report.json');assert meta['model_sha256']==sha(a.source/'weights.json')
        for file,key in (('rollout.jsonl','rollout_sha256'),('sequence.jsonl','sequence_sha256'),('cuda-verification.json','cuda_verification_sha256')):
            assert sha(data/file)==meta[key];pins[str(data/file)]=meta[key]
        rows += [json.loads(l) for l in (data/'rollout.jsonl').open(encoding='utf-8-sig')]
        history += [json.loads(l) for l in (data/'sequence.jsonl').open(encoding='utf-8-sig')]
    assert {r['seed'] for r in rows}==wanted
    prepared=prepare_context(history,rows,'cuda')
    x=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float32)
    exact=x.new_tensor([r['features'] for r in rows],dtype=torch.float64)
    cat=lambda k:torch.tensor([r['sample'].get(k,0) for r in rows],device='cuda',dtype=torch.long)
    target,mode,vertical,weapon=(cat(k) for k in ('target','aim_mode','vertical','weapon'))
    final=torch.tensor([r['sample']['latent'] for r in rows],device='cuda',dtype=torch.float32)
    attack=torch.tensor([float(r['sample']['attack']) for r in rows],device='cuda')
    source_z,choice,timing,active=events(rows,x,target,mode,vertical,weapon)
    score_inputs=(exact,torch.tensor([(r['sample'].get('aim_candidate') or {}).get('source_latent',r['sample']['latent']) for r in rows],device='cuda',dtype=torch.float64),
        torch.tensor([(r['sample'].get('aim_candidate') or {}).get('timing',[0.,0.]) for r in rows],device='cuda',dtype=torch.float64))
    def score(s,z,data=None,attacks=None):
        return probabilities(s.joint,raw,s.std,source_z,z,attack if attacks is None else attacks,vertical,weapon,x,target,mode,choice,timing,
            old.actor.sampled_motion_aim_branch,old.actor.motor_branch,old.actor.conditional_fire_branch,score_inputs=score_inputs,wide_data=data)[:2]
    with torch.no_grad():
        raw,memory_error=sequence(old.actor,prepared,32,True,'actor')
        lp,ent=score(old,final)
        native_error=float((lp-torch.tensor([r['sample']['log_probability'] for r in rows],device='cuda')).abs().max())
        assert native_error<.003 and memory_error<2e-5
        dd=wide_events(ds['disabled'],rows,x,target,mode,vertical,weapon)
        dlp,dent=score(disabled,final,dd)
        assert torch.equal(lp,dlp) and torch.equal(ent,dent),'Disabled likelihood/entropy changed'
        anchors,mask=geometry(exact,target,mode)
        assert int(mask.sum())>10
        legacy=torch.tensor([r['sample']['latent'] for r in rows],device='cuda',dtype=torch.float64)
        enabled.joint.wide.double()
        logits,mask2=enabled.joint.wide(exact,legacy,target,mode,vertical,weapon);assert torch.equal(mask,mask2)
        uniforms=torch.tensor([frame_uniforms(r['sample']['sampling_seed']^SALT,r['frame'],1)[0] for r in rows],device='cuda',dtype=torch.float64)
        selected=(mask&(uniforms<logits.sigmoid())).long();changed=transform(legacy,anchors,selected)
        from combat_conditional_fire import logits as fire_logits
        from combat_wide_aim import verify_fire
        fl=fire_logits(old.actor.conditional_fire_branch,x,raw,target,mode,changed.float(),vertical,weapon)
        synthetic_attack=torch.where(mask,(fl.sigmoid()>.5).to(attack.dtype),attack)
        annotations_lp=torch.distributions.Bernoulli(logits=logits).log_prob(selected.double())
        derived=copy.deepcopy(rows)
        for i,r in enumerate(derived):
            r['sample']['latent']=changed[i].tolist()
            if bool(mask[i]):
                r['sample']['attack']=bool(synthetic_attack[i])
                r['sample']['wide_aim']=dict(version=VERSION,source_latent=legacy[i].tolist(),anchor_degrees=anchors[i].tolist(),
                    choice=int(selected[i]),uniform=float(uniforms[i]),fire_uniform=.5,logit=float(logits[i]),log_probability=float(annotations_lp[i]))
        data=wide_events(ds['enabled'],derived,x,target,mode,vertical,weapon)
        recovered,old_choice,_,_=events(derived,x,target,mode,vertical,weapon)
        assert torch.equal(recovered,source_z) and torch.equal(old_choice,choice)
        verify_fire(data,fl,synthetic_attack)
        newlp,newent=score(enabled,changed.float(),data,synthetic_attack)
        base_changed,_=score(old,changed.float(),attacks=synthetic_attack)
        gate_lp,_=terms(enabled.joint.wide,data,target,mode,vertical,weapon,verify=True)
        assert float((newlp-base_changed-gate_lp).abs().max())<1e-10
        assert torch.equal(changed[:,:2],legacy[:,:2]) and bool(torch.isfinite(newent).all())
        for mutate in ('missing','anchor','uniform','source'):
            bad=copy.deepcopy(derived);i=int(mask.nonzero()[0]);record=bad[i]['sample']['wide_aim']
            if mutate=='missing':bad[i]['sample'].pop('wide_aim')
            elif mutate=='anchor':record['anchor_degrees'][0]+=.01
            elif mutate=='uniform':record['uniform']+=.01
            else:record['source_latent'][0]+=.01
            try:wide_events(ds['enabled'],bad,x,target,mode,vertical,weapon)
            except AssertionError:pass
            else:raise AssertionError('Bad event accepted: '+mutate)
    # Synthetic numerical gradient/resume exercise ONLY. No synthetic rollout
    # or resulting toy weights are written or admitted as PPO experience.
    initial=torch.load(a.migration/'enabled/checkpoint.pt',map_location='cuda',weights_only=True)
    def step(s):
        s.actor_optimizer.zero_grad(set_to_none=True)
        ids=torch.randint(len(rows),(64,),device='cuda')
        l,am=s.joint.wide(data['features'][ids],data['source'][ids],target[ids],mode[ids],vertical[ids],weapon[ids])
        labels=(ids%2).to(l.dtype);loss=((l-labels).square()*am).mean()
        loss.backward();s.actor_optimizer.step();s.verify()
        return ids.detach().clone()
    first=FullJointScope(ds['enabled'],config);first.restore(initial,sha(a.migration/'enabled/weights.json'))
    draws=[step(first) for _ in range(4)]
    second=FullJointScope(ds['enabled'],config);second.restore(initial,sha(a.migration/'enabled/weights.json'))
    d2=[step(second) for _ in range(2)]
    mid_model=second.export();mid_hash=hashlib.sha256(json.dumps(mid_model,sort_keys=True,allow_nan=False).encode()).hexdigest()
    mid=second.snapshot(mid_hash,initial['consumed_rollouts'])
    stream=io.BytesIO();torch.save(mid,stream);stream.seek(0);mid=torch.load(stream,map_location='cuda',weights_only=True)
    resumed=FullJointScope(mid_model,config)
    resumed.restore(mid,mid_hash)
    d2 += [step(resumed) for _ in range(2)]
    assert all(torch.equal(a,b) for a,b in zip(draws,d2))
    assert equal_tree(first.joint.state_dict(),resumed.joint.state_dict())
    assert equal_tree(first.actor_optimizer.state_dict(),resumed.actor_optimizer.state_dict())
    assert equal_tree(first.actor.state_dict(),initial['actor']) and equal_tree(first.value.state_dict(),initial['value'])
    assert not equal_tree(first.joint.wide.state_dict(),FullJointScope(ds['enabled'],config).joint.wide.state_dict()),'Gate did not learn in gradient test'
    for p,h in pins.items():assert sha(p)==h
    a.out.mkdir(parents=True)
    save(a.out/'report.json',dict(state='passed',device='cuda',rows=len(rows),worlds=3,eligible_rows=int(mask.sum()),chosen_rows=int(selected.sum()),
        legacy_likelihood_max_error=native_error,disabled_likelihood_entropy_exact=True,original_gaussian_and25_event_preserved=True,
        final_fire_and_gate_likelihood_verified=True,receipt_rejections=4,synthetic_gate_gradient_nonzero=True,resume_2_plus2_exact=True,
        old_actor_value_weights_unchanged_in_gradient_test=True,native_training_performed=False,
        synthetic_numerical_gradient_test_only=True,promotion=False,inputs=pins,
        scope='Three frozen source9 TRAIN histories. Derived hypothetical wide events and toy gradient/resume exercise verify math only; no synthetic native labels, on-policy corpus, combat result or fitted policy is published.'))
    save(a.out/'complete.json',dict(state='complete',device='cuda',report_sha256=sha(a.out/'report.json'),training_performed=False,promotion=False))
    print(read(a.out/'report.json'))


if __name__=='__main__':main()

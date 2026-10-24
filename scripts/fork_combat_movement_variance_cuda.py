"""Preserve training state while forking the exact previously evaluated movement-variance policy."""
import argparse
import copy
import json
from pathlib import Path
import shutil
from evaluate_registered_combat_models import read,save,sha
from ppo_combat import torch,nn,training_devices
from ppo_recurrent import prepare_context,sequence,durable_write
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model,JointAim,events,probabilities
from combat_geometry_ppo import GeometryJoint
from combat_weapon_state_residual import optimizer as actor_optimizer
from fork_conditional_weapon_exploration_cuda import equal_tree
from train_combat_joint_aim_cuda import cuda_tree


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','data','protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--variant',choices=('noise-only',),default='noise-only')
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or expected==h,str(path)
        pins[str(path)]=h;return path
    protocol=read(pin(a.protocol));assert protocol['version']=='combat_movement_variance_checkpoint_v1'
    artifact=Path(protocol['policy_artifact']);artifact_seal=read(pin(artifact/'complete.json'))
    for filename,key in [('weights.json','weights_sha256'),('report.json','report_sha256')]:pin(artifact/filename,artifact_seal[key])
    evaluated=read(artifact/'weights.json');assert not evaluated.get('movement_inertia')
    cache=Path(protocol['source_cache']);cache_seal=read(pin(cache/'complete.json'))
    cache_report=read(pin(cache/'report.json',cache_seal['report_sha256']))
    selected=cache_report['chunks'][0];assert selected['split']=='train'
    assert (Path(selected['case']['folder'])/'cuda').resolve()==a.data.resolve()
    ss=read(pin(a.source/'complete.json'))
    for name,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:pin(a.source/name,ss[key])
    assert ss['weights_sha256']==protocol['source_weights_sha256']
    model=read(a.source/'weights.json');core=base_model(model,allow_shared=True)
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    cp=torch.load(a.source/'checkpoint.pt',map_location='cuda',weights_only=True)
    assert cp['weights_sha256']==ss['weights_sha256']
    actor,value=branch(core,'actor'),branch(core,'value');joint=GeometryJoint(model)
    assert equal_tree(actor.state_dict(),cp['actor']) and equal_tree(value.state_dict(),cp['value'])
    assert equal_tree(joint.geometry_state(),cp['geometry_aim_head']) and equal_tree(joint.head.state_dict(),cp['aim_candidate_head'])
    std=nn.Parameter(torch.tensor(model['log_std'],device='cuda'));assert torch.equal(std,cp['log_std'])
    rates=[g['lr'] for g in cp['actor_optimizer']['param_groups']];assert len(rates)==4
    ao=actor_optimizer(actor,std,*rates);ao.load_state_dict(copy.deepcopy(cp['actor_optimizer']))
    assert equal_tree(cuda_tree(ao.state_dict()),cp['actor_optimizer'])
    vo=torch.optim.Adam(value.parameters(),lr=cp['config']['value_lr']);vo.load_state_dict(copy.deepcopy(cp['value_optimizer']))
    ho=joint.geometry_optimizer();ho.load_state_dict(copy.deepcopy(cp['geometry_aim_optimizer']))
    assert equal_tree(cuda_tree(vo.state_dict()),cp['value_optimizer']) and equal_tree(cuda_tree(ho.state_dict()),cp['geometry_aim_optimizer'])
    scale=torch.tensor(protocol['variants'][a.variant],device='cuda');changed=scale!=1
    with torch.no_grad():std.copy_(torch.tensor(evaluated['log_std'],device='cuda'))
    assert bool((std>=-8).all()) and bool((std<=1).all())
    assert torch.equal(std[~changed],cp['log_std'][~changed])
    assert torch.allclose((std-cp['log_std']).exp(),scale,atol=1e-7,rtol=1e-7)
    moments=ao.state[std];assert moments
    for key,tensor in moments.items():
        if key!='step':assert tensor.shape==(4,);tensor[changed]=0
    calibrated_optimizer=cuda_tree(ao.state_dict());source_optimizer=cp['actor_optimizer']
    std_index=next(i for i,v in enumerate(ao.param_groups[0]['params']) if v is std)
    std_id=source_optimizer['param_groups'][0]['params'][std_index]
    for key,state in calibrated_optimizer['state'].items():
        if key!=std_id:assert equal_tree(state,source_optimizer['state'][key])
        else:
            assert torch.equal(state['step'],source_optimizer['state'][key]['step'])
            for name,tensor in state.items():
                if name!='step':assert torch.equal(tensor[~changed],source_optimizer['state'][key][name][~changed]) and not bool(torch.count_nonzero(tensor[changed]))
    new=copy.deepcopy(evaluated)
    old_dict=copy.deepcopy(model);new_dict=copy.deepcopy(new);old_dict.pop('log_std');new_dict.pop('log_std');assert old_dict==new_dict
    meta=read(pin(a.data/'report.json'));assert meta['model_sha256']==ss['weights_sha256'] and meta['observation_timing_contract']=='frozen_world_v1'
    rows=list(map(json.loads,pin(a.data/'rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig')))
    context=list(map(json.loads,pin(a.data/'sequence.jsonl',meta['sequence_sha256']).open(encoding='utf-8-sig')))
    assert meta['rollout_sha256']==selected['case']['source_rollout_sha256'] and meta['sequence_sha256']==selected['case']['source_sequence_sha256']
    prepared=prepare_context(context,rows,'cuda')
    other=branch(base_model(new,allow_shared=True),'actor');other_value=branch(base_model(new,allow_shared=True),'value');other_joint=JointAim(new,allow_shared=True)
    assert equal_tree(actor.state_dict(),other.state_dict()) and equal_tree(value.state_dict(),other_value.state_dict())
    assert equal_tree(JointAim(model,allow_shared=True).state_dict(),other_joint.state_dict())
    x=torch.tensor([r['features'] for r in rows],device='cuda');cat=lambda k:torch.tensor([r['sample'].get(k,0) for r in rows],device='cuda',dtype=torch.long)
    target,mode,vertical,weapon=(cat(k) for k in ('target','aim_mode','vertical','weapon'))
    z=torch.tensor([r['sample']['latent'] for r in rows],device='cuda');attack=torch.tensor([float(r['sample']['attack']) for r in rows],device='cuda')
    source_z,choice,timing,active=events(rows,x,target,mode,vertical,weapon)
    with torch.no_grad():
        raw,ea=sequence(actor,prepared,32,True,'actor');other_raw,eb=sequence(other,prepared,32,True,'actor')
        va,ev=sequence(value,prepared,32,True,'value');vb,ew=sequence(other_value,prepared,32,True,'value')
        assert torch.equal(raw,other_raw) and torch.equal(va,vb) and max(ea,eb,ev,ew)<2e-5
        args=(source_z,z,attack,vertical,weapon,x,target,mode,choice,timing,actor.sampled_motion_aim_branch,actor.motor_branch,actor.conditional_fire_branch)
        old_q=probabilities(joint,raw,cp['log_std'],*args);new_q=probabilities(other_joint,other_raw,std,*args)
        assert torch.equal(old_q[2],new_q[2]) and torch.equal(old_q[3],new_q[3])
        from combat_precision_head import selected_means
        means=selected_means(raw,target,mode,x,source_z[:,:2],actor.sampled_motion_aim_branch,vertical,actor.motor_branch)
        def gaussian(logstd):return (-.5*((source_z-means)/logstd.exp()).square()-logstd-.5*torch.log(torch.tensor(2*3.141592653589793,device='cuda'))).sum(-1)
        density_error=float(((new_q[0]-old_q[0])-(gaussian(std)-gaussian(cp['log_std']))).abs().max());assert density_error<.003
    a.out.mkdir(parents=True);shutil.copy2(artifact/'weights.json',a.out/'weights.json');assert sha(a.out/'weights.json')==artifact_seal['weights_sha256'];shutil.copy2(__file__,a.out/Path(__file__).name)
    # CPU deserialization/serialization only preserves usable RNG byte tensors; NN checks stay CUDA.
    fork=torch.load(a.source/'checkpoint.pt',map_location='cpu',weights_only=True)
    fork.update(weights_sha256=sha(a.out/'weights.json'),log_std=std.detach().clone(),actor_optimizer=ao.state_dict())
    migration=dict(version='combat_movement_variance_checkpoint_fork_v1',variant=a.variant,scale=scale.tolist(),source_weights_sha256=ss['weights_sha256'],training_performed=False,
        std_coordinate_moments_reset=changed.tolist(),source_history_and_retention_preserved=True,fresh_own_policy_data_required=True)
    fork['model_migrations']=copy.deepcopy(fork.get('model_migrations',[]))+[migration]
    assert fork['rng'].device.type=='cpu' and all(t.device.type=='cpu' for t in fork['cuda_rng'])
    durable_write(a.out/'checkpoint.pt',lambda stream:torch.save(fork,stream))
    restored=torch.load(a.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    for key in cp:
        if key not in ('weights_sha256','log_std','actor_optimizer','model_migrations'):assert equal_tree(cp[key],restored[key]),key
    assert torch.equal(restored['log_std'],std.detach()) and equal_tree(restored['actor_optimizer'],calibrated_optimizer)
    decoded_std=nn.Parameter(torch.tensor(read(a.out/'weights.json')['log_std'],device='cuda'))
    decoded_opt=actor_optimizer(other,decoded_std,*rates);decoded_opt.load_state_dict(copy.deepcopy(restored['actor_optimizer']))
    assert equal_tree(cuda_tree(decoded_opt.state_dict()),calibrated_optimizer)
    assert torch.equal(decoded_std,restored['log_std'])
    for path,h in pins.items():assert sha(path)==h,path
    report=dict(state='complete',version='combat_movement_variance_checkpoint_fork_v1',device='cuda',variant=a.variant,scale=scale.tolist(),
        before_log_std=model['log_std'],after_log_std=new['log_std'],neural_tensors_except_std_exact=True,raw_actor_value_exact=True,
        fixed_action_conditional_heads_exact=True,source_optimizer_clocks_preserved=True,only_changed_std_moments_reset=True,
        critic_geometry_optimizer_rng_history_config_retention_exact=True,checkpoint_roundtrip_verified=True,
        density_difference_max_error=density_error,technical_train_rows=len(rows),source_weights_sha256=ss['weights_sha256'],inputs=pins,
        evaluation_weights_bytes_exact=True,training_performed=False,persisted_optimizer_steps=0,promotion=False,
        scope='Explicit stochastic-policy std calibration, not training. Closed-loop conditional outputs may change with sampled movement. Fresh quality and own-policy data required; no automatic promotion.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',weights_sha256=sha(a.out/'weights.json'),checkpoint_sha256=sha(a.out/'checkpoint.pt'),report_sha256=sha(a.out/'report.json')))
    print(dict(variant=a.variant,rows=len(rows),density_error=density_error,checkpoint_verified=True))


if __name__=='__main__':main()

"""Learn a shot-prediction basis while preserving the complete combat policy."""
import argparse
import copy
import json
from pathlib import Path
from evaluate_registered_combat_models import read,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','dataset','data','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();assert not args.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path);assert expected is None or digest==expected
        pins[str(path)]=digest;return path
    seal=read(pin(args.source/'complete.json'))
    for name,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:pin(args.source/name,seal[key])
    dataset_seal=read(pin(args.dataset/'complete.json'));dataset_report=read(pin(args.dataset/'report.json',dataset_seal['report_sha256']))
    pin(args.dataset/'dataset.pt',dataset_report['dataset_sha256'])
    assert not dataset_seal['ppo_ready'] and not dataset_report['policy_training']
    model=read(args.source/'weights.json');assert model.get('conditional_fire')
    from ppo_combat import torch,nn,training_devices
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    from ppo_recurrent import durable_json,durable_write,prepare_context,sequence
    from migrate_combat_navigation_cuda import branch
    from fork_conditional_weapon_exploration_cuda import equal_tree
    from combat_conditional_fire import export
    from combat_precision_head import probabilities
    cp=torch.load(args.source/'checkpoint.pt',map_location='cuda',weights_only=True)
    before=branch(model,'actor');assert equal_tree(before.state_dict(),cp['actor'])
    assert len(cp['actor_optimizer']['param_groups'])==4
    assert all(not cp['actor_optimizer']['state'].get(k) for k in cp['actor_optimizer']['param_groups'][-1]['params']), 'Initializer requires an untrained fire group'
    assert bool((before.conditional_fire_branch[-1].weight==0).all()) and bool((before.conditional_fire_branch[-1].bias==0).all())
    data=torch.load(args.dataset/'dataset.pt',map_location='cuda',weights_only=True)
    assert data['version']=='combat_shot_prediction_data_v2'
    x,y=data['features'],data['labels'];assert x.shape==(len(y),181)
    mean=x.mean(0);std=x.std(0,unbiased=False);std=torch.where(std<1e-3,torch.ones_like(std),std)
    torch.manual_seed(20261022);predictor=nn.Linear(181,1,device='cuda')
    optimizer=torch.optim.Adam(predictor.parameters(),lr=.001,weight_decay=.0001)
    for _ in range(200):
        loss=nn.functional.binary_cross_entropy_with_logits(predictor((x-mean)/std)[:,0],y)
        assert bool(torch.isfinite(loss));optimizer.zero_grad();loss.backward();optimizer.step()
    module=copy.deepcopy(before.conditional_fire_branch)
    with torch.no_grad():
        weight=predictor.weight[0]/std;bias=predictor.bias[0]-(predictor.weight[0]*mean/std).sum()
        module[0].weight[0].copy_(weight);module[0].bias[0]=bias
        module[0].weight[1].copy_(-weight);module[0].bias[1]=-bias
        module[2].weight[:2].zero_();module[2].bias[:2].zero_()
        module[2].weight[0,0]=1;module[2].weight[1,1]=1
        module[2].weight[2:,:2].zero_()
        hidden=module[:4](x);decoded=hidden[:,0]-hidden[:,1]
        error=float((decoded-predictor((x-mean)/std)[:,0]).abs().max());assert error<1e-4
        assert bool((module[-1].weight==0).all()) and bool((module[-1].bias==0).all())
        assert bool((module(x)==0).all())
    updated=copy.deepcopy(model);export(updated,module);after=branch(updated,'actor')
    for key,value in before.state_dict().items():
        if not key.startswith('conditional_fire_branch.'):assert torch.equal(value,after.state_dict()[key]),key
    assert equal_tree(branch(updated,'value').state_dict(),cp['value'])
    meta=read(pin(args.data/'report.json'));assert meta['model_sha256']==seal['weights_sha256']
    pin(args.data/'rollout.jsonl',meta['rollout_sha256']);pin(args.data/'sequence.jsonl',meta['sequence_sha256'])
    rows=[json.loads(s) for s in (args.data/'rollout.jsonl').read_text().splitlines()]
    context=[json.loads(s) for s in (args.data/'sequence.jsonl').read_text().splitlines()]
    prepared=prepare_context(context,rows,'cuda');features=torch.tensor([r['features'] for r in rows]);latent=torch.tensor([r['sample']['latent'] for r in rows])
    cat=lambda key:torch.tensor([r['sample'].get(key,0) for r in rows],dtype=torch.long)
    target,mode,vertical,weapon=(cat(k) for k in ('target','aim_mode','vertical','weapon'))
    attack=torch.tensor([float(r['sample']['attack']) for r in rows]);scale=torch.tensor(model['log_std'])
    with torch.no_grad():
        old,old_memory=sequence(before,prepared,check_states=True);new,new_memory=sequence(after,prepared,check_states=True)
        assert torch.equal(old,new)
        def likelihood(actor,raw):
            return probabilities(raw,scale,latent,attack,vertical,weapon,features,target,mode,
                actor.sampled_motion_aim_branch,actor.motor_branch,actor.conditional_fire_branch)[0]
        old_lp,new_lp=likelihood(before,old),likelihood(after,new);assert torch.equal(old_lp,new_lp)
        native_error=float((new_lp-torch.tensor([r['sample']['log_probability'] for r in rows])).abs().max());assert native_error<.003
    receipt=dict(state='complete',version='combat_fire_prediction_basis_initialization_v1',device='cuda',predictive_shots=len(y),
        initialization='two_ReLU_signed_linear_score_units_in_trainable_hidden_basis',predictor_steps=200,
        decoder_max_error=error,zero_fire_output_structurally_verified=True,raw_and_likelihood_exact=True,history_rows=len(rows),
        native_log_probability_max_error=native_error,memory_max_error=max(old_memory,new_memory),
        old_actor_value_std_adam_rng_history_preserved=True,new_fire_optimizer_empty=True,
        inputs=pins,script_sha256=sha(__file__),policy_training=False,promotion=False,
        scope='All predictive labels are existing TRAIN only. Final linear refit on all9248shots after development comparisons. Hidden representation changes, final fire output remains identically zero; combat behavior unchanged. No predictor threshold or runtime veto.')
    args.out.mkdir(parents=True)
    durable_write(args.out/'predictor.pt',lambda f:torch.save(dict(version='combat_shot_prediction_checkpoint_v1',kind='linear181',model=predictor.state_dict(),mean=mean,std=std,steps=200,dataset_sha256=dataset_report['dataset_sha256']),f))
    receipt['predictor_sha256']=sha(args.out/'predictor.pt')
    durable_json(args.out/'weights.json',updated)
    fork=torch.load(args.source/'checkpoint.pt',map_location='cpu',weights_only=True)
    fork.update(actor=after.state_dict(),weights_sha256=sha(args.out/'weights.json'),sequence_retention=None)
    fork['model_migrations']=fork.get('model_migrations',[])+[receipt]
    durable_write(args.out/'checkpoint.pt',lambda f:torch.save(fork,f))
    reloaded=torch.load(args.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    for key in cp:
        if key not in ('actor','weights_sha256','sequence_retention','model_migrations'):assert equal_tree(cp[key],reloaded[key]),key
    assert equal_tree(after.state_dict(),reloaded['actor'])
    for path,digest in pins.items():assert sha(Path(path))==digest
    durable_json(args.out/'report.json',receipt)
    durable_json(args.out/'complete.json',dict(version='combat_update_complete_v1',weights_sha256=sha(args.out/'weights.json'),checkpoint_sha256=sha(args.out/'checkpoint.pt'),report_sha256=sha(args.out/'report.json')))
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('inputs','scope')}))


if __name__=='__main__':main()

"""Stage a sealed experimental checkpoint with weapon residual and separate Adam group.

Private candidate format deliberately cannot resume in the existing trainer.
No native contract, gameplay training or promotion is claimed by this tool.
"""
import argparse
import copy
import math
import pathlib
from ppo_recurrent import torch,nn,read,sha,durable_json,durable_write,prepare_context,sequence
from ppo_combat import network,layers
from migrate_combat_navigation_cuda import branch
from combat_weapon_state_residual import WeaponStateResidual,inputs,VERSION


def candidate_actor(model):
    # Historical private prototype owns its hook; the standard wrapper now
    # owns only the explicitly versioned native residual. Never apply twice.
    actor=branch({k:v for k,v in model.items() if k!='weapon_state_residual'},'actor')
    actor.weapon_state_branch=network(model['weapon_state_residual']['layers']).to('cuda')
    def residual(module,args,output):
        raw=output[0] if isinstance(output,tuple) else output
        correction=module.weapon_state_branch(inputs(args[0],raw));result=raw.clone()
        result[...,8:20]=result[...,8:20]+correction
        return (result,*output[1:]) if isinstance(output,tuple) else result
    actor.register_forward_hook(residual)
    return actor


def optimizer_for(actor,std,cp,weapon_lr):
    parameters=dict(actor.named_parameters())
    names=[name for name in parameters if not name.startswith('weapon_state_branch.')]
    optimizer=torch.optim.Adam([parameters[name] for name in names]+[std],lr=cp['config']['actor_lr'])
    optimizer.load_state_dict(copy.deepcopy(cp['actor_optimizer']))
    if not any(name.startswith('weapon_state_branch.') for name in parameters): raise ValueError('Missing residual')
    group={key:copy.deepcopy(value) for key,value in optimizer.param_groups[0].items() if key!='params'}
    group.update(params=list(actor.weapon_state_branch.parameters()),lr=weapon_lr,name='weapon_state_residual')
    if 'initial_lr' in group: group['initial_lr']=weapon_lr
    optimizer.add_param_group(group)
    for parameter in actor.weapon_state_branch.parameters():
        state=optimizer.state[parameter]
        state['step']=torch.tensor(0.,device='cuda' if group.get('capturable') else 'cpu')
        state['exp_avg']=torch.zeros_like(parameter);state['exp_avg_sq']=torch.zeros_like(parameter)
        if group.get('amsgrad'): state['max_exp_avg_sq']=torch.zeros_like(parameter)
    return optimizer


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=pathlib.Path,required=True)
    parser.add_argument('--data',type=pathlib.Path,required=True)
    parser.add_argument('--out',type=pathlib.Path,required=True)
    parser.add_argument('--weapon-lr',type=float,default=.0003)
    args=parser.parse_args();assert torch.cuda.is_available() and not args.out.exists()
    assert math.isfinite(args.weapon_lr) and 0<args.weapon_lr<=.01
    torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    seal=read(args.source/'complete.json')
    for name,field in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):
        assert sha(args.source/name)==seal[field]
    model=read(args.source/'weights.json');assert model['feature_version']=='combat_features_v10' and model.get('spatial_aim')
    source=torch.load(args.source/'checkpoint.pt',map_location='cuda',weights_only=True)
    assert source['weights_sha256']==seal['weights_sha256'] and source['version']=='combat_architecture_checkpoint_v1'
    original=branch(model,'actor')
    assert original.state_dict().keys()==source['actor'].keys()
    assert all(torch.equal(v,source['actor'][name]) for name,v in original.state_dict().items())
    torch.cuda.manual_seed(20261011);prototype=WeaponStateResidual()
    candidate=copy.deepcopy(model)
    candidate['weapon_state_residual']=dict(version=VERSION,layers=layers(prototype.branch))
    actor=candidate_actor(candidate);std=nn.Parameter(source['log_std'].clone())
    assert all(torch.equal(actor.state_dict()[name],v) for name,v in source['actor'].items())
    optimizer=optimizer_for(actor,std,source,args.weapon_lr)
    transferred=optimizer.state_dict();previous=source['actor_optimizer']
    assert transferred['param_groups'][0]==previous['param_groups'][0]
    old_ids=set(previous['state']);new_ids=set(transferred['state'])-old_ids
    assert len(new_ids)==6
    for identifier,state in previous['state'].items():
        assert state.keys()==transferred['state'][identifier].keys()
        assert all(torch.equal(value.to('cuda'),transferred['state'][identifier][key].to('cuda')) for key,value in state.items())
    for identifier in new_ids:
        assert all(not bool(value.any()) for value in transferred['state'][identifier].values())
    meta=read(args.data/'report.json')
    assert meta['model_sha256']==seal['weights_sha256'] and meta['numerical_verification']=='cuda_verified_v1'
    for filename in ('rollout','sequence'): assert sha(args.data/(filename+'.jsonl'))==meta[filename+'_sha256']
    with (args.data/'rollout.jsonl').open(encoding='utf-8-sig') as stream: rows=[__import__('json').loads(line) for line in stream]
    with (args.data/'sequence.jsonl').open(encoding='utf-8-sig') as stream: context=[__import__('json').loads(line) for line in stream]
    prepared=prepare_context(context,rows,'cuda')
    with torch.no_grad():
        before,before_error=sequence(original,prepared,check_states=True)
        after,after_error=sequence(actor,prepared,check_states=True)
        assert torch.equal(before,after) and max(before_error,after_error)<2e-5
    result=copy.deepcopy(source)
    result['version']='combat_weapon_state_candidate_checkpoint_v1'
    result['source_checkpoint_version']=source['version']
    result['actor']=copy.deepcopy(actor.state_dict());result['actor_optimizer']=copy.deepcopy(transferred)
    old_sequence=copy.deepcopy(result.get('sequence_retention'));result['sequence_retention']=None
    result['weapon_state_optimizer_fork']=dict(weapon_lr=args.weapon_lr,old_groups=1,new_groups=2,
        previous_sequence_retention=old_sequence,requires_fresh_own_training=True)
    args.out.mkdir(parents=True)
    durable_json(args.out/'candidate-weights.json',candidate)
    result['weights_sha256']=sha(args.out/'candidate-weights.json')
    durable_write(args.out/'candidate-checkpoint.pt',lambda stream:torch.save(result,stream))
    restored=torch.load(args.out/'candidate-checkpoint.pt',map_location='cuda',weights_only=True)
    module=candidate_actor(read(args.out/'candidate-weights.json'));module.load_state_dict(restored['actor'],strict=True)
    restored_std=nn.Parameter(restored['log_std'].clone())
    legacy=copy.deepcopy(restored);legacy['actor_optimizer']=copy.deepcopy(source['actor_optimizer'])
    disposable=optimizer_for(module,restored_std,legacy,args.weapon_lr)
    disposable.load_state_dict(copy.deepcopy(restored['actor_optimizer']))
    for field in ('config','consumed_rollouts','updates_completed','total_actor_steps','anchor_sha256','bank_sha256','actor_training'):
        assert restored[field]==source[field]
    assert torch.equal(restored['rng'],source['rng']) and torch.equal(restored['log_std'],source['log_std'])
    assert len(restored['cuda_rng'])==len(source['cuda_rng']) and all(torch.equal(a,b) for a,b in zip(restored['cuda_rng'],source['cuda_rng']))
    assert all(torch.equal(restored['value'][name],value) for name,value in source['value'].items())
    # API restoration check on disposable objects; zero gradients provide no
    # information from evaluation rows, and the persisted PT is never stepped.
    x=torch.tensor([r['features'] for r in rows[:16]],device='cuda',dtype=torch.float32)[:,None,:]
    output=module(x);raw=output[0] if isinstance(output,tuple) else output
    ((raw.sum()+restored_std.sum())*0).backward();disposable.step()
    assert all(bool(torch.isfinite(p).all()) for p in module.parameters())
    stepped=disposable.state_dict()
    for identifier in new_ids: assert float(stepped['state'][identifier]['step'])==1
    assert sha(args.source/'weights.json')==seal['weights_sha256'] and sha(args.source/'checkpoint.pt')==seal['checkpoint_sha256']
    receipt=dict(state='complete',device='cuda',source_weights_sha256=seal['weights_sha256'],source_checkpoint_sha256=seal['checkpoint_sha256'],
        weights_sha256=result['weights_sha256'],checkpoint_sha256=sha(args.out/'candidate-checkpoint.pt'),
        data_report_sha256=sha(args.data/'report.json'),real_history_rows=len(rows),zero_output_preservation_exact=True,
        state_max_error=max(before_error,after_error),old_optimizer_state_preserved=True,new_zero_state_tensors=len(new_ids),
        optimizer_groups=2,weapon_lr=args.weapon_lr,persisted_optimizer_steps=0,training_performed=False,
        disposable_cuda_adam_restore_verified=True,ready_for_native_training=False,
        previous_sequence_retention=old_sequence,script_sha256=sha(pathlib.Path(__file__)),
        scope='Private candidate format and filenames reject accidental use by existing Go/trainer. Old actor/value/moments/clocks/RNG/history retained, zero weapon residual added in a separate optimizer group. CUDA real-history equivalence + disposable zero-gradient API restore, not native support, fitting on evaluation data or gameplay quality.')
    durable_json(args.out/'report.json',receipt)
    durable_json(args.out/'complete.json',dict(state='complete',weights_sha256=receipt['weights_sha256'],checkpoint_sha256=receipt['checkpoint_sha256'],report_sha256=sha(args.out/'report.json'),ready_for_native_training=False))
    print({key:receipt[key] for key in ('state','device','real_history_rows','zero_output_preservation_exact','optimizer_groups','ready_for_native_training')})

if __name__=='__main__': main()

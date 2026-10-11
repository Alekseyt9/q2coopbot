"""Promote a sealed private zero-init candidate to the native/trainer contract.

This validates complete observed histories on CUDA, without fitting evaluation
rows. It preserves existing tensors, moments, clocks, RNG and lineage, and
records the separate residual optimizer group/objective fork explicitly.
"""
import argparse,copy,json,pathlib
from ppo_recurrent import torch,nn,read,sha,durable_json,durable_write,prepare_context,sequence
from migrate_combat_navigation_cuda import branch
from combat_weapon_state_residual import VERSION,CONTRACT_VERSION,validate,optimizer


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',required=True,type=pathlib.Path)
    parser.add_argument('--data',required=True,type=pathlib.Path)
    parser.add_argument('--out',required=True,type=pathlib.Path)
    args=parser.parse_args()
    assert torch.cuda.is_available() and not args.out.exists()
    torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    seal=read(args.candidate/'complete.json')
    assert seal['state']=='complete' and seal['ready_for_native_training'] is False
    for filename,field in (('candidate-weights.json','weights_sha256'),('candidate-checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):
        assert sha(args.candidate/filename)==seal[field]
    private_model=read(args.candidate/'candidate-weights.json')
    source=torch.load(args.candidate/'candidate-checkpoint.pt',map_location='cuda',weights_only=True)
    assert source['version']=='combat_weapon_state_candidate_checkpoint_v1' and source['weights_sha256']==seal['weights_sha256']
    assert private_model['weapon_state_residual']['version']==VERSION
    model=copy.deepcopy(private_model);model['weapon_state_residual']['version']=CONTRACT_VERSION;validate(model)
    actor=branch(model,'actor');value=branch(model,'value')
    assert actor.state_dict().keys()==source['actor'].keys()
    assert all(torch.equal(v,source['actor'][name]) for name,v in actor.state_dict().items())
    assert all(torch.equal(v,source['value'][name]) for name,v in value.state_dict().items())
    last=actor.weapon_state_branch[-1]
    assert not bool(last.weight.any()) and not bool(last.bias.any())
    std=nn.Parameter(source['log_std'].clone());weapon_lr=source['weapon_state_optimizer_fork']['weapon_lr']
    opt=optimizer(actor,std,source['config']['actor_lr'],weapon_lr)
    opt.load_state_dict(copy.deepcopy(source['actor_optimizer']))
    state=opt.state_dict()
    assert state['param_groups']==source['actor_optimizer']['param_groups']
    assert state['state'].keys()==source['actor_optimizer']['state'].keys()
    for identifier,fields in state['state'].items():
        assert all(torch.equal(v.to('cuda'),source['actor_optimizer']['state'][identifier][k].to('cuda')) for k,v in fields.items())
    meta=read(args.data/'report.json')
    assert meta['numerical_verification']=='cuda_verified_v1'
    assert meta['model_sha256']==read(args.candidate/'report.json')['source_weights_sha256']
    for name in ('rollout','sequence'):assert sha(args.data/(name+'.jsonl'))==meta[name+'_sha256']
    with (args.data/'rollout.jsonl').open(encoding='utf-8-sig') as stream:rows=[json.loads(line) for line in stream]
    with (args.data/'sequence.jsonl').open(encoding='utf-8-sig') as stream:context=[json.loads(line) for line in stream]
    prepared=prepare_context(context,rows,'cuda')
    baseline=branch({k:v for k,v in model.items() if k!='weapon_state_residual'},'actor')
    with torch.no_grad():
        before,before_error=sequence(baseline,prepared,check_states=True)
        after,after_error=sequence(actor,prepared,check_states=True)
        assert torch.equal(before,after) and max(before_error,after_error)<2e-5
    migrated=copy.deepcopy(source)
    migrated['version']='combat_architecture_checkpoint_v1'
    migrated['actor_training']={**source['actor_training'],'weapon_state_residual':dict(version=CONTRACT_VERSION,learning_rate=weapon_lr)}
    migration=dict(version='combat_weapon_state_contract_migration_v1',private_weights_sha256=seal['weights_sha256'],
        private_checkpoint_sha256=seal['checkpoint_sha256'],old_contract=VERSION,new_contract=CONTRACT_VERSION,
        optimizer_fork=copy.deepcopy(source['weapon_state_optimizer_fork']),
        previous_actor_training=copy.deepcopy(source['actor_training']),requires_fresh_own_policy_training=True)
    migrated['model_migrations']=[*source.get('model_migrations',[]),migration]
    args.out.mkdir(parents=True)
    durable_json(args.out/'weights.json',model);migrated['weights_sha256']=sha(args.out/'weights.json')
    durable_write(args.out/'checkpoint.pt',lambda stream:torch.save(migrated,stream))
    restored=torch.load(args.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    roundtrip=branch(read(args.out/'weights.json'),'actor');roundtrip.load_state_dict(restored['actor'],strict=True)
    reopt=optimizer(roundtrip,nn.Parameter(restored['log_std'].clone()),restored['config']['actor_lr'],weapon_lr)
    reopt.load_state_dict(copy.deepcopy(restored['actor_optimizer']))
    assert reopt.state_dict()['param_groups']==source['actor_optimizer']['param_groups']
    for field in ('config','consumed_rollouts','updates_completed','total_actor_steps','anchor_sha256','bank_sha256','sequence_retention','retention_weights'):
        assert restored.get(field)==source.get(field)
    for field in ('actor','value'):
        assert all(torch.equal(v,restored[field][name]) for name,v in source[field].items())
    assert torch.equal(restored['rng'],source['rng']) and all(torch.equal(a,b) for a,b in zip(restored['cuda_rng'],source['cuda_rng']))
    assert sha(args.candidate/'candidate-checkpoint.pt')==seal['checkpoint_sha256']
    report=dict(state='complete',device='cuda',weights_sha256=migrated['weights_sha256'],checkpoint_sha256=sha(args.out/'checkpoint.pt'),
        source_candidate_weights_sha256=seal['weights_sha256'],source_candidate_checkpoint_sha256=seal['checkpoint_sha256'],
        rows=len(rows),sequence_rows=len(context),zero_init_output_exact=True,state_max_error=max(before_error,after_error),
        optimizer_groups=len(state['param_groups']),weapon_lr=weapon_lr,optimizer_steps_performed=0,training_performed=False,
        model_contract=CONTRACT_VERSION,checkpoint_contract=migrated['version'],script_sha256=sha(pathlib.Path(__file__)),
        data_report_sha256=sha(args.data/'report.json'),requires_fresh_own_policy_training=True,
        scope='CUDA standard-wrapper/optimizer restore plus zero-init equivalence on complete real histories. No evaluation fitting or native battle/quality acceptance; standard trainer must collect fresh own-policy data and fork sequence retention.')
    durable_json(args.out/'report.json',report)
    durable_json(args.out/'complete.json',dict(state='complete',weights_sha256=report['weights_sha256'],checkpoint_sha256=report['checkpoint_sha256'],report_sha256=sha(args.out/'report.json')))
    print(json.dumps({key:report[key] for key in ('state','device','rows','zero_init_output_exact','optimizer_groups','optimizer_steps_performed')}))

if __name__=='__main__':main()

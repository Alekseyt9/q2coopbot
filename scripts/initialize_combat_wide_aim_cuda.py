"""Explicit CUDA migration: retain old Adam moments, add fresh learned gate."""
import argparse
import copy
import math
from pathlib import Path
from process_combat_architecture_pool import read,save,sha
from ppo_combat import torch,nn,layers,training_devices
from ppo_recurrent import durable_json,durable_write
from combat_full_joint_ppo import FullJointScope,WIDE_SCOPE,equal_tree
from combat_joint_aim_policy import base_model,validate
from combat_wide_aim import VERSION,DISTRIBUTION,SAMPLER,WIDTH


def migrate_optimizer(old,old_layout,scope):
    new=scope.actor_optimizer.state_dict();layout=scope.layout();old_names={};new_names={}
    assert len(old['param_groups'])==len(old_layout)==len(new['param_groups'])==len(layout)
    for oldg,oldl,newg,newl in zip(old['param_groups'],old_layout,new['param_groups'],layout):
        assert oldl['name']==newl['name'] and len(oldg['params'])==len(oldl['parameters']) and len(newg['params'])==len(newl['parameters'])
        old_names.update(zip(oldl['parameters'],oldg['params']));new_names.update(zip(newl['parameters'],newg['params']))
        for k,v in oldg.items():
            if k!='params':newg[k]=copy.deepcopy(v)
    assert old_names.keys()<=new_names.keys()
    added=set(new_names)-set(old_names);assert added and all(name.startswith('joint.wide.layers.') for name in added)
    for name,oldid in old_names.items():
        if oldid in old['state']:new['state'][new_names[name]]=copy.deepcopy(old['state'][oldid])
    scope.actor_optimizer.load_state_dict(new)
    restored=scope.actor_optimizer.state_dict()
    for name,oldid in old_names.items():
        if oldid in old['state']:assert equal_tree(old['state'][oldid],restored['state'][new_names[name]])
    assert all(new_names[name] not in restored['state'] for name in added)
    return dict(common_parameters=len(old_names),preserved_parameter_moments=len(old['state']),fresh_parameter_tensors=len(added))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--probability',type=float,default=.2)
    ap.add_argument('--actor-scope',choices=('all','wide_only'),default='all')
    ap.add_argument('--candidate-lr',type=float)
    ap.add_argument('--steps',type=int)
    a=ap.parse_args();assert not a.out.exists() and math.isfinite(a.probability) and 0<a.probability<1
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    source=a.source.resolve();seal=read(source/'complete.json');pins={}
    for file,key in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):
        assert sha(source/file)==seal[key];pins[str(source/file)]=seal[key]
    model=read(source/'weights.json');assert model.get('wide_aim') is None and not any(k.startswith('diagnostic_') for k in model)
    cp=torch.load(source/'checkpoint.pt',map_location='cuda',weights_only=True)
    assert cp['weights_sha256']==seal['weights_sha256']
    old_scope=FullJointScope(model,cp['config']);old_scope.restore(cp,seal['weights_sha256'])
    # Initialization is independent of saved training RNG, shared by both arms.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(20261026)
        gate=nn.Sequential(nn.Linear(WIDTH,32,device='cuda'),nn.ReLU(),nn.Linear(32,16,device='cuda'),nn.ReLU(),nn.Linear(16,1,device='cuda'))
        with torch.no_grad():gate[-1].weight.zero_();gate[-1].bias.fill_(math.log(a.probability/(1-a.probability)))
        head_layers=layers(gate)
    config=copy.deepcopy(cp['config']);config['version']=WIDE_SCOPE
    config['actor_scope']=a.actor_scope
    if a.candidate_lr is not None:
        assert math.isfinite(a.candidate_lr) and 0<a.candidate_lr<=.01
        config['candidate_lr']=a.candidate_lr
    if a.steps is not None:
        assert 1<=a.steps<=100
        config['actor_steps']=config['value_steps']=a.steps
    a.out.mkdir(parents=True);results=[]
    for name,enabled in (('disabled',False),('enabled',True)):
        out=a.out/name;out.mkdir();wide=copy.deepcopy(model)
        wide.update(action_distribution=DISTRIBUTION,wide_aim=dict(version=VERSION,sampler=SAMPLER,enabled=enabled,layers=head_layers))
        validate(wide,allow_shared=True,allow_wide=True)
        assert base_model(wide,allow_shared=True,allow_wide=True)==base_model(model,allow_shared=True)
        scope=FullJointScope(wide,config)
        for field in ('actor','value'):assert equal_tree(getattr(scope,field).state_dict(),cp[field])
        assert all(equal_tree(scope.joint.state_dict()[k],v) for k,v in cp['joint'].items())
        moments=migrate_optimizer(cp['actor_optimizer'],cp['optimizer_layout'],scope)
        rates={'actor_and_std':'actor_lr','memory_route_branch':'route_lr','fine_weapon_aim_branch':'fine_lr','aim_candidate_head':'candidate_lr'}
        for group in scope.actor_optimizer.param_groups:group['lr']=config[rates[group['name']]]
        scope.value_optimizer.load_state_dict(copy.deepcopy(cp['value_optimizer']))
        assert equal_tree(cp['value_optimizer'],scope.value_optimizer.state_dict())
        torch.set_rng_state(cp['rng'].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in cp['cuda_rng']])
        durable_json(out/'weights.json',wide);durable_json(out/'config.json',config)
        fork=scope.snapshot(sha(out/'weights.json'),cp['consumed_rollouts'])
        fork.update(consumed_seeds=list(cp['consumed_seeds']),updates_completed=cp['updates_completed'],
            parent_checkpoint_sha256=seal['checkpoint_sha256'],migration='wide_gate_named_adam_state_v1')
        durable_write(out/'checkpoint.pt',lambda f:torch.save(fork,f))
        loaded=torch.load(out/'checkpoint.pt',map_location='cuda',weights_only=True)
        reload_scope=FullJointScope(read(out/'weights.json'),config);reload_scope.restore(loaded,sha(out/'weights.json'))
        assert equal_tree(loaded['actor_optimizer'],reload_scope.actor_optimizer.state_dict())
        assert equal_tree(loaded['value_optimizer'],reload_scope.value_optimizer.state_dict())
        report=dict(state='passed',device='cuda',version='combat_wide_aim_migration_v1',enabled=enabled,
            initial_probability=a.probability,gate_parameters=sum(p.numel() for p in scope.joint.wide.parameters()),
            actor_scope=a.actor_scope,candidate_lr=config['candidate_lr'],
            migration=moments,legacy_actor_critic_and_joint_parameters_exact=True,legacy_actor_and_critic_moments_exact=True,
            checkpoint_reload_exact=True,training_rng_preserved=True,consumed_seeds=len(cp['consumed_seeds']),
            parent_weights_sha256=seal['weights_sha256'],parent_checkpoint_sha256=seal['checkpoint_sha256'],
            training_performed=False,promotion=False,runtime_compatibility_verified=False,inputs=pins)
        durable_json(out/'report.json',report)
        durable_json(out/'complete.json',dict(state='complete',weights_sha256=sha(out/'weights.json'),checkpoint_sha256=sha(out/'checkpoint.pt'),report_sha256=sha(out/'report.json')))
        results.append(report)
    for p,h in pins.items():assert sha(p)==h
    save(a.out/'complete.json',dict(state='complete',device='cuda',variants=['disabled','enabled'],parent_weights_sha256=seal['weights_sha256'],training_performed=False,promotion=False))
    print([dict(enabled=r['enabled'],gate_parameters=r['gate_parameters'],migration=r['migration']) for r in results])


if __name__=='__main__':main()

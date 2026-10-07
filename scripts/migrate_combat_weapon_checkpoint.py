"""Preserve Temporal Adam history while expanding V5 to V6/weapon actor20.

CPU tensor copies only, no forward/backward/optimizer steps. Active retention
requires its own observation migration; inactive anchor/bank pins stay intact.
"""
import argparse,copy,json,pathlib
from ppo_recurrent import torch,sha,durable_write,durable_json
from combat_attention import CausalAttention,TEMPORAL
from combat_weapon_head import HEAD_VERSION

def module(model,role):
    spec=model['attention']
    return CausalAttention(model[role],spec[role],spec['heads'],spec['window'])

def expanded(value,shape):
    if value.ndim!=len(shape) or any(a>b for a,b in zip(value.shape,shape)):
        raise ValueError('Migration cannot truncate or change parameter rank')
    result=value.new_zeros(shape)
    result[tuple(slice(0,n) for n in value.shape)]=value
    return result

def migrate(cp,old,new):
    if cp['architecture']!=TEMPORAL or old['feature_version']!='combat_features_v5' or new['feature_version']!='combat_features_v6' or new.get('weapon_head')!=HEAD_VERSION:
        raise ValueError('Requires V5 Temporal -> V6 masked weapon Temporal')
    if cp.get('retention_weights',[1.,1.])!=[0.,0.]:raise ValueError('Active retention requires V6 bank/anchor migration')
    if old['attention']['heads']!=new['attention']['heads'] or old['attention']['window']!=new['attention']['window']:
        raise ValueError('Attention architecture changed')
    if old['log_std']!=new['log_std']:raise ValueError('Standard deviations changed')
    result=copy.deepcopy(cp);changes=[]
    for role in ('actor','value'):
        before,after=module(old,role),module(new,role)
        old_state,new_state=before.state_dict(),after.state_dict()
        if old_state.keys()!=new_state.keys() or cp[role].keys()!=old_state.keys():raise ValueError('Parameter names changed')
        permitted={'encoder.0.weight'}
        if role=='actor':permitted|={'head.weight','head.bias','residual.weight','residual.bias'}
        for name,value in old_state.items():
            if not torch.equal(value,cp[role][name]):raise ValueError('Source weights/checkpoint mismatch: '+role+'/'+name)
            target=new_state[name]
            if value.shape==target.shape:
                if not torch.equal(value,target):raise ValueError('Existing parameter changed: '+role+'/'+name)
            else:
                if name not in permitted:raise ValueError('Unexpected expanded tensor')
                if not torch.equal(target[tuple(slice(0,n) for n in value.shape)],value):raise ValueError('Existing prefix changed')
                expected=(64,845) if name=='encoder.0.weight' else ((20,64) if value.ndim==2 else (20,))
                if tuple(target.shape)!=expected:raise ValueError('Unexpected target width')
                changes.append(dict(role=role,parameter=name,old_shape=list(value.shape),new_shape=list(target.shape)))
        result[role]=new_state
        old_names=[name for name,_ in before.named_parameters()]
        new_names=[name for name,_ in after.named_parameters()]
        if role=='actor':old_names+=['log_std'];new_names+=['log_std']
        optimizer=result[role+'_optimizer']
        if old_names!=new_names or len(optimizer['param_groups'])!=1:raise ValueError('Optimizer ordering/groups changed')
        ids=optimizer['param_groups'][0]['params']
        if len(ids)!=len(old_names) or set(ids)!=set(optimizer['state']):raise ValueError('Incomplete Adam state')
        for identifier,name in zip(ids,old_names):
            shape=new_state[name].shape if name!='log_std' else cp['log_std'].shape
            original_shape=old_state[name].shape if name!='log_std' else cp['log_std'].shape
            state=optimizer['state'][identifier]
            for key,value in state.items():
                if key=='step':continue
                if key not in ('exp_avg','exp_avg_sq','max_exp_avg_sq') or value.shape!=original_shape:raise ValueError('Unknown Adam tensor')
                state[key]=expanded(value,shape)
    if not torch.equal(cp['log_std'],torch.tensor(new['log_std'])):raise ValueError('Checkpoint log_std mismatch')
    return result,changes

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for key in ('source','model','checkpoint','anchor','bank','out'):ap.add_argument('--'+key,required=True,type=pathlib.Path)
    a=ap.parse_args();read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    cp=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if cp['version']!='combat_architecture_checkpoint_v1' or cp['weights_sha256']!=sha(a.source):raise ValueError('Source pin mismatch')
    if cp['anchor_sha256']!=sha(a.anchor) or cp['bank_sha256']!=sha(a.bank):raise ValueError('Inactive retention pins changed')
    result,changes=migrate(cp,read(a.source),read(a.model))
    result['weights_sha256']=sha(a.model)
    receipt=dict(version='combat_weapon_checkpoint_migration_v1',source_weights_sha256=sha(a.source),source_checkpoint_sha256=sha(a.checkpoint),weights_sha256=sha(a.model),expanded_parameters=changes,updates_completed=cp['updates_completed'],total_actor_steps=cp['total_actor_steps'],consumed_rollouts=len(cp['consumed_rollouts']),optimizer_steps=0,retention='Inactive anchor/bank pins and zero weights preserved; activation requires V6 migration')
    result['model_migrations']=cp.get('model_migrations',[])+[receipt]
    a.out.mkdir(parents=True,exist_ok=False)
    durable_write(a.out/'checkpoint.pt',lambda output:torch.save(result,output))
    receipt['checkpoint_sha256']=sha(a.out/'checkpoint.pt')
    durable_json(a.out/'migration.json',receipt)
    print(json.dumps(receipt,indent=2))

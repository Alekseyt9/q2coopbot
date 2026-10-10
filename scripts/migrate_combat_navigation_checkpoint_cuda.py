"""V7 -> V8 architecture checkpoint migration, preserving Adam and provenance.

All model comparisons and optimizer verification run on CUDA. No training.
Inactive retention pins remain unchanged; active retention needs a separate
versioned migration and is rejected rather than silently discarded.
"""
import argparse
import copy
import json
from pathlib import Path
from migrate_combat_navigation_cuda import branch, migrate, output, sha, torch
from ppo_recurrent import durable_json, durable_write


def expanded(tensor, extra=27):
    return torch.cat((tensor, tensor.new_zeros(tensor.shape[0],extra)),1)


def transfer(cp, old, new):
    widths = {('combat_features_v7','combat_features_v8'):(854,881),
              ('combat_features_v8','combat_features_v9'):(881,1121)}
    old_width,new_width = widths[(old['feature_version'],new['feature_version'])]
    extra = new_width-old_width
    if cp['version'] != 'combat_architecture_checkpoint_v1':
        raise ValueError('Unsupported checkpoint contract')
    key = 'memory' if old.get('memory') else 'attention'
    if cp['architecture'] != old[key]['version']:
        raise ValueError('Checkpoint architecture mismatch')
    if cp.get('retention_weights',[1.,1.]) != [0.,0.] or cp.get('sequence_retention'):
        raise ValueError('Active retention requires separate observation/objective migration')
    result = copy.deepcopy(cp)
    changes = []
    for role in ('actor','value'):
        before, after = branch(old,role), branch(new,role)
        bs, ns = before.state_dict(), after.state_dict()
        if bs.keys()!=ns.keys() or bs.keys()!=cp[role].keys():
            raise ValueError('Changed checkpoint parameter names')
        changed = []
        for name, value in bs.items():
            if not torch.equal(value,cp[role][name].to('cuda')):
                raise ValueError('Source checkpoint/weights mismatch: '+role+'/'+name)
            target = ns[name]
            if target.shape != value.shape:
                if value.ndim!=2 or value.shape[1]!=old_width or target.shape!=(value.shape[0],new_width):
                    raise ValueError('Unexpected shape change')
                if not torch.equal(target,expanded(value,extra)):
                    raise ValueError('Expansion changed existing weights')
                changed.append(name)
            elif not torch.equal(value,target):
                raise ValueError('Unchanged parameter modified')
        if len(changed)!=1:
            raise ValueError('Expected one first-layer expansion per module')
        result[role]=ns
        names = [n for n,_ in before.named_parameters()]
        if role=='actor': names.append('log_std')
        state = result[role+'_optimizer']
        if len(state['param_groups'])!=1:
            raise ValueError('Unsupported optimizer groups')
        ids = state['param_groups'][0]['params']
        if len(ids)!=len(names) or set(ids)!=set(state['state']):
            raise ValueError('Incomplete or ambiguous optimizer state')
        for identifier,name in zip(ids,names):
            shape = bs[name].shape if name!='log_std' else cp['log_std'].shape
            for field,tensor in state['state'][identifier].items():
                if field=='step':
                    continue
                if field not in ('exp_avg','exp_avg_sq','max_exp_avg_sq') or tensor.shape!=shape:
                    raise ValueError('Unknown Adam moment')
                if name in changed:
                    state['state'][identifier][field]=expanded(tensor,extra)
        changes.append(dict(role=role,parameter=changed[0],old_width=old_width,new_width=new_width))
    if not torch.equal(cp['log_std'].to('cuda'),torch.tensor(new['log_std'],device='cuda')):
        raise ValueError('Checkpoint standard deviations differ')
    return result, changes


def verify(cp, new):
    checks = {}
    for role in ('actor','value'):
        net = branch(new,role)
        net.load_state_dict(cp[role],strict=True)
        parameters = list(net.parameters())
        if role=='actor':
            parameters.append(torch.nn.Parameter(cp['log_std'].clone().to('cuda')))
        optimizer = torch.optim.Adam(parameters,lr=cp['config'][role+'_lr'])
        optimizer.load_state_dict(cp[role+'_optimizer'])
        loaded = optimizer.state_dict()
        assert loaded['param_groups']==cp[role+'_optimizer']['param_groups']
        for identifier, saved in cp[role+'_optimizer']['state'].items():
            for field,value in saved.items():
                actual = loaded['state'][identifier][field]
                if isinstance(value,torch.Tensor):
                    assert bool(torch.isfinite(actual).all()) and torch.equal(actual.to('cuda'),value.to('cuda'))
                else:
                    assert actual==value
        # Zero-gradient step on disposable restored state exercises actual Adam
        # with widened moment shapes. The persisted checkpoint is not stepped.
        x = torch.randn(2,8,len(new['actor'][0]['weight'][0]),device='cuda')
        net.zero_grad()
        (output(net,x).sum()*0).backward()
        optimizer.step()
        assert all(bool(torch.isfinite(p).all()) for p in parameters)
        checks[role]=dict(optimizer_restored=True,disposable_cuda_step=True)
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True,help='Closed source update directory')
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.set_default_device('cuda')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    seal=json.loads((args.source/'complete.json').read_text(encoding='utf-8-sig'))
    for filename,field in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
        if sha(args.source/filename)!=seal[field]: raise ValueError('Broken source seal: '+filename)
    old=json.loads((args.source/'weights.json').read_text(encoding='utf-8-sig'))
    cp=torch.load(args.source/'checkpoint.pt',map_location='cuda',weights_only=True)
    if cp['weights_sha256']!=sha(args.source/'weights.json'): raise ValueError('Source pin mismatch')
    new=migrate(old)
    result,changes=transfer(cp,old,new)
    with torch.no_grad():
        x=torch.randn(2,12,854,device='cuda')
        x[...,426:466:5]=1
        nx=torch.cat((x,torch.randn(2,12,27,device='cuda')),dim=-1)
        for role in ('actor','value'):
            torch.testing.assert_close(output(branch(old,role),x),output(branch(new,role),nx),rtol=1e-5,atol=1e-5)
    checks=verify(result,new)
    args.out.mkdir(parents=True,exist_ok=False)
    durable_json(args.out/'weights.json',new)
    result['weights_sha256']=sha(args.out/'weights.json')
    receipt=dict(version='combat_navigation_checkpoint_migration_v1',device='cuda',
                 gpu=torch.cuda.get_device_name(),source_weights_sha256=sha(args.source/'weights.json'),
                 source_checkpoint_sha256=sha(args.source/'checkpoint.pt'),
                 weights_sha256=result['weights_sha256'],expanded_parameters=changes,
                 optimizer_checks=checks,updates_completed=cp['updates_completed'],
                 total_actor_steps=cp['total_actor_steps'],consumed_rollouts=cp['consumed_rollouts'],
                 training_performed=False,persisted_optimizer_steps=0,
                 scope='Existing moments/steps, RNG and consumed rollouts preserved; new-column moments zero. Disposable CUDA Adam check only. No fresh PPO update or quality claim.')
    result['model_migrations']=cp.get('model_migrations',[])+[copy.deepcopy(receipt)]
    durable_write(args.out/'checkpoint.pt',lambda f:torch.save(result,f))
    restored=torch.load(args.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    for key in ('updates_completed','total_actor_steps','consumed_rollouts','config','anchor_sha256','bank_sha256'):
        assert restored[key]==cp[key]
    assert torch.equal(restored['rng'],cp['rng'])
    assert len(restored['cuda_rng'])==len(cp['cuda_rng'])
    assert all(torch.equal(a,b) for a,b in zip(restored['cuda_rng'],cp['cuda_rng']))
    verify(restored,new)
    for filename,field in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
        assert sha(args.source/filename)==seal[field], 'Source changed during migration'
    receipt['persisted_restoration_verified']=True
    receipt['checkpoint_sha256']=sha(args.out/'checkpoint.pt')
    durable_json(args.out/'migration.json',receipt)
    print(json.dumps(dict(state='passed',device='cuda',updates_completed=cp['updates_completed'],out=str(args.out)),indent=2))


if __name__=='__main__':main()

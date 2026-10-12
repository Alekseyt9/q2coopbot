"""Fork a sealed checkpoint's observation-only temperature without changing tensors.

Real histories verify the transform on CUDA; no evaluation data is fitted.
Saved Adam moments/clocks/RNG/history remain intact, with old retention archived.
"""
import argparse,copy,json,pathlib
from ppo_recurrent import torch,nn,read,durable_json,durable_write,prepare_context,sequence
from migrate_combat_navigation_cuda import branch
from process_combat_architecture_pool import sha
from combat_weapon_exploration import VERSION,validate,apply
from combat_weapon_head import feature_mask
from combat_weapon_state_residual import optimizer


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=pathlib.Path,required=True)
    ap.add_argument('--data',type=pathlib.Path,required=True)
    ap.add_argument('--temperature',type=float,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    args=ap.parse_args();assert torch.cuda.is_available() and not args.out.exists()
    torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    seal=read(args.source/'complete.json');pins={}
    for name,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
        p=args.source/name;assert sha(p)==seal[key];pins[p]=seal[key]
    model=read(args.source/'weights.json');assert not model.get('weapon_exploration')
    new=copy.deepcopy(model);new['weapon_exploration']=dict(version=VERSION,temperature=args.temperature);validate(new)
    checkpoint=torch.load(args.source/'checkpoint.pt',map_location='cuda',weights_only=True)
    assert checkpoint['weights_sha256']==seal['weights_sha256'] and not model['deterministic']
    meta=read(args.data/'report.json');assert meta['model_sha256']==seal['weights_sha256']
    for name in ['rollout','sequence']:
        p=args.data/(name+'.jsonl');assert sha(p)==meta[name+'_sha256'];pins[p]=meta[name+'_sha256']
    with (args.data/'rollout.jsonl').open(encoding='utf-8-sig') as f:rows=[json.loads(l) for l in f]
    with (args.data/'sequence.jsonl').open(encoding='utf-8-sig') as f:context=[json.loads(l) for l in f]
    prepared=prepare_context(context,rows,'cuda');before=branch(model,'actor');after=branch(new,'actor')
    for role in ['actor','value']:
        assert equal_tree(branch(model,role).state_dict(),checkpoint[role])
        assert equal_tree(branch(new,role).state_dict(),checkpoint[role])
    x=torch.tensor([r['features'] for r in rows],dtype=torch.float32,device='cuda')
    with torch.no_grad():
        a,ea=sequence(before,prepared,check_states=True);b,eb=sequence(after,prepared,check_states=True)
        expected=apply(new['weapon_exploration'],x,a)
        assert torch.equal(expected,b) and max(ea,eb)<2e-5
        gate=(x[:,1132]==1)&feature_mask(x[:,:845])[:,1:5].any(-1)
        assert bool(gate.any()) and bool((~gate).any())
        assert torch.equal(a[~gate],b[~gate]) and torch.equal(a[:,:8],b[:,:8]) and torch.equal(a[:,20:],b[:,20:])
        mask=feature_mask(x[:,:845]);pa=a[:,8:20].masked_fill(~mask,-torch.inf).softmax(-1);pb=b[:,8:20].masked_fill(~mask,-torch.inf).softmax(-1)
        assert torch.equal(pa.argmax(-1),pb.argmax(-1))
    # Analytic CUDA derivative of the gate itself, including ungated rows.
    raw=a[:8].detach().clone().requires_grad_();fx=x[:8].detach().clone()
    fx[:,1132]=0;fx[:4,1132]=1;fx[:,833:845]=0;fx[:,833:835]=1
    y=apply(new['weapon_exploration'],fx,raw);y.sum().backward()
    grad=torch.ones_like(raw);grad[:4,8:20]/=args.temperature
    assert torch.equal(raw.grad,grad)
    # Restore original group IDs and all moments against the new wrapper.
    std=nn.Parameter(checkpoint['log_std'].clone())
    ao=optimizer(after,std,checkpoint['config']['actor_lr'],checkpoint['actor_training']['weapon_state_residual']['learning_rate'])
    ao.load_state_dict(copy.deepcopy(checkpoint['actor_optimizer']))
    assert equal_tree(ao.state_dict(),checkpoint['actor_optimizer'])
    vo=torch.optim.Adam(branch(new,'value').parameters(),lr=checkpoint['config']['value_lr'])
    vo.load_state_dict(copy.deepcopy(checkpoint['value_optimizer']));assert equal_tree(vo.state_dict(),checkpoint['value_optimizer'])
    fork=copy.deepcopy(checkpoint);previous=copy.deepcopy(fork.get('sequence_retention'));fork['sequence_retention']=None
    receipt=dict(state='complete',device='cuda',version='conditional_weapon_exploration_fork_v1',temperature=args.temperature,
        source_weights_sha256=seal['weights_sha256'],source_checkpoint_sha256=seal['checkpoint_sha256'],
        data_report_sha256=sha(args.data/'report.json'),real_history_rows=len(rows),gated_rows=int(gate.sum()),
        state_max_error=max(ea,eb),unchanged_ungated_outputs_exact=True,unchanged_other_heads_exact=True,
        argmax_preserved=True,cuda_gate_gradient_verified=True,optimizer_groups_preserved=True,
        previous_sequence_retention=previous,training_performed=False,persisted_optimizer_steps=0,
        scope='Explicit sampling transform with identical tensors/Adam/RNG/history; fresh own-policy training required. Real evaluation histories used only for CUDA verification, never fitting or promotion.')
    args.out.mkdir(parents=True);durable_json(args.out/'weights.json',new);fork['weights_sha256']=sha(args.out/'weights.json')
    fork['model_migrations']=checkpoint.get('model_migrations',[])+[copy.deepcopy(receipt)]
    durable_write(args.out/'checkpoint.pt',lambda f:torch.save(fork,f))
    restored=torch.load(args.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    expected_cp=copy.deepcopy(checkpoint);expected_cp.update(weights_sha256=fork['weights_sha256'],sequence_retention=None,model_migrations=fork['model_migrations'])
    assert equal_tree(restored,expected_cp)
    assert equal_tree(branch(read(args.out/'weights.json'),'actor').state_dict(),restored['actor'])
    for p,d in pins.items():assert sha(p)==d,str(p)
    receipt.update(weights_sha256=fork['weights_sha256'],checkpoint_sha256=sha(args.out/'checkpoint.pt'),persisted_restoration_verified=True)
    durable_json(args.out/'report.json',receipt)
    durable_json(args.out/'complete.json',dict(state='complete',weights_sha256=receipt['weights_sha256'],checkpoint_sha256=receipt['checkpoint_sha256'],report_sha256=sha(args.out/'report.json')))
    print(dict(temperature=args.temperature,device='cuda',rows=len(rows),gated_rows=int(gate.sum()),optimizer_preserved=True),flush=True)


if __name__=='__main__':main()

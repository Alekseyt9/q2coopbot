"""Fit only conditional motion/aim residual; fresh registered data, CUDA only."""
import argparse
import copy
import gzip
import json
import pathlib

from ppo_recurrent import torch, nn, prepare_context, sequence, durable_json, durable_write
from migrate_combat_navigation_cuda import branch
from combat_precision_head import selected_means
from combat_sampled_motion_aim import correction, export, VERSION
from combat_sequence_retention import policy_kl
from combat_weapon_state_residual import optimizer
from fork_conditional_weapon_exploration_cuda import equal_tree
from process_combat_architecture_pool import read, sha


def main():
    ap=argparse.ArgumentParser()
    for name in ('source','labels','out'): ap.add_argument('--'+name,type=pathlib.Path,required=True)
    ap.add_argument('--steps',type=int,default=80)
    ap.add_argument('--learning-rate',type=float,default=.0003)
    ap.add_argument('--retention-weight',type=float,default=.02)
    args=ap.parse_args()
    assert torch.cuda.is_available() and not args.out.exists()
    assert 1<=args.steps<=200 and 0<args.learning_rate<=.001 and 0<=args.retention_weight<=1
    torch.set_default_device('cuda');torch.set_num_threads(2);torch.manual_seed(20261022)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    pins={}
    def pin(path,expected=None):
        path=pathlib.Path(path);digest=sha(path)
        assert expected is None or expected==digest
        if str(path) in pins: assert pins[str(path)]==digest
        pins[str(path)]=digest;return path
    seal=read(pin(args.source/'complete.json'))
    for name,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]: pin(args.source/name,seal[key])
    assert read(args.source/'report.json')['device']=='cuda'
    model=read(args.source/'weights.json');assert model.get('sampled_motion_aim')
    cp=torch.load(args.source/'checkpoint.pt',map_location='cuda',weights_only=True)
    actor=branch(model,'actor');assert equal_tree(actor.state_dict(),cp['actor'])
    assert cp['weights_sha256']==seal['weights_sha256'] and equal_tree(branch(model,'value').state_dict(),cp['value'])
    module=actor.sampled_motion_aim_branch;teacher=copy.deepcopy(module)
    for p in actor.parameters():p.requires_grad_(False)
    for p in teacher.parameters():p.requires_grad_(False)
    for p in module.parameters():p.requires_grad_(True)
    meta=read(pin(args.labels/'report.json'));assert meta['state']=='complete' and not meta['training_performed']
    for binding in meta['inputs']:pin(binding['path'],binding['sha256'])
    corpus=pathlib.Path(meta['corpus'])
    assert read(corpus/'protocol.json')['source']==str(args.source.resolve())
    std=torch.tensor(model['log_std'],dtype=torch.float32,device='cuda')
    datasets={};seen=set()
    with torch.no_grad():
        for split in ('train','development'):
            info=meta['splits'][split]
            with gzip.open(pin(info['output'],info['sha256']),'rt',encoding='utf-8') as stream:labels=list(map(json.loads,stream))
            root=corpus/('processing-'+split)/'merged';report=read(pin(root/'report.json'))
            assert report['model_sha256']==seal['weights_sha256']
            content={}
            for name in ('rollout','sequence'):
                path=pin(root/(name+'.jsonl'),report[name+'_sha256'])
                content[name]=[json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()]
            rows=content['rollout'];seeds={r['seed'] for r in rows};assert not seen.intersection(seeds);seen.update(seeds)
            prepared=prepare_context(content['sequence'],rows,'cuda')
            raw,error=sequence(actor,prepared,check_states=True);assert error<2e-5
            x=torch.tensor([r['features'] for r in rows],dtype=torch.float32,device='cuda')
            movement=torch.tensor([r['sample']['latent'][:2] for r in rows],dtype=torch.float32,device='cuda')
            positions={(r['seed'],r['index']):i for i,r in enumerate(rows)}
            selected=[];excluded=0
            for label in labels:
                i=positions[(label['seed'],label['index'])];row=rows[i]
                assert label['split']==split and label['features']==row['features'] and label['sample']==row['sample']
                target=label['sample']['target'];mode=label['sample'].get('aim_mode',0)
                desired=label['offline_label']['desired_command_delta_degrees'][::-1];scale=15. if mode else 180.
                if max(map(abs,desired))>=scale*.99 or label['offline_label']['muzzle_blocked'] or label['offline_label']['water']:
                    excluded+=1;continue
                selected.append((i,target,mode,desired,scale))
            assert selected
            ix=torch.tensor([s[0] for s in selected],dtype=torch.long,device='cuda')
            target=torch.tensor([s[1] for s in selected],dtype=torch.long,device='cuda')
            mode=torch.tensor([s[2] for s in selected],dtype=torch.long,device='cuda')
            scale=torch.tensor([s[4] for s in selected],dtype=torch.float32,device='cuda')[:,None]
            desired=torch.tensor([s[3] for s in selected],dtype=torch.float32,device='cuda')
            baseline=selected_means(raw[ix],target,mode)[:,2:4]
            # Pin a deterministic subset of TRAIN histories for retention. Dev
            # histories are evaluated only and never enter loss or selection.
            stride=max(1,(len(rows)+511)//512);reg=torch.arange(0,len(rows),stride,device='cuda')
            datasets[split]=dict(x=x[ix],raw=raw[ix],movement=movement[ix],target=target,mode=mode,
                baseline=baseline,desired=desired,desired_latent=torch.atanh(desired/scale),scale=scale,
                regularizer_x=x[reg],regularizer_raw=raw[reg],regularizer_movement=movement[reg],
                regularizer_keys=[(rows[i]['seed'],rows[i]['index']) for i in range(0,len(rows),stride)],
                samples=len(selected),excluded=excluded,rows=len(rows))

    def prediction(d):
        residual=correction(module,d['x'],d['raw'],d['movement'],d['target'],d['mode'])
        return d['baseline']+residual
    def retention(d):
        raw=d['regularizer_raw']
        return policy_kl(raw,std,raw,std,d['regularizer_x'],module,teacher,d['regularizer_movement']).mean()
    def metrics(d):
        predicted=prediction(d).tanh()*d['scale'];error=(predicted-d['desired']).abs()
        return dict(samples=d['samples'],excluded=d['excluded'],mean_absolute_axis_error_degrees=float(error.mean()),
            mean_joint_error_degrees=float(error.square().sum(-1).sqrt().mean()),conditional_retention=float(retention(d)))
    opt=torch.optim.Adam(module.parameters(),lr=args.learning_rate);trajectory=[]
    with torch.no_grad():trajectory.append(dict(step=0,**{s:metrics(d) for s,d in datasets.items()}))
    for step in range(1,args.steps+1):
        opt.zero_grad(set_to_none=True);d=datasets['train']
        fitting=nn.functional.smooth_l1_loss(prediction(d),d['desired_latent'],beta=.1)
        loss=fitting+args.retention_weight*retention(d);assert torch.isfinite(loss)
        loss.backward();assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in module.parameters())
        torch.nn.utils.clip_grad_norm_(module.parameters(),1.);opt.step()
        if step in (10,30,args.steps):
            with torch.no_grad():trajectory.append(dict(step=step,**{s:metrics(d) for s,d in datasets.items()}))
    updated=copy.deepcopy(model);export(updated,module)
    a,b=copy.deepcopy(model),copy.deepcopy(updated);a.pop('sampled_motion_aim');b.pop('sampled_motion_aim');assert a==b
    for name,tensor in cp['actor'].items():
        if not name.startswith('sampled_motion_aim_branch.'):assert torch.equal(tensor,actor.state_dict()[name])
    assert equal_tree(branch(updated,'actor').state_dict(),actor.state_dict())
    std_parameter=nn.Parameter(cp['log_std'].clone())
    ppo_opt=optimizer(actor,std_parameter,cp['config']['actor_lr'],cp['actor_training']['weapon_state_residual']['learning_rate'])
    ppo_opt.load_state_dict(copy.deepcopy(cp['actor_optimizer']))
    for p in module.parameters():ppo_opt.state.pop(p,None)
    args.out.mkdir(parents=True);durable_json(args.out/'weights.json',updated)
    receipt=dict(state='complete',device='cuda',version=VERSION,training_performed=True,promotion=False,
        source_weights_sha256=seal['weights_sha256'],source_checkpoint_sha256=seal['checkpoint_sha256'],
        steps=args.steps,learning_rate=args.learning_rate,retention_weight=args.retention_weight,
        trainable_parameters=sum(p.numel() for p in module.parameters()),gpu=torch.cuda.get_device_name(0),
        fixed_budget=True,selection_on_development=False,trajectory=trajectory,
        retention_train_keys=datasets['train']['regularizer_keys'],
        fresh_own_policy_required_before_ppo=True,ppo_moments_reset_scope='sampled_motion_aim_branch_only',
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),
        scope='Only sampled-motion conditional aim residual fitted from native MG body-point/recoil labels; independent development seeds. Frozen original actor/value/log_std. Fixed behavior-move conditional KL retention; geometric labels are not an optimal-action teacher. Native combat evaluation still required.')
    fork=copy.deepcopy(cp);fork.update(actor=actor.state_dict(),weights_sha256=sha(args.out/'weights.json'),
        actor_optimizer=ppo_opt.state_dict(),sequence_retention=None,sampled_motion_bc_optimizer=opt.state_dict())
    fork['model_migrations']=cp.get('model_migrations',[])+[copy.deepcopy(receipt)]
    durable_write(args.out/'checkpoint.pt',lambda f:torch.save(fork,f))
    assert equal_tree(torch.load(args.out/'checkpoint.pt',map_location='cuda',weights_only=True),fork)
    for path,digest in pins.items():assert sha(path)==digest
    durable_json(args.out/'report.json',receipt)
    durable_json(args.out/'complete.json',dict(version='combat_update_complete_v1',weights_sha256=sha(args.out/'weights.json'),checkpoint_sha256=sha(args.out/'checkpoint.pt'),report_sha256=sha(args.out/'report.json')))
    print(json.dumps(dict(trajectory=trajectory,trainable_parameters=receipt['trainable_parameters'])))


if __name__=='__main__':main()

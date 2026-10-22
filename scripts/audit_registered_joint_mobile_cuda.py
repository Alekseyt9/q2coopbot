"""Matched-budget joint BC ownership and current-encoder prefix replay audit."""
import argparse
import json
import pathlib
from process_combat_architecture_pool import read,save,sha


def main():
    ap=argparse.ArgumentParser()
    for name in ('source','heads','encoder','processing','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    args=ap.parse_args();assert not args.out.exists()
    import torch
    assert torch.cuda.is_available();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    from migrate_combat_navigation_cuda import branch
    from fork_conditional_weapon_exploration_cuda import equal_tree
    from ppo_recurrent import prepare_context,sequence
    from combat_sequence_retention import categorical_kl
    from combat_weapon_head import feature_mask
    from combat_weapon_state_residual import optimizer
    pins={};models={};cps={};protocols={};reports={}
    def pin(path,expected=None):
        path=pathlib.Path(path).resolve();digest=sha(path);assert expected is None or expected==digest
        pins[str(path)]=digest;return path
    for label,root in [('source',args.source),('heads',args.heads),('encoder',args.encoder)]:
        seal=read(pin(root/'complete.json'))
        for name,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:pin(root/name,seal[key])
        model=read(root/'weights.json');cp=torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=True)
        actor=branch(model,'actor');assert equal_tree(actor.state_dict(),cp['actor']) and equal_tree(branch(model,'value').state_dict(),cp['value'])
        assert cp['weights_sha256']==seal['weights_sha256']
        assert all(v.is_cuda and bool(torch.isfinite(v).all()) for tree in (cp['actor'],cp['value']) for v in tree.values())
        models[label]=actor;cps[label]=cp;reports[label]=read(root/'report.json')
        if label!='source':
            assert cp['joint_mobile_bc']['scope']==label and cp['joint_mobile_bc']['actor_adam_reset']
            protocol=read(pin(root/'protocol.json',cp['joint_mobile_bc']['protocol_sha256']))
            assert protocol['version']=='registered_joint_mobile_bc_v1' and protocol['scope']==label and not protocol['selection_on_development']
            for item in protocol['inputs']:pin(item['path'],item['sha256'])
            pin(root/'trainer.py',reports[label]['script_sha256']);protocols[label]=protocol
    for k in ('source','labels','steps','learning_rate','seed','movement_weight','aim_latent_huber_weight','mode_weight','target_weight','fire_weight','retention_weight','retention_train_keys','retention_train_draws'):
        assert protocols['heads'][k]==protocols['encoder'][k],k
    assert reports['heads']['before']==reports['encoder']['before']
    parent=cps['source'];allowed=torch.zeros(81,dtype=torch.bool,device='cuda');allowed[:5]=True;allowed[20:81]=True
    changes={}
    for label in ('heads','encoder'):
        child=cps[label]
        for k in ('value','value_optimizer','log_std','config','consumed_rollouts','updates_completed','total_actor_steps'):
            assert equal_tree(parent[k],child[k]),(label,k)
        assert not child['actor_optimizer']['state']
        std=torch.nn.Parameter(child['log_std'].detach().clone())
        standard=optimizer(models[label],std,child['config']['actor_lr'],child.get('actor_training',{}).get('weapon_state_residual',{}).get('learning_rate',.0003))
        standard.load_state_dict(child['actor_optimizer'])  # future PPO group compatibility
        changed=[]
        for k,v in child['actor'].items():
            old=parent['actor'][k]
            if k in ('base.head.weight','base.head.bias','base.residual.weight','base.residual.bias'):assert torch.equal(v[~allowed],old[~allowed]),(label,k)
            elif k.startswith(('branch.','sampled_motion_aim_branch.')):pass
            elif label=='encoder' and k.startswith(('base.encoder.','base.mha.')):pass
            else:assert torch.equal(v,old),(label,k)
            if not torch.equal(v,old):changed.append(k)
        if label=='encoder':
            assert any(k.startswith('base.encoder.') for k in changed) and any(k.startswith('base.mha.') for k in changed)
        changes[label]=changed
    meta=read(pin(args.processing/'merged/report.json'));assert meta['model_sha256']==sha(args.source/'weights.json')
    streams={}
    for name in ('rollout','sequence'):
        path=pin(args.processing/'merged'/(name+'.jsonl'),meta[name+'_sha256'])
        streams[name]=[json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()]
    prepared=prepare_context(streams['sequence'],streams['rollout'],'cuda');x,indices,inverse,segments=prepared
    with torch.no_grad():
        reference,error=sequence(models['source'],prepared,check_states=True);assert error<2e-5
        features=torch.tensor([r['features'] for r in streams['rollout']],dtype=torch.float32,device='cuda')
        diagnostic={}
        for label in ('heads','encoder'):
            actor=models[label];full,_=actor(x);raw=full[indices[:,0],indices[:,1]][inverse]
            prefix_errors=[]
            members=[b for b,s in enumerate(segments) if len(s)>=40][:2];assert members
            for b in members:
                t=39;prefix,_=actor(x[b:b+1,:t+1]);delta=float((prefix[0,-1]-full[b,t]).abs().max())
                torch.testing.assert_close(prefix[0,-1],full[b,t],rtol=1e-5,atol=3e-5);prefix_errors.append(delta)
            # Candidate encoder is recomputed on raw histories. Its new tokens
            # must NOT be compared to old-source cached encoded memory.
            diagnostic[label]=dict(prefix_causal_max_error=max(prefix_errors),
                mean_weapon_kl=float(categorical_kl(reference[:,8:20].double(),raw[:,8:20].double(),feature_mask(features[:,:845])).mean()),
                mean_vertical_kl=float(categorical_kl(reference[:,5:8].double(),raw[:,5:8].double()).mean()))
    for p,h in pins.items():assert sha(p)==h
    save(args.out,dict(state='complete',version='joint_mobile_bc_pair_cuda_audit_v1',device='cuda',
        rows=len(streams['rollout']),sequence_rows=len(streams['sequence']),source_memory_error=error,matched_initial_metrics=True,
        matched_train_retention_draws=True,changed_actor_tensors=changes,diagnostic=diagnostic,actor_adam_reset=True,critic_std_unchanged=True,
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),training_performed=False,promotion=False,
        scope='Same source/data/budget/seed/loss/draws;only encoder eligibility differs. CUDA finite checkpoint/JSON and projected parameter ownership. Current encoder prefixes recomputed,causality checked without old-source encoded caches. Frozen weapon/vertical parameter rows do not guarantee frozen outputs. Fresh native replay/quality still required.'))
    print(json.dumps(dict(state='complete',rows=len(streams['rollout']),changed_tensors={k:len(v) for k,v in changes.items()},diagnostic=diagnostic)))


if __name__=='__main__':main()

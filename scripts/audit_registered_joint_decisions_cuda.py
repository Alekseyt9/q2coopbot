"""Independent checkpoint/frozen-head audit plus native-history CUDA replay."""
import argparse
import json
import pathlib

from process_combat_architecture_pool import read, save, sha


def main():
    ap=argparse.ArgumentParser()
    for name in ('source','candidate','processing','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    args=ap.parse_args();assert not args.out.exists()
    import torch
    assert torch.cuda.is_available();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    from migrate_combat_navigation_cuda import branch
    from ppo_recurrent import prepare_context,sequence
    from fork_conditional_weapon_exploration_cuda import equal_tree
    pins={};models={};checkpoints={}
    def pin(path,expected=None):
        path=pathlib.Path(path).resolve();digest=sha(path);assert expected is None or expected==digest
        pins[str(path)]=digest;return path
    for label,root in [('source',args.source),('candidate',args.candidate)]:
        seal=read(pin(root/'complete.json'))
        for name,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:pin(root/name,seal[key])
        model=read(root/'weights.json');cp=torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=True)
        actor=branch(model,'actor');assert equal_tree(actor.state_dict(),cp['actor']) and equal_tree(branch(model,'value').state_dict(),cp['value'])
        assert cp['weights_sha256']==seal['weights_sha256']
        for tree in (cp['actor'],cp['value']):
            assert all(v.is_cuda and bool(torch.isfinite(v).all()) for v in tree.values())
        models[label]=actor;checkpoints[label]=cp
    old,new=checkpoints['source'],checkpoints['candidate']
    for key in ('value','value_optimizer','log_std','config','consumed_rollouts','updates_completed','total_actor_steps'):
        assert equal_tree(old[key],new[key]),key
    assert new['actor_optimizer']['state']=={} and new['joint_decision_bc']['actor_adam_reset']
    chosen=torch.zeros(81,dtype=torch.bool,device='cuda');chosen[4]=True;chosen[8:29]=True
    changed=[]
    for key,value in new['actor'].items():
        reference=old['actor'][key]
        if key in ('base.head.weight','base.head.bias','base.residual.weight','base.residual.bias'):
            assert torch.equal(value[~chosen],reference[~chosen])
        elif not key.startswith('target_branch.'):
            assert torch.equal(value,reference),key
        if not torch.equal(value,reference):changed.append(key)
    assert changed
    meta=read(pin(args.processing/'merged/report.json'));assert meta['model_sha256']==sha(args.source/'weights.json')
    streams={}
    for name in ('rollout','sequence'):
        path=pin(args.processing/'merged'/(name+'.jsonl'),meta[name+'_sha256'])
        streams[name]=[json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()]
    prepared=prepare_context(streams['sequence'],streams['rollout'],'cuda')
    outputs={};memory={}
    with torch.no_grad():
        for label,actor in models.items():
            outputs[label],memory[label]=sequence(actor,prepared,check_states=True)
            assert memory[label]<2e-5
        error=float((outputs['source'][:,~chosen]-outputs['candidate'][:,~chosen]).abs().max())
        assert error==0.
    for path,digest in pins.items():assert sha(path)==digest
    save(args.out,dict(state='complete',version='joint_decision_bc_cuda_audit_v1',device='cuda',rows=len(streams['rollout']),
        sequence_rows=len(streams['sequence']),changed_actor_tensors=changed,frozen_nondecision_max_error=error,
        actor_memory_errors=memory,critic_optimizer_unchanged=True,actor_adam_reset=True,training_performed=False,promotion=False,
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),
        scope='Exact projected parameter ownership, checkpoint/JSON agreement and full native source histories on CUDA. No quality evaluation or teacher BC optimality claim. Live target decisions may change subsequent observations/movement.'))
    print(json.dumps(dict(state='complete',rows=len(streams['rollout']),frozen_error=error,changed=changed)))


if __name__=='__main__':main()

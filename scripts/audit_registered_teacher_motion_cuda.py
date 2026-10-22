"""Independent CUDA ownership/checkpoint and real source-history motion audit."""
import argparse
import json
import pathlib
from process_combat_architecture_pool import read,save,sha


def main():
    ap=argparse.ArgumentParser()
    for name in ('source','candidate','processing','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    args=ap.parse_args();assert not args.out.exists()
    import torch
    assert torch.cuda.is_available();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    from migrate_combat_navigation_cuda import branch
    from fork_conditional_weapon_exploration_cuda import equal_tree
    from ppo_recurrent import prepare_context,sequence
    pins={};models={};checkpoints={}
    def pin(path,expected=None):
        path=pathlib.Path(path).resolve();digest=sha(path);assert expected is None or expected==digest
        pins[str(path)]=digest;return path
    for name,root in [('source',args.source),('candidate',args.candidate)]:
        seal=read(pin(root/'complete.json'))
        for filename,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:pin(root/filename,seal[key])
        cp=torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=True);model=read(root/'weights.json')
        assert cp['weights_sha256']==seal['weights_sha256']
        assert equal_tree(cp['actor'],branch(model,'actor').state_dict()) and equal_tree(cp['value'],branch(model,'value').state_dict())
        assert all(v.is_cuda and bool(torch.isfinite(v).all()) for tree in (cp['actor'],cp['value']) for v in tree.values())
        checkpoints[name]=cp;models[name]=branch(model,'actor')
    parent,child=checkpoints['source'],checkpoints['candidate']
    for k in ('value','value_optimizer','log_std','config','consumed_rollouts','updates_completed','total_actor_steps'):
        assert equal_tree(parent[k],child[k]),k
    assert not child['actor_optimizer']['state'] and child['teacher_motion_bc']['actor_adam_reset']
    protocol=read(pin(args.candidate/'protocol.json',child['teacher_motion_bc']['protocol_sha256']))
    for item in protocol['inputs']:pin(item['path'],item['sha256'])
    mask=torch.zeros(81,dtype=torch.bool,device='cuda');mask[:2]=True;mask[63:81]=True
    changed=[]
    for k,v in child['actor'].items():
        old=parent['actor'][k]
        if k in ('base.head.weight','base.head.bias','base.residual.weight','base.residual.bias'):assert torch.equal(v[~mask],old[~mask]),k
        elif not k.startswith('sampled_motion_aim_branch.'):assert torch.equal(v,old),k
        if not torch.equal(v,old):changed.append(k)
    assert changed
    meta=read(pin(args.processing/'merged/report.json'));assert meta['model_sha256']==sha(args.source/'weights.json')
    streams={}
    for name in ('rollout','sequence'):
        path=pin(args.processing/'merged'/(name+'.jsonl'),meta[name+'_sha256'])
        streams[name]=[json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()]
    prepared=prepare_context(streams['sequence'],streams['rollout'],'cuda');out={};errors={}
    with torch.no_grad():
        for name,model in models.items():
            out[name],errors[name]=sequence(model,prepared,check_states=True);assert errors[name]<2e-5
        preserved=float((out['source'][:,2:29]-out['candidate'][:,2:29]).abs().max());assert preserved==0.
    for path,digest in pins.items():assert sha(path)==digest
    save(args.out,dict(state='complete',version='registered_teacher_motion_cuda_audit_v1',device='cuda',
        rows=len(streams['rollout']),sequence_rows=len(streams['sequence']),actor_memory_errors=errors,
        frozen_raw_2_29_max_error=preserved,changed_actor_tensors=changed,actor_adam_reset=True,critic_unchanged=True,
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),training_performed=False,promotion=False,
        scope='CUDA finite checkpoint/JSON/parameter ownership and complete real source histories. Only movement/mode rows and conditional aim weights differ. Base spatial weights frozen, their effective aim may change through movement inputs. Target/fire/weapon/vertical raw2:29/std/value unchanged on matched histories. No live success or optimal-action proof.'))
    print(json.dumps(dict(state='complete',rows=len(streams['rollout']),frozen_error=preserved,changed=changed)))


if __name__=='__main__':main()

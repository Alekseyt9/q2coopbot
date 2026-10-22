"""Independent two-bias calibration ownership and native-history CUDA audit."""
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
    pins={};models={};cps={}
    def pin(path,expected=None):
        path=pathlib.Path(path).resolve();digest=sha(path);assert expected is None or expected==digest
        pins[str(path)]=digest;return path
    for name,root in [('source',args.source),('candidate',args.candidate)]:
        seal=read(pin(root/'complete.json'))
        for file,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:pin(root/file,seal[key])
        model=read(root/'weights.json');cp=torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=True)
        actor=branch(model,'actor');assert equal_tree(actor.state_dict(),cp['actor']) and equal_tree(branch(model,'value').state_dict(),cp['value'])
        assert cp['weights_sha256']==seal['weights_sha256']
        assert all(bool(torch.isfinite(v).all()) for tree in (cp['actor'],cp['value']) for v in tree.values())
        models[name]=actor;cps[name]=cp
    before,after=cps['source'],cps['candidate'];calibration=after['aim_mode_calibration']
    assert calibration['fit_split']=='train' and calibration['source_weights_sha256']==sha(args.source/'weights.json')
    protocol=read(pin(args.candidate/'protocol.json',after['teacher_motion_bc']['protocol_sha256']))
    assert protocol['fit_split']=='train' and not protocol['selection_on_development']
    for binding in protocol['inputs']:pin(binding['path'],binding['sha256'])
    report=read(args.candidate/'report.json');pin(args.candidate/'calibrator.py',report['script_sha256'])
    for k in ('value','value_optimizer','log_std','actor_optimizer','config','consumed_rollouts','updates_completed','total_actor_steps'):
        assert equal_tree(before[k],after[k]),k
    indices=[]
    for change in calibration['changes']:
        t=change['target'];assert change['coarse']>0 and change['fine']>0
        indices.extend((63+2*t,64+2*t))
    assert indices and len(indices)==len(set(indices));mask=torch.zeros(81,dtype=torch.bool,device='cuda');mask[indices]=True
    for k,v in after['actor'].items():
        if k=='base.head.bias':assert torch.equal(v[~mask],before['actor'][k][~mask])
        else:assert torch.equal(v,before['actor'][k]),k
    for c in calibration['changes']:
        j=63+2*c['target'];actual=after['actor']['base.head.bias'][j:j+2]-before['actor']['base.head.bias'][j:j+2]
        expected=torch.tensor([c['logit_difference_shift']/2,-c['logit_difference_shift']/2],device='cuda')
        torch.testing.assert_close(actual,expected,rtol=0,atol=2e-6)
    meta=read(pin(args.processing/'merged/report.json'));assert meta['model_sha256']==sha(args.source/'weights.json')
    streams={}
    for name in ('rollout','sequence'):
        path=pin(args.processing/'merged'/(name+'.jsonl'),meta[name+'_sha256'])
        streams[name]=[json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()]
    prepared=prepare_context(streams['sequence'],streams['rollout'],'cuda');raw={};memory={}
    with torch.no_grad():
        for name,actor in models.items():raw[name],memory[name]=sequence(actor,prepared,check_states=True);assert memory[name]<2e-5
        frozen_error=float((raw['source'][:,~mask]-raw['candidate'][:,~mask]).abs().max());assert frozen_error==0.
    for p,h in pins.items():assert sha(p)==h
    save(args.out,dict(state='complete',version='aim_mode_intercept_cuda_audit_v1',device='cuda',changed_bias_indices=indices,
        rows=len(streams['rollout']),sequence_rows=len(streams['sequence']),actor_memory_errors=memory,
        frozen_raw_max_error=frozen_error,inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),
        training_performed=False,promotion=False,
        scope='Only reported target-mode bias entries differ. CUDA JSON/checkpoint and complete native source histories agree. Other outputs/actor weights/std/value/Adam/config/history preserved;live coarse/fine sampling and resulting trajectory may change.'))
    print(json.dumps(dict(state='complete',rows=len(streams['rollout']),changed_bias_indices=indices,frozen_error=frozen_error)))


if __name__=='__main__':main()

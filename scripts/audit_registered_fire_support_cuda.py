"""Own-trajectory/cross-policy fire probabilities on CUDA; no fitting."""
import argparse
import json
import pathlib
from process_combat_architecture_pool import read,save,sha


def main():
    ap=argparse.ArgumentParser()
    for name in ('source','candidate','evaluation','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    ap.add_argument('--source-label',default='control-target-fire-bc');ap.add_argument('--candidate-label',default='joint-heads')
    args=ap.parse_args();assert not args.out.exists()
    import torch
    assert torch.cuda.is_available();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    from migrate_combat_navigation_cuda import branch
    from fork_conditional_weapon_exploration_cuda import equal_tree
    from ppo_recurrent import prepare_context,sequence
    from combat_target_head import availability
    pins={};models={};weights={}
    def pin(path,expected=None):
        path=pathlib.Path(path).resolve();digest=sha(path);assert expected is None or expected==digest
        pins[str(path)]=digest;return path
    for name,root in [('source',args.source),('candidate',args.candidate)]:
        seal=read(pin(root/'complete.json'))
        for file,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256')]:pin(root/file,seal[key])
        model=read(root/'weights.json');actor=branch(model,'actor');cp=torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=True)
        assert equal_tree(actor.state_dict(),cp['actor']);models[name]=actor;weights[name]=seal['weights_sha256']
    closed=read(pin(args.evaluation/'complete.json'));assert closed['state']=='complete' and not closed['training_performed']
    result={}
    with torch.no_grad():
        for owner,label in [('source',args.source_label),('candidate',args.candidate_label)]:
            root=args.evaluation/('processing-'+label)/'merged';meta=read(pin(root/'report.json'))
            assert meta['model_sha256']==weights[owner]
            streams={}
            for name in ('rollout','sequence'):
                path=pin(root/(name+'.jsonl'),meta[name+'_sha256'])
                streams[name]=[json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()]
            rows=streams['rollout'];prepared=prepare_context(streams['sequence'],rows,'cuda')
            features=torch.tensor([r['features'] for r in rows],dtype=torch.float32,device='cuda')
            selected=torch.tensor([r['sample'].get('target',0) for r in rows],dtype=torch.long,device='cuda')
            attacks=torch.tensor([r['sample']['attack'] for r in rows],dtype=torch.float32,device='cuda')
            available=availability(features)[:,1:].any(-1)
            masks=dict(no_available_target=~available,no_selected_target=selected==0,
                none_despite_available=(selected==0)&available,selected_target=selected>0)
            result[owner]=dict(rows=len(rows),models={})
            for name,actor in models.items():
                own=weights[name]==meta['model_sha256'];raw,error=sequence(actor,prepared,check_states=own)
                if own:assert error<2e-5
                p=raw[:,4].sigmoid();cells={}
                for condition,mask in masks.items():
                    n=int(mask.sum());cells[condition]=dict(rows=n,
                        fire_probability_mean=float(p[mask].mean()) if n else None,
                        expected_fire_commands=float(p[mask].sum()),
                        logged_sample_fire_fraction=float(attacks[mask].mean()) if n else None)
                result[owner]['models'][name]=dict(own_memory_verified=own,memory_error=error if own else None,conditions=cells)
    for p,h in pins.items():assert sha(p)==h
    save(args.out,dict(state='complete',version='registered_fire_support_cuda_v1',device='cuda',contexts=result,
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),training_performed=False,promotion=False,
        scope='Unmodified models on same raw own-policy source/candidate histories. Mean Bernoulli fire probability versus logged samples;not guarded commands or native shots. Empty availability/selected target groups differ;missing enemy does not prove every shot useless. Context distributions are observed,not a causal proof or a training dataset.'))
    print(json.dumps(result))


if __name__=='__main__':main()

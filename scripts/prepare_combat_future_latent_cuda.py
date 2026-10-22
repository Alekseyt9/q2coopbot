"""Freeze current encoder and causal attention readouts for the future probe."""
import argparse
import json
import pathlib

from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser()
    for name in ('source','data','corpus','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    args=ap.parse_args();assert not args.out.exists()
    import numpy as np
    import torch
    assert torch.cuda.is_available(),'CUDA required; no CPU fallback'
    torch.set_default_device('cuda');torch.set_num_threads(2);torch.manual_seed(20261022)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    from migrate_combat_navigation_cuda import branch
    from ppo_recurrent import prepare_context,sequence
    from fork_conditional_weapon_exploration_cuda import equal_tree
    pins={}
    def pin(path,expected=None):
        path=pathlib.Path(path).resolve();digest=sha(path);assert expected is None or expected==digest,str(path)
        pins[str(path)]=digest;return path
    source=args.source.resolve();corpus=args.corpus.resolve()
    seal=read(pin(source/'complete.json'))
    for f,k in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):pin(source/f,seal[k])
    model=read(source/'weights.json');cp=torch.load(source/'checkpoint.pt',map_location='cuda',weights_only=True)
    actor=branch(model,'actor').eval();assert equal_tree(actor.state_dict(),cp['actor'])
    for p in actor.parameters():p.requires_grad_(False)
    assert model['attention']['version']=='combat_causal_attention_v1' and model['attention']['window']==32
    data_meta=read(pin(args.data/'complete.json'));assert data_meta['version']=='combat_observed_future_dataset_v1'
    for item in data_meta['inputs']:pin(item['path'],item['sha256'])
    closed=read(pin(corpus/'complete.json'));assert closed['state']=='complete'
    protocol=read(pin(corpus/'protocol.json',closed['protocol_sha256']));assert pathlib.Path(protocol['source']).resolve()==source
    pool=read(pin(corpus/'pool/report.json',closed['pool_sha256']));assert pool['source_unchanged'] and pool['state']=='complete'
    assert not any(j['error'] for j in pool['jobs'])
    cache={};proofs=[];max_memory=max_causal=max_decompose=0.
    for plan_index,assignment in enumerate(protocol['assignments']):
        split=assignment['label'];assert split in ('train','development')
        with np.load(pin(data_meta['splits'][split]['path'],data_meta['splits'][split]['sha256']),allow_pickle=False) as arrays:
            keys=list(zip(arrays['seed'].tolist(),arrays['frame'].tolist()))
            features=torch.as_tensor(arrays['x'][:,:1147],device='cuda');case_index=torch.as_tensor(arrays['case_index'],device='cuda')
        assert len(keys)==len(set(keys));lookup={k:i for i,k in enumerate(keys)}
        base=torch.zeros(len(keys),64,device='cuda');attn=torch.zeros_like(base);filled=set()
        receipt=next(p for p in closed['processed'] if p['label']==split)
        process=read(pin(receipt['report'],receipt['sha256']));assert process['device']=='cuda' and process['model_sha256']==seal['weights_sha256']
        for job in sorted((j for j in pool['jobs'] if j['plan_index']==plan_index),key=lambda j:j['job']):
            directory=corpus/('processing-'+split)/f"case-{job['job']:04d}"/'native'
            native=read(pin(directory/'report.json'));assert native['model_sha256']==seal['weights_sha256']
            with pin(directory/'sequence.jsonl',native['sequence_sha256']).open(encoding='utf-8') as stream:context=list(map(json.loads,stream))
            # Use all canonical provider context, not only surviving label rows.
            rows=[dict(seed=c['seed'],index=c['index'],frame=c['frame'],features=c['features'],sample=dict(memory=c['memory'])) for c in context]
            prepared=prepare_context(context,rows,'cuda');x,indices,inverse,segments=prepared
            with torch.no_grad():
                _,memory_error=sequence(actor,prepared,check_states=True)
                assert memory_error<2e-5;max_memory=max(max_memory,memory_error)
                core=actor.base;encoded=core.encoder(x);raw,tokens=core(x);length=x.shape[1]
                positions=torch.arange(length,device='cuda');mask=(positions[None,:]>positions[:,None])|(positions[None,:]<positions[:,None]-core.window+1)
                attended,_=core.mha(tokens,tokens,tokens,attn_mask=mask,need_weights=False)
                err=float((core.head(encoded)+core.residual(attended)-raw).abs().max());assert err<1e-6;max_decompose=max(max_decompose,err)
                cut=max(0,length//2-1);future=x.clone();future[:,cut+1:]+=3*torch.randn_like(future[:,cut+1:])
                _,future_tokens=core(future)
                changed,_=core.mha(future_tokens,future_tokens,future_tokens,attn_mask=mask,need_weights=False)
                causal=float((changed[:,:cut+1]-attended[:,:cut+1]).abs().max());assert causal<2e-5
                prefix_tokens=core(x[:,:cut+1])[1];p=torch.arange(cut+1,device='cuda');prefix_mask=(p[None,:]>p[:,None])|(p[None,:]<p[:,None]-core.window+1)
                prefix,_=core.mha(prefix_tokens,prefix_tokens,prefix_tokens,attn_mask=prefix_mask,need_weights=False)
                cropped=float((prefix-attended[:,:cut+1]).abs().max());assert cropped<2e-5;max_causal=max(max_causal,causal,cropped)
                e=encoded[indices[:,0],indices[:,1]][inverse];a=attended[indices[:,0],indices[:,1]][inverse]
                canonical=x[indices[:,0],indices[:,1]][inverse]
                retained=0
                for i,c in enumerate(context):
                    k=(c['seed'],c['frame'])
                    if k not in lookup:continue
                    j=lookup[k];assert j not in filled;filled.add(j);retained+=1
                    assert torch.equal(canonical[i],features[j]),'Current features changed before encoder'
                    base[j]=e[i];attn[j]=a[i]
            proofs.append(dict(split=split,job=job['job'],seed=job['seed'],context_rows=len(context),retained=retained,
                memory_max_error=memory_error,future_perturbation_max_error=causal,prefix_crop_max_error=cropped))
        assert len(filled)==len(keys)
        cache[split]=dict(encoder=base,attention=attn,seed=torch.tensor([k[0] for k in keys],device='cuda',dtype=torch.long),
                         frame=torch.tensor([k[1] for k in keys],device='cuda',dtype=torch.long),case_index=case_index)
    assert equal_tree(actor.state_dict(),cp['actor']) and all(p.grad is None for p in actor.parameters())
    args.out.mkdir(parents=True)
    torch.save(cache,args.out/'cache.pt');restored=torch.load(args.out/'cache.pt',map_location='cuda',weights_only=True)
    assert all(torch.equal(v,restored[s][k]) for s,d in cache.items() for k,v in d.items())
    for p,h in pins.items():assert sha(p)==h
    save(args.out/'complete.json',dict(state='complete',version='combat_future_frozen_actor_cache_v1',device='cuda',
        cache_sha256=sha(args.out/'cache.pt'),dataset=str(args.data.resolve()),dataset_sha256=sha(args.data/'complete.json'),
        source=str(source),source_weights_sha256=seal['weights_sha256'],source_checkpoint_sha256=seal['checkpoint_sha256'],
        encoder_width=64,attention_width=64,combined_width=128,memory_max_error=max_memory,causal_max_error=max_causal,
        head_decomposition_max_error=max_decompose,cases=proofs,inputs=[dict(path=p,sha256=h) for p,h in pins.items()],
        script_sha256=sha(__file__),training_performed=False,actor_changed=False,promotion=False,
        scope='Current encoder plus causal window32 attention result; full canonical Go memory context and real row keys. Future perturbation/cropped-prefix audits. Not all policy information: spatial/weapon branches also read raw observed features directly. Frozen-trunk readout quality is not total policy information or game quality.'))
    print(args.out/'complete.json')


if __name__=='__main__':main()

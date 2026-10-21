"""Measure learned dependence on sampled commands at fixed real histories."""
import argparse
import gzip
import json
import pathlib
from ppo_recurrent import torch, prepare_context, sequence
from migrate_combat_navigation_cuda import branch
from combat_sampled_motion_aim import correction
from combat_precision_head import selected_means
from process_combat_architecture_pool import read, save, sha


def main():
    ap=argparse.ArgumentParser()
    for name in ('update','labels','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    args=ap.parse_args();assert torch.cuda.is_available() and not args.out.exists()
    torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    pins={}
    def pin(path,expected=None):
        path=pathlib.Path(path);digest=sha(path);assert expected is None or digest==expected
        pins[str(path)]=digest;return path
    seal=read(pin(args.update/'complete.json'))
    model=read(pin(args.update/'weights.json',seal['weights_sha256']))
    meta=read(pin(args.labels/'report.json'));corpus=pathlib.Path(meta['corpus'])
    protocol=read(pin(corpus/'protocol.json'));source=read(pin(pathlib.Path(protocol['source'])/'weights.json'))
    a,b=dict(model),dict(source);a.pop('sampled_motion_aim');b.pop('sampled_motion_aim');assert a==b
    actor=branch(model,'actor');assert actor.sampled_motion_aim_branch is not None
    for p in actor.parameters():p.requires_grad_(False)
    results={}
    for split,binding in meta['splits'].items():
        with gzip.open(pin(binding['output'],binding['sha256']),'rt',encoding='utf-8') as stream:labels=list(map(json.loads,stream))
        root=corpus/('processing-'+split)/'merged';info=read(pin(root/'report.json'));streams={}
        for name in ('rollout','sequence'):
            path=pin(root/(name+'.jsonl'),info[name+'_sha256'])
            streams[name]=[json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()]
        rows=streams['rollout'];positions={(r['seed'],r['index']):i for i,r in enumerate(rows)}
        prepared=prepare_context(streams['sequence'],rows,'cuda')
        with torch.no_grad():
            raw,error=sequence(actor,prepared,check_states=True);assert error<2e-5
            ix=torch.tensor([positions[(r['seed'],r['index'])] for r in labels],dtype=torch.long,device='cuda')
            raw=raw[ix]
            x=torch.tensor([r['features'] for r in labels],dtype=torch.float32,device='cuda')
            movement=torch.tensor([r['sample']['latent'][:2] for r in labels],dtype=torch.float32,device='cuda')
            targets=torch.tensor([r['sample']['target'] for r in labels],dtype=torch.long,device='cuda')
            modes=torch.tensor([r['sample'].get('aim_mode',0) for r in labels],dtype=torch.long,device='cuda')
            module=actor.sampled_motion_aim_branch
            sampled=correction(module,x,raw,movement,targets,modes)
            mean=correction(module,x,raw,raw[:,:2],targets,modes)
            base=selected_means(raw,targets,modes)[:,2:4]
            scale=torch.where(modes==1,15.,180.)[:,None]
            effect=((base+sampled).tanh()-(base+mean).tanh())*scale
            result=dict(shots=len(labels),mean_absolute_physical_axis_effect_degrees=float(effect.abs().mean()),
                max_physical_axis_effect_degrees=float(effect.abs().max()),
                latent_response_rms=float((sampled-mean).square().mean().sqrt()))
        movement=movement.detach().requires_grad_(True)
        output=correction(module,x,raw.detach(),movement,targets,modes)
        gradients=[]
        for axis in range(2):
            gradient,=torch.autograd.grad(output[:,axis].sum(),movement,retain_graph=axis==0)
            assert bool(torch.isfinite(gradient).all());gradients.append(gradient)
        jacobian=torch.stack(gradients,dim=1)
        result['mean_latent_jacobian_norm']=float(jacobian.square().sum((1,2)).sqrt().mean())
        result['nonzero_gradient_shots']=int((jacobian.abs().amax((1,2))>1e-8).sum())
        assert result['nonzero_gradient_shots']>0 and result['latent_response_rms']>0
        results[split]=result
    for path,digest in pins.items():assert sha(path)==digest
    save(args.out,dict(state='complete',device='cuda',version='sampled_motion_dependence_v1',
        results=results,inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),
        training_performed=False,promotion=False,
        scope='Fixed recorded histories, sampled-command vs mean-command conditional aim and Jacobian. Shows learned dependence, not whether it is useful in closed-loop combat or actual post-guard displacement. No weights fitted.'))
    print(json.dumps(results))


if __name__=='__main__':main()

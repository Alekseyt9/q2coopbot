"""Measure initial shared-trunk PPO/future gradient scales on TRAIN only."""
import argparse
import json
import pathlib

from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    a=ap.parse_args();assert not a.out.exists()
    import torch
    assert torch.cuda.is_available()
    torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    from migrate_combat_navigation_cuda import branch
    from ppo_recurrent import prepare_context,sequence
    from ppo_combat import advantages
    from combat_precision_head import probabilities
    from combat_future_auxiliary import FutureAuxiliary
    protocol=read(a.root/'protocol.json');source=pathlib.Path(protocol['source']);corpus=pathlib.Path(protocol['corpus'])
    model=read(source/'weights.json');actor=branch(model,'actor');config=read(corpus/'training-config.json')
    merged=corpus/'processing-train/merged';meta=read(merged/'report.json')
    assert meta['model_sha256']==sha(source/'weights.json') and sha(merged/'rollout.jsonl')==meta['rollout_sha256']
    with (merged/'rollout.jsonl').open() as f:rows=list(map(json.loads,f))
    with (merged/'sequence.jsonl').open() as f:context=list(map(json.loads,f))
    prepared=prepare_context(context,rows,'cuda')
    x=torch.tensor([r['features'] for r in rows],dtype=torch.float32,device='cuda')
    z=torch.tensor([r['sample']['latent'] for r in rows],device='cuda')
    condition=torch.tensor([[int(r['sample']['attack']),r['sample']['vertical'],r['sample'].get('weapon',0),r['sample'].get('target',0),r['sample'].get('aim_mode',0)] for r in rows],device='cuda',dtype=torch.long)
    adv,_=advantages(rows,config['gamma'],config['lambda']);adv=torch.tensor(adv,device='cuda');adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    old=torch.tensor([r['sample']['log_probability'] for r in rows],device='cuda');std=torch.tensor(model['log_std'],device='cuda')
    raw,_=sequence(actor,prepared)
    lp,entropy=probabilities(raw,std,z,condition[:,0].float(),condition[:,1],condition[:,2],x,condition[:,3],condition[:,4],actor.sampled_motion_aim_branch)
    ratio=(lp-old).exp();loss=-torch.minimum(ratio*adv,ratio.clamp(1-config['clip'],1+config['clip'])*adv).mean()-config['entropy']*entropy.mean()
    params=[p for n,p in actor.base.named_parameters() if n.startswith(('encoder.','mha.'))]
    rl=torch.autograd.grad(loss,params,allow_unused=True)
    aux=FutureAuxiliary(actor,prepared,protocol['dataset'],protocol['decoder'],.1,sha(source/'weights.json'))
    future=torch.autograd.grad(aux.actor_loss(actor),params,allow_unused=True)
    rn=sum(float(g.square().sum()) for g in rl if g is not None)**.5
    fn=sum(float(g.square().sum()) for g in future if g is not None)**.5
    dot=sum(float((p*q).sum()) for p,q in zip(rl,future) if p is not None and q is not None)
    aux.verify()
    save(a.out,dict(state='complete',device='cuda',ppo_shared_trunk_grad_norm=rn,weighted_future_shared_trunk_grad_norm=fn,
        weighted_future_to_ppo_norm_ratio=fn/rn if rn else None,gradient_cosine=dot/(rn*fn) if rn and fn else None,
        auxiliary_weight=.1,shared_parameters=sum(p.numel() for p in params),rows=len(rows),future_rows=aux.spec['rows'],
        rollout_sha256=meta['rollout_sha256'],source_weights_sha256=sha(source/'weights.json'),training_performed=False,actor_changed=False,
        script_sha256=sha(__file__),scope='Initial encoder/MHA gradients before clipping/Adam only. PPO surrogate+entropy; initial same-policy retention has zero theoretical gradient. Not per-step influence, a calibrated coefficient, validation fit, or native quality.'))
    print(a.out)


if __name__=='__main__':main()

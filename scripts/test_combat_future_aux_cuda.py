"""Real-data CUDA gradient/RNG isolation check for the PPO auxiliary seam."""
import argparse
import json
import os
import pathlib

from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser()
    for name in ('source','data','context','decoder','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    ap.add_argument('--weight',type=float,default=.1)
    a=ap.parse_args();assert not a.out.exists()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    assert torch.cuda.is_available()
    torch.set_default_device('cuda');torch.set_num_threads(2);torch.manual_seed(20261022)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    from migrate_combat_navigation_cuda import branch
    from ppo_recurrent import prepare_context
    from combat_future_auxiliary import FutureAuxiliary
    model=read(a.source/'weights.json');actor=branch(model,'actor')
    with a.context.open(encoding='utf-8') as f:context=list(map(json.loads,f))
    rows=[dict(seed=c['seed'],index=c['index'],frame=c['frame'],features=c['features'],sample=dict(memory=c['memory'])) for c in context]
    prepared=prepare_context(context,rows,'cuda')
    snapshot={k:v.detach().clone() for k,v in actor.state_dict().items()}
    rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
    control=FutureAuxiliary(actor,prepared,a.data,a.decoder,0,sha(a.source/'weights.json'))
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
    assert not control.actor_loss(actor).requires_grad and float(control.actor_loss(actor))==0
    assert all(p.grad is None for p in actor.parameters())
    control.head_step(actor)
    assert all(p.grad is None for p in actor.parameters()) and all(torch.equal(v,snapshot[k]) for k,v in actor.state_dict().items())
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
    resumed=FutureAuxiliary(actor,prepared,a.data,a.decoder,0,sha(a.source/'weights.json'),control.checkpoint())
    assert resumed.steps==1 and all(torch.equal(v,resumed.decoder.state_dict()[k]) for k,v in control.decoder.state_dict().items())
    candidate=FutureAuxiliary(actor,prepared,a.data,a.decoder,a.weight,sha(a.source/'weights.json'))
    candidate.actor_loss(actor).backward()
    grad=sum(float(p.grad.square().sum()) for p in actor.base.parameters() if p.grad is not None)
    assert grad>0 and all(p.grad is None for p in candidate.decoder.parameters())
    assert all(p.grad is None for n,p in actor.named_parameters() if not n.startswith('base.'))
    assert all(torch.equal(v,snapshot[k]) for k,v in actor.state_dict().items())
    control.verify();candidate.verify();resumed.verify()
    save(a.out,dict(state='complete',device='cuda',control_zero_actor_gradient=True,head_update_detaches_actor=True,
        candidate_trunk_gradient_squared_norm=grad,candidate_decoder_frozen=True,nontrunk_gradient_absent=True,
        constructor_and_head_rng_neutral=True,head_resume_roundtrip=True,actor_parameters_unchanged=True,
        rows=control.spec['rows'],enemy_labels=control.spec['enemy_labels'],weight=a.weight,script_sha256=sha(__file__),promotion=False))
    print(a.out)


if __name__=='__main__':main()

"""GPU architecture, likelihood, conditional gradients and export checks."""
import argparse,copy,pathlib
from process_combat_architecture_pool import read,sha,save
from prepare_combat_target_heads import build,forward
from combat_precision_head import migrate,probabilities,selected_means
from combat_spatial_aim import initialize,inputs,export
from ppo_combat import torch,log_prob,network
from ppo_recurrent import sequence

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args()
    assert torch.cuda.is_available();torch.manual_seed(20261010);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    reports=[]
    for path in sorted(a.models.glob('m[0-7]/weights.json')):
        precision=migrate(read(path));model=initialize(precision);actor=build(model,'actor');x=torch.randn(2,6,854,device='cuda')*.1
        with torch.no_grad():
            raw=forward(build(precision,'actor'),x);output=forward(actor,x);assert torch.equal(raw,output)
            assert torch.equal(forward(build(precision,'value'),x),forward(build(model,'value'),x))
        # A nonzero branch must export/load, preserve all old45 outputs, and
        # flow through both attention and GRU context collection paths.
        with torch.no_grad():actor.branch[-1].bias.copy_(torch.tensor([.2,-.1,-.2,.3],device='cuda'))
        result=copy.deepcopy(model);export(result,actor);reloaded=build(result,'actor')
        with torch.no_grad():
            updated=forward(actor,x);assert torch.equal(updated,forward(reloaded,x));assert torch.equal(raw[...,:45],updated[...,:45])
            if model.get('memory') or model.get('attention'):
                indices=torch.tensor([(b,t) for b in range(2) for t in range(6)],device='cuda');inverse=torch.arange(12,device='cuda')
                collected,_=sequence(actor,(x,indices,inverse,[[],[]]),32,False)
                assert torch.allclose(collected,updated.reshape(-1,81),atol=1e-6)
        features=torch.zeros(4,854,device='cuda');features[:,833]=1;features[1:,426]=1;features[2:,431]=1
        target=torch.tensor([0,1,2,1],device='cuda');mode=torch.tensor([0,1,0,1],device='cuda')
        raw=forward(actor,features[:,None,:]).reshape(4,81);z=selected_means(raw,target,mode).detach()+.1;std=torch.tensor(model['log_std'],device='cuda');attack=torch.zeros(4,device='cuda');vertical=torch.zeros(4,device='cuda',dtype=torch.long)
        lp,ent=probabilities(raw,std,z,attack,vertical,vertical,features,target,mode);assert torch.isfinite(lp).all() and torch.isfinite(ent).all()
        (-lp.mean()).backward();assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in actor.parameters())
        assert actor.branch[-1].bias.grad is not None and bool((actor.branch[-1].bias.grad!=0).any())
        if not model.get('memory') and not model.get('attention'):
            dispatch,_=log_prob(actor,std,features,z,attack,vertical,vertical,target,mode);assert torch.equal(dispatch,lp)
        reports.append(dict(parent_sha256=sha(path),architecture='gru' if model.get('memory') else 'attention' if model.get('attention') else 'mlp',migration_exact=True,export_reload_exact=True,old45_preserved=True,finite_likelihood_gradients=True))
    assert len(reports)==8
    sample=torch.zeros(3,854,device='cuda');sample[:,74]=torch.tensor([32,64,256],device='cuda')/512;sample[:,503]=.25;sample[:,504]=-.375;sample[:,505]=.5
    explicit=inputs(sample,torch.zeros(3,81,device='cuda'));assert explicit.shape==(3,8,119)
    assert bool((explicit[:-1,0,-5]>explicit[1:,0,-5]).all()) and bool((explicit[:-1,0,-4]>explicit[1:,0,-4]).all())
    save(a.out,dict(state='passed',device='cuda',models=reports,explicit_distance_features=True,scope='CUDA-only contract/likelihood/context/export checks across eight priors. No Go model numerical checks, native runtime acceptance or combat quality claim.'))
    print('CUDA spatial audit passed for eight MLP/GRU/attention priors',flush=True)

if __name__=='__main__':main()

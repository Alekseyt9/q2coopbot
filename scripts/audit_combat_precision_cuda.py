"""GPU-only migration, likelihood and gradient checks across actor families."""
import argparse,pathlib
from process_combat_architecture_pool import read,save,sha
from prepare_combat_target_heads import build,forward
from combat_precision_head import migrate,validate_precision_model,probabilities,selected_means,mode_distribution,physical_actions
from combat_target_head import distribution
from ppo_combat import torch,log_prob

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args()
    assert torch.cuda.is_available();torch.manual_seed(20261010)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    results=[]
    for path in sorted(a.models.glob('m[0-7]/weights.json')):
        parent=read(path);model=migrate(parent);validate_precision_model(model)
        x=torch.randn(2,6,854,device='cuda')*.1
        with torch.no_grad():
            for name in ('actor','value'):
                old=forward(build(parent,name),x);new=forward(build(model,name),x)
                assert torch.equal(old,new[...,:old.shape[-1]])
            raw=forward(build(model,'actor'),x).reshape(-1,81)
            for target in range(9):
                selected=torch.full((len(raw),),target,device='cuda',dtype=torch.long)
                assert bool((mode_distribution(raw,selected).probs.argmax(-1)==0).all())
        results.append(dict(parent=str(path),parent_sha256=sha(path),architecture='gru' if parent.get('memory') else 'attention' if parent.get('attention') else 'mlp',old_actor_value_exact=True,initial_deterministic_mode='coarse'))
    assert len(results)==8
    raw=torch.randn(4,81,device='cuda',requires_grad=True);features=torch.zeros(4,854,device='cuda');features[:,833]=1;features[1:,426]=1;features[2:,431]=1
    target=torch.tensor([0,1,2,1],device='cuda');mode=torch.tensor([0,1,0,1],device='cuda');std=torch.full((4,),-.5,device='cuda')
    z=selected_means(raw,target,mode).detach()+.1
    attack=torch.zeros(4,device='cuda');vertical=torch.zeros(4,device='cuda',dtype=torch.long);weapon=torch.zeros_like(vertical)
    lp,entropy=probabilities(raw,std,z,attack,vertical,weapon,features,target,mode)
    assert torch.isfinite(lp).all() and torch.isfinite(entropy).all()
    dispatch,_=log_prob(lambda _:raw,std,features,z,attack,vertical,weapon,target,mode);assert torch.equal(lp,dispatch)
    (-lp.mean()).backward();assert torch.isfinite(raw.grad).all()
    for row in range(4):
        allowed=torch.zeros(81,device='cuda',dtype=torch.bool);allowed[:2]=True;allowed[4:20]=True;allowed[20:29]=True
        pair=(45+2*int(target[row])) if int(mode[row]) else (2 if int(target[row])==0 else 29+2*(int(target[row])-1))
        allowed[pair:pair+2]=True;allowed[63+2*int(target[row]):65+2*int(target[row])]=True
        assert (raw.grad[row,~allowed]==0).all()
    physical=physical_actions(z,mode);assert (physical[mode==1,2:].abs()<=15).all()
    try:probabilities(raw,std,z,attack,vertical,weapon,features,torch.full_like(target,8),mode)
    except ValueError:pass
    else:raise AssertionError('Unavailable target accepted')
    save(a.out,dict(state='passed',device='cuda',gpu=torch.cuda.get_device_name(),models=results,conditional_gradient_routing=True,mlp_ppo_dispatch=True,unavailable_target_rejected=True,fine_angle_bound=15,scope='CUDA contract audit, not Go numerical parity or native combat quality.'))
    print('CUDA precision audit passed for eight MLP/GRU/attention parents',flush=True)

if __name__=='__main__':main()

"""CUDA regression: repairing a trained head must preserve frozen Adam states."""
import copy
from ppo_combat import torch,guarded_actor_step
from combat_full_joint_ppo import equal_tree


def main():
    assert torch.cuda.is_available()
    x=torch.nn.Parameter(torch.tensor([1.],device='cuda'));frozen=torch.nn.Parameter(torch.tensor([2.],device='cuda'))
    opt=torch.optim.Adam([x,frozen],lr=.01)
    # Prime moments in the opposite direction to today's gradient.
    x.grad=torch.tensor([-100.],device='cuda');frozen.grad=torch.tensor([3.],device='cuda');opt.step()
    old=copy.deepcopy(opt.state[frozen]);fv=frozen.detach().clone()
    opt.zero_grad(set_to_none=True)
    def objective():return x.square().sum(),torch.zeros((),device='cuda')
    loss,_=objective();loss.backward()
    result=guarded_actor_step([x,frozen],opt,objective,loss.detach(),.01,.01,momentum_parameters=[x])
    assert result['accepted'] and result['direction_fallback']
    assert torch.equal(frozen.detach(),fv) and equal_tree(old,opt.state[frozen])
    print(dict(state='passed',device='cuda',uphill_momentum_repaired=True,frozen_weights_and_adam_state_exact=True))


if __name__=='__main__':main()

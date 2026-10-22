"""Motor scope fallback isolation and explicit-reference failure on CUDA."""
import copy
import unittest
from ppo_combat import torch,nn,training_devices,guarded_actor_step
from combat_motor_residual import freeze_scope
from combat_sampled_motion_aim import correction


class MotorTests(unittest.TestCase):
    def test_fallback_preserves_frozen_momentum(self):
        training_devices()
        class Actor(nn.Module):
            def __init__(self):
                super().__init__();self.base=nn.Linear(2,2,device='cuda');self.motor_branch=nn.Linear(2,2,device='cuda')
            def forward(self,x):return self.base(x)+self.motor_branch(x)
        actor=Actor();std=nn.Parameter(torch.zeros(4,device='cuda'))
        opt=torch.optim.Adam([dict(params=[*actor.base.parameters(),std]),dict(params=list(actor.motor_branch.parameters()))],lr=.001)
        x=torch.ones(3,2,device='cuda');(actor(x).sum()+std.sum()).backward();opt.step()
        for state in opt.state.values():state['exp_avg'].fill_(100)
        before={p:p.detach().clone() for p in [*actor.base.parameters(),std]}
        states={p:copy.deepcopy(opt.state[p]) for p in before}
        project,_=freeze_scope(actor,std,opt)
        def objective():return -actor(x).sum(),torch.zeros((),device='cuda')
        loss,_=objective();opt.zero_grad();loss.backward()
        result=guarded_actor_step([*actor.parameters(),std],opt,objective,loss.detach(),.001,.005,project,group_lrs=[.001,.001])
        self.assertTrue(result['accepted']);self.assertTrue(result['direction_fallback'])
        for parameter,value in before.items():
            self.assertTrue(torch.equal(parameter,value))
            for key,state in states[parameter].items():self.assertTrue(torch.equal(state,opt.state[parameter][key]))

    def test_missing_reference_is_rejected(self):
        training_devices();module=nn.Linear(1,2,device='cuda');module.requires_motor_reference=True
        with self.assertRaisesRegex(AssertionError,'explicit reference'):
            correction(module,torch.zeros(1,1147,device='cuda'),torch.zeros(1,81,device='cuda'),
                       torch.zeros(1,2,device='cuda'),torch.ones(1,dtype=torch.long,device='cuda'),
                       torch.ones(1,dtype=torch.long,device='cuda'))


if __name__=='__main__':unittest.main()

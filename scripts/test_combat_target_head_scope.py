import copy
import unittest

from ppo_combat import torch, nn, training_devices, guarded_actor_step
from combat_target_head_scope import freeze_target_scope


class TargetScopeTests(unittest.TestCase):
    def test_head_kl_isolates_target_and_masked_branches(self):
        training_devices()
        from audit_combat_head_learning_cuda import components
        features=torch.zeros(3,854,device='cuda');features[:,833]=1;features[:,426]=1
        std=torch.zeros(4,device='cuda')
        for width in (45,81):
            old=torch.zeros(3,width,device='cuda');new=old.clone();new[:,21]=.5
            terms,total=components(old,new,std,std,features)
            self.assertGreater(float(terms['target'].mean()),0)
            for name in ('movement','aim','attack','vertical','weapon','aim_mode'):
                self.assertLess(float(terms[name].abs().max()),1e-12)
            # Unavailable target and weapon logits cannot change the policy.
            masked=old.clone();masked[:,28]=50;masked[:,19]=50
            terms,total=components(old,masked,std,std,features)
            self.assertLess(float(total.abs().max()),1e-12)

    def test_projection_preserves_frozen_outputs_and_momentum(self):
        training_devices()
        actor = nn.Sequential(nn.Linear(2,3,device='cuda'),nn.ReLU(),nn.Linear(3,45,device='cuda'))
        std = nn.Parameter(torch.zeros(4,device='cuda'))
        opt = torch.optim.Adam(list(actor.parameters())+[std],lr=.001)
        x = torch.ones(5,2,device='cuda')
        # Populate old momentum on every row before freezing.
        (actor(x).sum()+std.sum()).backward()
        opt.step()
        initial = actor(x).detach().clone()
        initial_std = std.detach().clone()
        frozen_state = copy.deepcopy(opt.state[actor[0].weight])
        old_moments = opt.state[actor[-1].weight]['exp_avg'].clone()
        project, count = freeze_target_scope(actor,std,opt)
        self.assertEqual(count,36)
        opt.zero_grad()
        actor(x)[:,20:29].sum().backward()
        opt.step()
        project()
        current = actor(x).detach()
        outside = torch.ones(45,dtype=torch.bool,device='cuda');outside[20:29]=False
        self.assertTrue(torch.equal(initial[:,outside],current[:,outside]))
        self.assertFalse(torch.equal(initial[:,20:29],current[:,20:29]))
        self.assertTrue(torch.equal(initial_std,std))
        for name,value in frozen_state.items():
            self.assertTrue(torch.equal(value,opt.state[actor[0].weight][name]))
        self.assertTrue(torch.equal(old_moments[outside],opt.state[actor[-1].weight]['exp_avg'][outside]))
        # Rejected line-search trials must restore weights and all optimizer state.
        before = [p.detach().clone() for p in list(actor.parameters())+[std]]
        state = copy.deepcopy(opt.state_dict())
        def objective():
            return actor(x)[:,20:29].sum(),torch.ones((),device='cuda')
        loss,_ = objective()
        opt.zero_grad();loss.backward()
        trial = guarded_actor_step(list(actor.parameters())+[std],opt,objective,loss.detach(),.001,.005,project)
        self.assertFalse(trial['accepted'])
        for p,old in zip(list(actor.parameters())+[std],before):
            self.assertTrue(torch.equal(p,old))
        for index,old in state['state'].items():
            for name,value in old.items():
                self.assertTrue(torch.equal(value,opt.state_dict()['state'][index][name]))


if __name__=='__main__':unittest.main()

"""CUDA regression checks for unequal Adam rates and guarded rollback."""
import copy,unittest
from ppo_combat import torch,nn,guarded_actor_step
from combat_weapon_state_residual import optimizer


class WeaponOptimizerCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; CPU fallback forbidden')

    def pair(self):
        a=nn.Parameter(torch.tensor(1.,device='cuda'));b=nn.Parameter(torch.tensor(1.,device='cuda'))
        opt=torch.optim.Adam([dict(params=[a],lr=.01),dict(params=[b],lr=.003)],lr=.01)
        return a,b,opt

    def test_accepted_keeps_different_rates(self):
        a,b,opt=self.pair()
        objective=lambda:((a+b).square(),(a+b)*0)
        loss,_=objective();loss.backward()
        trial=guarded_actor_step([a,b],opt,objective,loss.detach(),.01,.1,group_lrs=[.01,.003])
        self.assertTrue(trial['accepted'])
        self.assertEqual([g['lr'] for g in opt.param_groups],[.01,.003])
        self.assertAlmostEqual(float((1-a).detach()),.01,places=6);self.assertAlmostEqual(float((1-b).detach()),.003,places=6)

    def test_rejected_restores_parameters_moments_and_rates(self):
        a,b,opt=self.pair();(a+b).square().backward();opt.step();opt.zero_grad()
        objective=lambda:((a+b).square(),(a+b)*0+1)
        loss,_=objective();loss.backward()
        before=[p.detach().clone() for p in (a,b)];state=copy.deepcopy(opt.state_dict())
        trial=guarded_actor_step([a,b],opt,objective,loss.detach(),.01,.1,group_lrs=[.01,.003])
        self.assertFalse(trial['accepted'])
        self.assertTrue(all(torch.equal(p,v) for p,v in zip((a,b),before)))
        after=opt.state_dict();self.assertEqual(state['param_groups'],after['param_groups'])
        for key,fields in state['state'].items():
            self.assertTrue(all(torch.equal(value,after['state'][key][name]) for name,value in fields.items()))

    def test_old_parameter_ids_and_std_stay_in_first_group(self):
        actor=nn.Module();actor.base=nn.Linear(2,1,device='cuda');std=nn.Parameter(torch.zeros(4,device='cuda'))
        legacy=optimizer(actor,std,.001,.003)
        actor.weapon_state_branch=nn.Linear(2,12,device='cuda')
        migrated=optimizer(actor,std,.001,.003)
        self.assertEqual(legacy.state_dict()['param_groups'][0]['params'],migrated.state_dict()['param_groups'][0]['params'])
        self.assertIs(migrated.param_groups[0]['params'][-1],std)
        self.assertEqual(set(migrated.param_groups[0]['params'])&set(migrated.param_groups[1]['params']),set())

    def test_accepted_backtrack_scales_both_rates(self):
        a,b,opt=self.pair();origin=[p.detach().clone() for p in (a,b)]
        objective=lambda:((a+b).square(),(a-origin[0]).square()+(b-origin[1]).square())
        loss,_=objective();loss.backward()
        trial=guarded_actor_step([a,b],opt,objective,loss.detach(),.4,.05,group_lrs=[.4,.1])
        self.assertTrue(trial['accepted']);self.assertEqual(trial['retry'],1)
        self.assertAlmostEqual(float((origin[0]-a).detach()),.2,places=6)
        self.assertAlmostEqual(float((origin[1]-b).detach()),.05,places=6)
        self.assertEqual([g['lr'] for g in opt.param_groups],[.4,.1])

    def test_momentum_fallback_preserves_rate_ratio(self):
        a,b,opt=self.pair();(-40*(a+b)).backward();opt.step();opt.zero_grad()
        objective=lambda:((a+b).square(),(a+b)*0)
        loss,_=objective();loss.backward()
        trial=guarded_actor_step([a,b],opt,objective,loss.detach(),.01,.1,group_lrs=[.01,.003])
        self.assertTrue(trial['accepted']);self.assertTrue(trial['direction_fallback'])
        self.assertEqual([g['lr'] for g in opt.param_groups],[.01,.003])
        self.assertTrue(all(float(s['step'])==2 for s in opt.state.values()))

if __name__=='__main__':unittest.main()

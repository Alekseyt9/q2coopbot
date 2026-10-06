import unittest, tempfile, pathlib, copy
from ppo_combat import advantages, restore_checkpoint, validate_objective, restore_optimizer, guarded_actor_step, retention_schedule, anchor_kl, training_devices, torch, nn, sha

def row(seed,index,reward,value,next_value,terminal=False,truncated=False):
    return {'seed':seed,'index':index,'frame':index,'next_frame':index+1,'reward':reward,'sample':{'value':value},'next_value':next_value,'terminal':terminal,'truncated':truncated}

class GAETest(unittest.TestCase):
    def test_terminal_no_bootstrap(self):
        a,r=advantages([row(1,1,1,2,999,True)],.9,.8)
        self.assertEqual(a,[-1]);self.assertEqual(r,[1])
    def test_truncation_bootstrap_but_no_cross_episode(self):
        a,r=advantages([row(1,1,1,2,3,truncated=True),row(2,2,100,0,0,True)],.9,.8)
        self.assertAlmostEqual(a[0],1.7);self.assertAlmostEqual(r[0],3.7)
    def test_gap_cuts_recurrence(self):
        a,_=advantages([row(1,1,1,0,0),row(1,3,100,0,0,True)],.9,.8)
        self.assertEqual(a[0],1)
    def test_contiguous_returns(self):
        a,_=advantages([row(1,1,1,0,0),row(1,2,2,0,0,True)],.9,.8)
        self.assertAlmostEqual(a[0],2.44)

class ResumeTest(unittest.TestCase):
    def test_constant_retention_keeps_weight_and_pins_mode_on_resume(self):
        spec,weight=retention_schedule('sha',1.,4,53,mode='constant')
        self.assertEqual(spec['mode'],'constant');self.assertEqual(weight,1.)
        for elapsed in (1,3,4,100):
            resumed,weight=retention_schedule('sha',1.,4,53+elapsed,spec,'constant')
            self.assertEqual(resumed,spec);self.assertEqual(weight,1.)
        with self.assertRaisesRegex(ValueError,'changed on resume'):retention_schedule('sha',1.,4,54,spec,'linear')
        legacy,_=retention_schedule('sha',1.,4,53)
        self.assertNotIn('mode',legacy)
        with self.assertRaisesRegex(ValueError,'changed on resume'):retention_schedule('sha',1.,4,54,legacy,'constant')

    def test_invalid_retention_mode_or_missing_anchor_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unknown'):retention_schedule('sha',1.,4,53,mode='other')
        with self.assertRaisesRegex(ValueError,'requires an anchor'):retention_schedule(None,1.,4,53,mode='constant')

    def direction_case(self):
        p=nn.Parameter(torch.tensor([1.],device=training_devices()[0]));opt=torch.optim.Adam([p],lr=.01)
        p.grad=-10*torch.ones_like(p);opt.step()
        with torch.no_grad():p.fill_(1.)
        opt.zero_grad();loss=p.square().sum();loss.backward()
        return p,opt,loss.detach(),copy.deepcopy(opt.state_dict())

    def test_uphill_momentum_repaired_without_resetting_clock_or_second_moment(self):
        p,opt,loss,snapshot=self.direction_case()
        result=guarded_actor_step([p],opt,lambda:(p.square().sum(),(p-1).square().sum()),loss,.01,.01)
        self.assertTrue(result['accepted']);self.assertTrue(result['direction_fallback'])
        self.assertGreater(result['initial_grad_dot_delta'],0);self.assertLess(result['accepted_grad_dot_delta'],0)
        self.assertEqual(int(opt.state[p]['step']),2);self.assertLess(float(p.detach()),1.)
        expected=.999*snapshot['state'][0]['exp_avg_sq']+.001*p.grad.square()
        self.assertTrue(torch.allclose(opt.state[p]['exp_avg_sq'],expected))
        self.assertTrue(torch.allclose(opt.state[p]['exp_avg'],.1*p.grad))
        self.assertEqual(opt.param_groups[0]['lr'],.01)
        self.assertEqual(int(snapshot['state'][0]['step']),1)

    def test_rejected_fallback_restores_parameters_moments_clock_and_lr(self):
        p,opt,loss,snapshot=self.direction_case();before=p.detach().clone()
        result=guarded_actor_step([p],opt,lambda:(p.square().sum(),torch.ones((),device=p.device)),loss,.01,0.)
        self.assertFalse(result['accepted']);self.assertTrue(result['direction_fallback'])
        self.assertTrue(torch.equal(p,before));self.assertEqual(opt.state_dict()['param_groups'],snapshot['param_groups'])
        for k,v in opt.state[p].items():self.assertTrue(torch.equal(v,snapshot['state'][0][k]))

    def test_downhill_step_preserves_normal_adam_momentum(self):
        p=nn.Parameter(torch.tensor([1.],device=training_devices()[0]));opt=torch.optim.Adam([p],lr=.01)
        loss=p.square().sum();loss.backward()
        result=guarded_actor_step([p],opt,lambda:(p.square().sum(),(p-1).square().sum()),loss.detach(),.01,.01)
        self.assertTrue(result['accepted']);self.assertFalse(result['direction_fallback'])
        self.assertEqual(int(opt.state[p]['step']),1)

    def test_retention_covers_every_head_and_has_cuda_gradients(self):
        device=training_devices()[0];teacher=torch.zeros(3,8,device=device);std=torch.zeros(4,device=device)
        self.assertEqual(float(anchor_kl(teacher,std,teacher,std)),0.)
        for column in (0,2,4,5):
            raw=teacher.clone();raw[:,column]=1;raw.requires_grad_()
            loss=anchor_kl(raw,std,teacher,std);self.assertGreater(float(loss.detach()),0.)
            loss.backward();self.assertGreater(float(raw.grad[:,column].abs().sum()),0.)
        narrow=std.clone();narrow[0]=-1
        self.assertGreater(float(anchor_kl(teacher,narrow,teacher,std)),0.)

    def test_retention_anneals_and_resume_pins_schedule(self):
        spec,weight=retention_schedule('sha',1.,4,53)
        self.assertEqual(weight,1.)
        for i,expected in ((1,2/3),(2,1/3),(3,0.),(4,0.)):
            resumed,weight=retention_schedule('sha',1.,4,53+i,spec)
            self.assertEqual(resumed,spec);self.assertAlmostEqual(weight,expected)
        for args in ((None,1.,4,54),('other',1.,4,54),('sha',2.,4,54),('sha',1.,5,54)):
            with self.assertRaises(ValueError):retention_schedule(*args,previous=spec)
        with self.assertRaises(ValueError):retention_schedule('sha',float('nan'),4,53)

    def test_backtracking_snapshot_survives_rejected_cuda_trials(self):
        parameter=nn.Parameter(torch.tensor([1.],device=training_devices()[0]))
        optimizer=torch.optim.Adam([parameter],lr=.01)
        parameter.grad=torch.ones_like(parameter);optimizer.step()
        baseline=copy.deepcopy(optimizer.state_dict())
        reference=copy.deepcopy(baseline)
        for gradient in (2.,3.,4.):
            restore_optimizer(optimizer,baseline)
            parameter.grad=torch.full_like(parameter,gradient);optimizer.step()
            self.assertEqual(int(optimizer.state[parameter]['step']),2)
            for key,value in baseline['state'][0].items():
                self.assertTrue(torch.equal(value,reference['state'][0][key]))
            self.assertEqual(int(baseline['state'][0]['step']),1)
        restore_optimizer(optimizer,baseline)
        for key,value in optimizer.state[parameter].items():
            self.assertTrue(torch.equal(value,baseline['state'][0][key]))

    def test_objective_pin_rejects_wrong_or_unpinned_kill_reward(self):
        validate_objective({}, {})
        validate_objective({'objective_reward_sha256':'abc'}, {'reward_config_sha256':'ABC','reward_version':'combat_reward_v2'})
        with self.assertRaisesRegex(AssertionError,'must be pinned'):validate_objective({}, {'reward_version':'combat_reward_v2'})
        validate_objective({'objective_reward_sha256':'abc','gamma':.99},{'reward_version':'combat_reward_v3','reward_config_sha256':'abc','aim_gamma':.99})
        validate_objective({'objective_reward_sha256':'abc','gamma':.99},{'reward_version':'combat_reward_v4','reward_config_sha256':'abc','aim_gamma':.99})
        with self.assertRaisesRegex(AssertionError,'must be pinned'):validate_objective({}, {'reward_version':'combat_reward_v4'})
        with self.assertRaisesRegex(AssertionError,'discount differs'):
            validate_objective({'objective_reward_sha256':'abc','gamma':.98},{'reward_version':'combat_reward_v3','reward_config_sha256':'abc','aim_gamma':.99})
        with self.assertRaisesRegex(AssertionError,'reward differs'):validate_objective({'objective_reward_sha256':'abc'}, {'reward_config_sha256':'other'})
    def test_resume_and_reject_reused_rollout_or_other_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);model=root/'weights.json';model.write_text('{}')
            device=training_devices()[0];actor=nn.Linear(2,2).to(device);value=nn.Linear(2,1).to(device);std=nn.Parameter(torch.zeros(4,device=device))
            opt=torch.optim.Adam(list(actor.parameters())+[std],lr=.001)
            def step(module,optimizer):
                optimizer.zero_grad();(module(torch.ones(1,2,device=device)).sum()).backward();optimizer.step()
            step(actor,opt)
            ck={'version':'combat_ppo_checkpoint_v2','weights_sha256':sha(model),'consumed_rollouts':['old'],'updates_completed':2,'total_actor_steps':20,'actor':copy.deepcopy(actor.state_dict()),'value':value.state_dict(),'log_std':std.detach(),'config':{},'actor_optimizer':copy.deepcopy(opt.state_dict())}
            path=root/'checkpoint.pt';torch.save(ck,path)
            loaded,consumed,updates,steps=restore_checkpoint(path,model,{},actor,value,std,'fresh')
            self.assertEqual((consumed,updates,steps),(['old'],2,20))
            with self.assertRaisesRegex(AssertionError,'already consumed'):restore_checkpoint(path,model,{},actor,value,std,'old')
            clone=copy.deepcopy(actor);clone_std=nn.Parameter(std.detach().clone());resumed=torch.optim.Adam(list(clone.parameters())+[clone_std],lr=.001);resumed.load_state_dict(loaded['actor_optimizer'])
            step(actor,opt);step(clone,resumed)
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(actor.parameters(),clone.parameters())))
            model.write_text('{"other":true}')
            with self.assertRaisesRegex(AssertionError,'other weights'):restore_checkpoint(path,model,{},clone,value,clone_std,'fresh')

if __name__=='__main__':unittest.main()

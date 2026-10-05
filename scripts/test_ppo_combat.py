import unittest, tempfile, pathlib, copy
from ppo_combat import advantages, restore_checkpoint, validate_objective, torch, nn, sha

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
            actor=nn.Linear(2,2);value=nn.Linear(2,1);std=nn.Parameter(torch.zeros(4))
            opt=torch.optim.Adam(list(actor.parameters())+[std],lr=.001)
            def step(module,optimizer):
                optimizer.zero_grad();(module(torch.ones(1,2)).sum()).backward();optimizer.step()
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

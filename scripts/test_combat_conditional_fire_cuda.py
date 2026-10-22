"""CUDA-only standalone contract, zero identity and gradient ownership tests."""
import copy
import json
import pathlib
import unittest
from ppo_combat import torch, nn, training_devices
from combat_conditional_fire import initialize, build, inputs, logits, validate, INPUT_WIDTH


def context(n=2):
    x = torch.zeros(n, 1147, device='cuda')
    x[:, 833] = 1
    x[:, 426] = 1
    x[:, 73:85] = 2
    x[:, 466:506] = 3
    return x, torch.arange(n, device='cuda') % 2, torch.ones(n, dtype=torch.long, device='cuda'), torch.zeros(n, 4, device='cuda'), torch.zeros(n, dtype=torch.long, device='cuda'), torch.zeros(n, dtype=torch.long, device='cuda')


class ConditionalFireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        training_devices()

    def test_input_layout_and_no_target_mask(self):
        x, target, mode, z, vertical, weapon = context()
        z[:, 2] = .5
        value = inputs(x, target, mode, z, vertical, weapon)
        self.assertEqual(value.shape, (2, INPUT_WIDTH))
        self.assertTrue(torch.equal(value[0, 99:151], torch.zeros(52, device='cuda')))
        self.assertTrue(torch.equal(value[1, 99:111], x[1, 73:85]))
        self.assertTrue(torch.equal(value[1, 111:151], x[1, 466:506]))
        self.assertEqual(float(value[0, 151]), 1)
        self.assertEqual(float(value[1, 152]), 1)
        self.assertEqual(float(value[1, 161]), 1)
        self.assertLess(float((value[:, 164] - z[:, 2].tanh() / 12).abs().max()), 1e-7)

    def test_zero_identity_and_rng_preservation(self):
        model = dict(feature_version='combat_features_v10', aim_mode_head='combat_target_coarse_fine_aim_v1', spatial_aim=True, sampled_motion_aim=True)
        cpu_rng = torch.get_rng_state().clone(); cuda_rng = torch.cuda.get_rng_state().clone()
        migrated = initialize(model)
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))
        self.assertNotIn('conditional_fire', model)
        module = build(migrated)
        self.assertEqual(sum(p.numel() for p in module.parameters()), 6369)
        x, target, mode, z, vertical, weapon = context()
        raw = torch.randn(2, 81, device='cuda')
        old = logits(None, x, raw, target, mode, z, vertical, weapon)
        new = logits(module, x, raw, target, mode, z, vertical, weapon)
        self.assertTrue(torch.equal(old, new))
        attack = torch.tensor([0., 1.], device='cuda')
        self.assertTrue(torch.equal(torch.distributions.Bernoulli(logits=old).log_prob(attack), torch.distributions.Bernoulli(logits=new).log_prob(attack)))

    def test_context_sensitivity_and_detached_action_gradient(self):
        x, target, mode, z, vertical, weapon = context()
        x.requires_grad_(); z.requires_grad_()
        raw = torch.zeros(2, 81, device='cuda', requires_grad=True)
        module = nn.Linear(INPUT_WIDTH, 1, bias=False, device='cuda')
        with torch.no_grad():
            module.weight.zero_(); module.weight[0, 151] = 2; module.weight[0, 152] = -1; module.weight[0, 164] = 1
        output = logits(module, x, raw, target, mode, z, vertical, weapon)
        self.assertTrue(torch.equal(output, torch.tensor([2., -1.], device='cuda')))
        changed = z.detach().clone(); changed[:, 2] = .5
        self.assertTrue(bool((logits(module, x, raw, target, mode, changed, vertical, weapon) > output).all()))
        output.sum().backward()
        self.assertIsNone(x.grad); self.assertIsNone(z.grad)
        self.assertIsNotNone(module.weight.grad)
        self.assertGreater(float(module.weight.grad.abs().sum()), 0)
        self.assertTrue(torch.equal(raw.grad[:, 4], torch.ones(2, device='cuda')))
        self.assertEqual(float(raw.grad[:, :4].abs().sum() + raw.grad[:, 5:].abs().sum()), 0)

    def test_rejects_missing_invalid_and_unavailable_context(self):
        values = context()
        for index, replacement in [(1, None), (1, torch.full((2,), 9, dtype=torch.long, device='cuda')), (2, torch.full((2,), 2, dtype=torch.long, device='cuda')), (5, torch.ones(2, dtype=torch.long, device='cuda'))]:
            bad = list(values); bad[index] = replacement
            with self.assertRaises((AssertionError, ValueError)):
                inputs(*bad)
        bad = list(values); bad[0] = bad[0].clone(); bad[0][:, 426] = 0
        with self.assertRaises(AssertionError):
            inputs(*bad)
        bad = list(values); bad[3] = torch.full((2, 4), float('nan'), device='cuda')
        with self.assertRaises(AssertionError):
            inputs(*bad)

    def test_parameter_contract(self):
        model = initialize(dict(feature_version='combat_features_v10', aim_mode_head='combat_target_coarse_fine_aim_v1', spatial_aim=True, sampled_motion_aim=True))
        bad = copy.deepcopy(model); bad['conditional_fire']['layers'][0]['weight'][0][0] = float('nan')
        with self.assertRaises(AssertionError):
            validate(bad)
        bad = copy.deepcopy(model); bad['conditional_fire']['version'] = 'unknown'
        with self.assertRaises(AssertionError):
            validate(bad)

    def test_joint_likelihood_and_export_on_source_actor(self):
        from combat_attention import CausalAttention
        from combat_spatial_aim import wrap, export
        from combat_precision_head import probabilities, selected_means
        root = pathlib.Path(__file__).resolve().parents[1]
        source = json.loads((root / 'workspace/artifacts/combat-motor-feedback-v1-20261022/round-2-full-update/weights.json').read_text())
        migrated = initialize(source)
        def actor(model):
            spec = model['attention']
            base = CausalAttention(model['actor'], spec['actor'], spec['heads'], spec['window'])
            return wrap(base, model).to('cuda')
        before, after = actor(source), actor(migrated)
        x, target, mode, z, vertical, weapon = context()
        std = torch.tensor(source['log_std'], device='cuda')
        attack = torch.tensor([0., 1.], device='cuda')
        oldraw, newraw = before.single(x), after.single(x)
        self.assertTrue(torch.equal(oldraw, newraw))
        # A plausible recorded action avoids subtracting two enormous negative
        # joint log probabilities when checking a small Bernoulli-only delta.
        z = selected_means(oldraw, target, mode, x, oldraw[:, :2].detach(),
            before.sampled_motion_aim_branch, vertical, before.motor_branch).detach()
        def joint(a, raw, include_fire=True):
            return probabilities(raw, std, z, attack, vertical, weapon, x, target, mode,
                a.sampled_motion_aim_branch, a.motor_branch,
                a.conditional_fire_branch if include_fire else None)
        oldlp, oldentropy = joint(before, oldraw)
        newlp, newentropy = joint(after, newraw)
        self.assertTrue(torch.equal(oldlp, newlp))
        self.assertTrue(torch.equal(oldentropy, newentropy))
        with self.assertRaisesRegex(ValueError, 'Conditional fire branch'):
            joint(after, newraw, False)
        from combat_weapon_state_residual import optimizer
        oldopt=optimizer(before,nn.Parameter(std.clone()),.0003,.0003)
        newopt=optimizer(after,nn.Parameter(std.clone()),.0003,.0003)
        self.assertEqual(len(oldopt.param_groups),3)
        self.assertEqual(len(newopt.param_groups),4)
        self.assertEqual(newopt.param_groups[-1]['name'],'conditional_fire')
        self.assertEqual([g['params'] for g in oldopt.state_dict()['param_groups']],
            [g['params'] for g in newopt.state_dict()['param_groups'][:-1]])
        from combat_sequence_retention import policy_kl
        with self.assertRaisesRegex(AssertionError, 'conditional retention contract'):
            policy_kl(newraw, std, oldraw, std, x, after.sampled_motion_aim_branch)
        exported = copy.deepcopy(migrated); export(exported, after)
        self.assertEqual(exported, migrated)
        with torch.no_grad():
            after.conditional_fire_branch[-1].bias.fill_(.7)
        changed, _ = joint(after, newraw)
        fire = torch.distributions.Bernoulli
        expected = fire(logits=newraw[:, 4]+.7).log_prob(attack)-fire(logits=newraw[:, 4]).log_prob(attack)
        self.assertTrue(torch.allclose(changed-newlp, expected, atol=2e-6, rtol=0),
            (newlp.detach().tolist(), changed.detach().tolist(), expected.detach().tolist()))
        (-changed.mean()).backward()
        self.assertGreater(float(after.conditional_fire_branch[-1].bias.grad.abs().sum()), 0)

    def test_conditional_retention_on_fixed_complete_action(self):
        from combat_sequence_retention import policy_kl
        x,target,mode,z,vertical,weapon=context()
        raw=torch.zeros(2,81,device='cuda',requires_grad=True)
        std=torch.zeros(4,device='cuda')
        teacher=nn.Linear(INPUT_WIDTH,1,device='cuda')
        with torch.no_grad():teacher.weight.zero_();teacher.bias.zero_()
        current=copy.deepcopy(teacher)
        for p in teacher.parameters():p.requires_grad_(False)
        context_values=(target,mode,z,vertical,weapon)
        def kl():
            return policy_kl(raw,std,raw.detach(),std,x,fire_branch=current,
                teacher_fire_branch=teacher,fire_context=context_values)
        self.assertTrue(torch.equal(kl(),torch.zeros(2,device='cuda',dtype=torch.float64)))
        with torch.no_grad():current.weight[0,151]=.4
        observed=kl()
        d=torch.distributions
        expected=d.kl_divergence(d.Bernoulli(logits=torch.zeros(2,device='cuda',dtype=torch.float64)),
            d.Bernoulli(logits=torch.tensor([.4,0.],device='cuda',dtype=torch.float64)))
        self.assertTrue(torch.allclose(observed,expected,atol=1e-8,rtol=0))
        observed.sum().backward()
        self.assertGreater(float(current.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        with self.assertRaises(AssertionError):
            policy_kl(raw,std,raw.detach(),std,x,fire_branch=current,teacher_fire_branch=teacher)

    def test_fire_only_scope_preserves_frozen_adam_on_fallback(self):
        from combat_conditional_fire import freeze_scope
        from ppo_combat import guarded_actor_step
        class Actor(nn.Module):
            def __init__(self):
                super().__init__();self.base=nn.Linear(2,1,device='cuda');self.conditional_fire_branch=nn.Linear(2,1,device='cuda')
            def forward(self,x):return self.base(x)+self.conditional_fire_branch(x)
        actor=Actor();std=nn.Parameter(torch.zeros(4,device='cuda'))
        opt=torch.optim.Adam([*actor.parameters(),std],lr=.001)
        x=torch.ones(3,2,device='cuda');(actor(x).sum()+std.sum()).backward();opt.step()
        for state in opt.state.values():state['exp_avg'].fill_(100)
        before={p:p.detach().clone() for p in [*actor.base.parameters(),std]}
        states={p:copy.deepcopy(opt.state[p]) for p in before}
        project,count=freeze_scope(actor,std,opt);self.assertEqual(count,3)
        def objective():return -actor(x).sum(),torch.zeros((),device='cuda')
        loss,_=objective();opt.zero_grad();loss.backward()
        result=guarded_actor_step([*actor.parameters(),std],opt,objective,loss.detach(),.001,.005,project)
        self.assertTrue(result['accepted']);self.assertTrue(result['direction_fallback'])
        for p,value in before.items():
            self.assertTrue(torch.equal(p,value))
            for key,state in states[p].items():self.assertTrue(torch.equal(state,opt.state[p][key]))


if __name__ == '__main__':
    unittest.main()

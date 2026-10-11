"""Input/Adam transfer tests, CUDA only; no CPU numerical fallback."""
import copy
import unittest

from combat_input_checkpoint_expansion_cuda import expand_checkpoint_inputs, torch


class InputCheckpointExpansionCUDA(unittest.TestCase):
    def setUp(self):
        assert torch.cuda.is_available()
        self.cp=dict(version='combat_architecture_checkpoint_v1',retention_weights=[0.,0.],
                     sequence_retention=None,log_std=torch.zeros(4,device='cuda'),
                     consumed_rollouts=['corpus'],updates_completed=9,total_actor_steps=85,
                     rng=torch.arange(8,device='cuda',dtype=torch.uint8),
                     cuda_rng=[torch.arange(8,device='cuda',dtype=torch.uint8)])
        for role in ('actor','value'):
            self.cp[role]={'encoder.weight':torch.randn(3,5,device='cuda'),
                           'encoder.bias':torch.randn(3,device='cuda')}
            shapes=[(3,5),(3,)] + ([(4,)] if role=='actor' else [])
            self.cp[role+'_optimizer']=dict(param_groups=[dict(params=list(range(len(shapes))),lr=.003)],
                state={i:dict(step=torch.tensor(17.,device='cuda'),
                              exp_avg=torch.randn(shape,device='cuda'),
                              exp_avg_sq=torch.rand(shape,device='cuda')) for i,shape in enumerate(shapes)})
        self.names={r:list(self.cp[r]) for r in ('actor','value')}

    def test_parameters_moments_and_history_preserved(self):
        result,changes=expand_checkpoint_inputs(self.cp,self.names,5,8)
        self.assertEqual(len(changes),2)
        for role in ('actor','value'):
            old,new=self.cp[role]['encoder.weight'],result[role]['encoder.weight']
            self.assertTrue(torch.equal(old,new[:,:5]))
            self.assertEqual(int(torch.count_nonzero(new[:,5:])),0)
            self.assertTrue(torch.equal(self.cp[role]['encoder.bias'],result[role]['encoder.bias']))
            for index,values in self.cp[role+'_optimizer']['state'].items():
                for field,old in values.items():
                    new=result[role+'_optimizer']['state'][index][field]
                    if index==0 and field!='step':
                        self.assertTrue(torch.equal(old,new[:,:5]))
                        self.assertEqual(int(torch.count_nonzero(new[:,5:])),0)
                    else:self.assertTrue(torch.equal(old,new))
            self.assertEqual(tuple(self.cp[role]['encoder.weight'].shape),(3,5))
        for field in ('consumed_rollouts','updates_completed','total_actor_steps'):
            self.assertEqual(result[field],self.cp[field])
        self.assertTrue(torch.equal(result['rng'],self.cp['rng']))
        self.assertTrue(torch.equal(result['cuda_rng'][0],self.cp['cuda_rng'][0]))

    def test_active_retention_requires_explicit_migration(self):
        for field,value in (('sequence_retention',dict(version='v2')),('retention_weights',[1.,0.])):
            cp=copy.deepcopy(self.cp);cp[field]=value
            with self.assertRaises(ValueError):expand_checkpoint_inputs(cp,self.names,5,8)

    def test_bad_parameter_order_and_moment_shape_rejected(self):
        names=copy.deepcopy(self.names);names['actor'].reverse()
        with self.assertRaises(AssertionError):expand_checkpoint_inputs(self.cp,names,5,8)
        cp=copy.deepcopy(self.cp);cp['actor_optimizer']['state'][0]['exp_avg']=torch.zeros(3,4,device='cuda')
        with self.assertRaises(AssertionError):expand_checkpoint_inputs(cp,self.names,5,8)

    def test_ambiguous_input_matrices_rejected(self):
        cp=copy.deepcopy(self.cp);cp['actor']['other.weight']=torch.zeros(2,5,device='cuda')
        names=copy.deepcopy(self.names);names['actor']=list(cp['actor'])
        with self.assertRaises(AssertionError):expand_checkpoint_inputs(cp,names,5,8)


if __name__=='__main__':unittest.main()

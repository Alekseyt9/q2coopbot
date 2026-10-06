import copy,unittest
from ppo_recurrent import torch,nn,initialize,Recurrent,prepare_context,sequence

def base():
    def layers(out):
        return [{'weight':[[.01*(i+1)]*n for i in range(m)],'bias':[.02]*m} for n,m in [(2,4),(4,4),(4,out)]]
    return dict(kind='combat_ppo_v1',feature_version='test',actor=layers(8),value=layers(1),log_std=[-2.]*4,sampling_seed=0,deterministic=False)

class RecurrentTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(),'CUDA is mandatory for gradient tests');torch.manual_seed(4)
    def test_zero_residual_preserves_base_and_memory_can_receive_gradient(self):
        m=initialize(base(),4);net=Recurrent(m['actor'],m['memory']['actor']).cuda();x=torch.rand(2,6,2,device='cuda')
        y,_,_=net(x);expected=net.head(net.encoder(x));self.assertTrue(torch.equal(y,expected))
        with torch.no_grad():net.output.weight.fill_(.1)
        y,_,_=net(x);y.square().mean().backward()
        self.assertGreater(float(net.gru.weight_hh_l0.grad.abs().sum()),0)
    def test_unselected_context_affects_future_and_reset_is_separate(self):
        m=initialize(base(),4);net=Recurrent(m['actor'],m['memory']['actor']).cuda()
        with torch.no_grad():net.output.weight.fill_(.2)
        context=[];rows=[]
        for i in range(4):
            c=dict(seed=3,index=i,frame=100+i,features=[1.+i,2.],memory=dict(actor=[0.]*4,value=[0.]*4,reset=i==0));context.append(c)
            if i in (0,3):rows.append(dict(seed=3,index=i,frame=c['frame'],features=c['features'],sample=dict(memory=c['memory'])))
        p=prepare_context(context,rows,'cuda');y,_=sequence(net,p,2)
        changed=copy.deepcopy(context);changed[1]['features']=[50.,50.]
        q=prepare_context(changed,rows,'cuda');z,_=sequence(net,q,2)
        self.assertGreater(float((y[1]-z[1]).detach().abs().max()),1e-6)
        altered=copy.deepcopy(context);altered[2]['frame']+=4
        with self.assertRaises(AssertionError):prepare_context(altered,rows,'cuda')
        y.sum().backward();self.assertIsNotNone(net.gru.weight_ih_l0.grad)
    def test_export_roundtrip_and_permutation(self):
        m=initialize(base(),4);net=Recurrent(m['actor'],m['memory']['actor']).cuda()
        with torch.no_grad():net.output.weight.fill_(.1)
        b,c=net.export();other=Recurrent(b,c).cuda();x=torch.rand(2,5,2,device='cuda')
        self.assertTrue(torch.equal(net(x)[0],other(x)[0]))

if __name__=='__main__':unittest.main()

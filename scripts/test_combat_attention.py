import unittest
from ppo_combat import torch
from combat_attention import CausalAttention,EntityAttention,initialize_attention,entity_tokens

def model():
    def layers(out):return [{'weight':[[.001]*n for _ in range(m)],'bias':[.01]*m} for n,m in ((810,8),(8,8),(8,out))]
    return {'kind':'combat_ppo_v1','feature_version':'combat_features_v4','actor':layers(8),'value':layers(1),'log_std':[-2.]*4,'sampling_seed':0,'deterministic':False}

class AttentionTests(unittest.TestCase):
    def setUp(self):self.assertTrue(torch.cuda.is_available());torch.manual_seed(8)
    def test_causal_mask_window_and_zero_start(self):
        m=initialize_attention(model(),heads=4,window=3);net=CausalAttention(m['actor'],m['attention']['actor'],4,3).cuda()
        x=torch.rand(1,8,810,device='cuda');y,_=net(x)
        self.assertTrue(torch.equal(y,net.head(net.encoder(x))))
        with torch.no_grad():net.residual.weight.fill_(.1)
        y,_=net(x);changed=x.clone();changed[:,7]=100.;z,_=net(changed)
        self.assertTrue(torch.equal(y[:,:7],z[:,:7]),'future leaked')
        changed=x.clone();changed[:,0]=100.;z,_=net(changed)
        self.assertTrue(torch.equal(y[:,4:],z[:,4:]),'out-of-window frame leaked')
        y.square().mean().backward();self.assertGreater(float(net.mha.in_proj_weight.grad.abs().sum()),0)
    def test_entity_masks_unused_slots_and_export(self):
        m=initialize_attention(model(),entity=True,heads=4);net=EntityAttention(m['actor'],m['entity_attention']['actor'],4).cuda()
        x=torch.zeros(2,810,device='cuda');x[:,73]=1;x[:,169]=1;x[:,170]=1;x[:,25]=1;x[:,223]=1;x[:,260]=1
        tokens,valid=entity_tokens(x);self.assertEqual(tokens.shape,(2,29,68));self.assertEqual(int(valid[0].sum()),6)
        self.assertEqual(tokens[0,1,63].item(),1.);self.assertEqual(tokens[0,9,64].item(),1.)
        y=net.single(x);self.assertTrue(torch.equal(y,net.head(net.encoder(x))))
        with torch.no_grad():net.residual.weight.fill_(.1)
        y=net.single(x);y.square().mean().backward();self.assertGreater(float(net.token.weight.grad.abs().sum()),0)
        base,cell=net.export();other=EntityAttention(base,cell,4).cuda();self.assertTrue(torch.equal(y,other.single(x)))
    def test_entity_attention_padding_is_masked(self):
        m=initialize_attention(model(),entity=True,heads=4);net=EntityAttention(m['actor'],m['entity_attention']['actor'],4).cuda()
        tokens=torch.randn(1,5,8,device='cuda');query=torch.randn(1,1,8,device='cuda');mask=torch.tensor([[False,False,True,True,True]],device='cuda')
        with torch.no_grad():
            first=net.mha(query,tokens,tokens,key_padding_mask=mask,need_weights=False)[0]
            tokens[:,2:]=1000.;second=net.mha(query,tokens,tokens,key_padding_mask=mask,need_weights=False)[0]
        self.assertTrue(torch.equal(first,second))

if __name__=='__main__':unittest.main()

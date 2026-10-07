import unittest
from combat_weapon_head import torch, probabilities, feature_mask, distribution
from ppo_combat import anchor_kl

class WeaponHeadTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(),'CUDA mandatory for gradient checks')

    def test_masked_joint_likelihood_and_gpu_gradient(self):
        raw=torch.zeros(3,20,device='cuda',requires_grad=True)
        features=torch.zeros(3,845,device='cuda');features[:,833]=1
        features[0,834]=1;features[0,837]=1
        mask=feature_mask(features)
        with torch.no_grad():raw[:,17]=1000  # unavailable Railgun
        std=torch.tensor([-2.]*4,device='cuda');z=torch.zeros(3,4,device='cuda')
        attack=torch.ones(3,device='cuda');vertical=torch.zeros(3,dtype=torch.long,device='cuda')
        weapon=torch.tensor([4,0,0],device='cuda')
        lp,entropy=probabilities(raw,std,z,attack,vertical,weapon,mask)
        old,_=probabilities(raw[:,:8],std,z,attack,vertical)
        self.assertAlmostEqual(float((lp[0]-old[0]).detach()),-float(torch.log(torch.tensor(3.))),places=5)
        self.assertTrue(torch.equal(lp[1:],old[1:]))
        self.assertTrue(torch.isfinite(entropy).all())
        (-lp.mean()-.01*entropy.mean()).backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertEqual(float(raw.grad[:,17].abs().sum()),0)
        self.assertGreater(float(raw.grad[0,12].abs()),0)
        self.assertEqual(float(raw.grad[1:,8:].abs().sum()),0)
        self.assertEqual(float(distribution(raw[:,8:],mask).probs[:,9].sum().detach()),0)

    def test_rejects_masked_samples_and_invalid_masks(self):
        raw=torch.zeros(1,20,device='cuda');mask=torch.zeros(1,12,dtype=torch.bool,device='cuda');mask[:,0]=True
        args=(raw,torch.zeros(4,device='cuda'),torch.zeros(1,4,device='cuda'),torch.zeros(1,device='cuda'),torch.zeros(1,dtype=torch.long,device='cuda'))
        for choice in (-1,4,12):
            with self.assertRaises(ValueError):probabilities(*args,torch.tensor([choice],device='cuda'),mask)
        mask[:,0]=False
        with self.assertRaises(ValueError):distribution(raw[:,8:],mask)

    def test_legacy_anchor_retains_only_legacy_heads(self):
        raw=torch.zeros(2,20,device='cuda',requires_grad=True);teacher=torch.zeros(2,8,device='cuda');std=torch.zeros(4,device='cuda')
        with torch.no_grad():raw[:,8:]=100
        kl=anchor_kl(raw,std,teacher,std)
        self.assertEqual(float(kl.detach()),0)
        kl.backward();self.assertEqual(float(raw.grad[:,8:].abs().sum()),0)
        mask=torch.zeros(2,12,dtype=torch.bool,device='cuda');mask[:,0]=True;mask[:,4]=True
        teacher20=torch.zeros(2,20,device='cuda');teacher20[:,12]=2
        self.assertGreater(float(anchor_kl(raw,std,teacher20,std,weapon_mask=mask).detach()),0)

if __name__=='__main__':unittest.main()

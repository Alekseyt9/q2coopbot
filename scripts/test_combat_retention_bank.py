import copy, unittest
from combat_retention_bank import bucket,select,validate,pin_bank,rollout_seeds
from ppo_combat import torch,nn,anchor_kl,training_devices

class BankTest(unittest.TestCase):
    def test_expanded_rollout_checks_tail_seed_overlap(self):
        self.assertEqual(rollout_seeds(100,4),list(range(100,116)))
        expanded=rollout_seeds(100,4,3)
        self.assertEqual(expanded,list(range(100,148)))
        bank=self.bank();bank['train_seeds']=[140];bank['train'][0]['seed']=140
        validate(bank,'parent','v',2,rollout_seeds(100,4))
        with self.assertRaises(ValueError):validate(bank,'parent','v',2,expanded)
        self.assertEqual(rollout_seeds(2147483600,4,3)[-1],2147483647)
        for args in ((2147483601,4,3),(1,4,0),(-1,4,3)):
            with self.assertRaises(ValueError):rollout_seeds(*args)

    def bank(self):
        r=lambda seed:dict(features=[0.,1.],bucket='test',seed=seed,index=1,weight=1.)
        return dict(version='combat_retention_bank_v1',anchor_sha256='parent',feature_version='v',train_seeds=[1],validation_seeds=[2],forbidden_seeds=[3],source_sha256={},train=[r(1)],validation=[r(2)])
    def test_split_and_ppo_overlap_rejected(self):
        b=self.bank();validate(b,'parent','v',2,[4])
        for mutation in ('validation','forbidden','ppo'):
            x=copy.deepcopy(b)
            if mutation=='validation':x['validation_seeds']=[1]
            if mutation=='forbidden':x['forbidden_seeds']=[2]
            with self.assertRaises(ValueError):validate(x,'parent','v',2,[1] if mutation=='ppo' else [4])
    def test_features_and_nonobservation_fields_rejected(self):
        for key,value in [('reward',5),('features',[float('nan'),1.]),('weight',0.)]:
            b=self.bank();b['train'][0][key]=value
            with self.assertRaises(ValueError):validate(b,'parent','v',2)
        with self.assertRaises(ValueError):validate(self.bank(),'other','v',2)
    def test_deterministic_equal_bucket_mass(self):
        rows=[dict(seed=1,index=i,features=[i],bucket='common') for i in range(100)]+[dict(seed=2,index=i,features=[i],bucket='rare') for i in range(2)]
        chosen=select(rows,8);self.assertEqual(chosen,select(list(reversed(rows)),8))
        self.assertAlmostEqual(sum(r['weight'] for r in chosen if r['bucket']=='rare'),.5)
        self.assertAlmostEqual(sum(r['weight'] for r in chosen),1.)
    def test_pin_change_and_removal_rejected(self):
        s=pin_bank('hash',1.);self.assertEqual(s,pin_bank('hash',1.,s))
        for digest,weight in [(None,1.),('new',1.),('hash',2.),('hash',float('nan'))]:
            with self.assertRaises(ValueError):pin_bank(digest,weight,s)
    def test_weighted_kl_gradient_only_training_states(self):
        device=training_devices()[0]
        raw=nn.Parameter(torch.zeros(2,8,device=device));std=nn.Parameter(torch.zeros(4,device=device));teacher=torch.zeros_like(raw);teacher[0,0]=1.;teacher[1,0]=4.
        loss=anchor_kl(raw,std,teacher,torch.zeros(4,device=device),torch.tensor([1.,0.],device=device));loss.backward()
        self.assertLess(float(raw.grad[0,0]),0.);self.assertEqual(float(raw.grad[1].abs().sum()),0.)
    def test_bucket_unknown_clearance_not_wall(self):
        o=dict(enemies=[],local_geometry=dict(probes=[dict(standing_hull_clearance=None)]))
        self.assertIn('wall0',bucket(o));o['local_geometry']['probes'][0]['standing_hull_clearance']=20
        self.assertIn('wall1',bucket(o))
    def test_composition_order_and_unknown_type_preserved(self):
        o=dict(enemies=[dict(**{'class':'monster_parasite'}),dict(**{'class':'unknown'})])
        key=bucket(o,'composition');o['enemies'].reverse()
        self.assertEqual(key,bucket(o,'composition'));self.assertIn('unknown',key)
        self.assertNotEqual(bucket(dict(enemies=[{'class':'monster_gunner'}]),'composition'),bucket(dict(enemies=[{'class':'monster_parasite'}]),'composition'))
    def test_equal_composition_mass_despite_unequal_contexts(self):
        rows=[dict(seed=1,index=i,features=[i],bucket='classes:gunner/wall1') for i in range(3)]
        rows += [dict(seed=2,index=100*j+i,features=[i],bucket=f'classes:parasite/wall{j}') for j in range(3) for i in range(100)]
        chosen=select(rows,8,'composition')
        self.assertAlmostEqual(sum(r['weight'] for r in chosen if r['bucket'].startswith('classes:gunner/')),.5)
        self.assertAlmostEqual(sum(r['weight'] for r in chosen),1.)
        self.assertEqual(chosen,select(list(reversed(rows)),8,'composition'))

if __name__=='__main__':unittest.main()

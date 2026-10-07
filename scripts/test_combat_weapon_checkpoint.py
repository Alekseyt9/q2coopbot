"""Real checkpoint migration invariants; tensor copies, no optimization."""
import copy,json,pathlib,unittest
from migrate_combat_weapon_checkpoint import migrate,expanded,torch

ROOT=pathlib.Path(__file__).resolve().parents[1]

class MigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source=ROOT/'workspace/artifacts/combat-mobile-parasite-curriculum-v2-20261007/cycle/iteration-4/update'
        target=ROOT/'workspace/artifacts/combat-weapon-head-v1-20261007/model/weights.json'
        if not source.exists() or not target.exists():raise unittest.SkipTest('Real Curriculum24 and initialized weapon model required')
        cls.cp=torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=True)
        cls.old=json.loads((source/'weights.json').read_text());cls.new=json.loads(target.read_text())

    def test_exact_adam_prefix_zero_new_moments_rng_and_counters(self):
        result,changes=migrate(self.cp,self.old,self.new)
        self.assertEqual(len(changes),6)
        for key in ('config','anchor_sha256','bank_sha256','updates_completed','total_actor_steps','consumed_rollouts','retention_weights'):
            self.assertEqual(result[key],self.cp[key])
        self.assertTrue(torch.equal(result['rng'],self.cp['rng']))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(result['cuda_rng'],self.cp['cuda_rng'])))
        for role in ('actor','value'):
            before=self.cp[role+'_optimizer'];after=result[role+'_optimizer']
            self.assertEqual(before['param_groups'],after['param_groups'])
            for identifier,state in before['state'].items():
                for key,value in state.items():
                    got=after['state'][identifier][key]
                    if key=='step':self.assertTrue(torch.equal(value,got));continue
                    prefix=tuple(slice(0,n) for n in value.shape)
                    self.assertTrue(torch.equal(value,got[prefix]))
                    tail=got.clone();tail[prefix]=0
                    self.assertEqual(int(torch.count_nonzero(tail)),0)
            # migrate must not mutate the source tensors or its shapes.
            self.assertEqual(tuple(before['state'][0]['exp_avg'].shape),(64,814))

    def test_rejects_active_retention_and_changed_legacy_weights(self):
        active=copy.deepcopy(self.cp);active['retention_weights']=[1.,0.]
        with self.assertRaises(ValueError):migrate(active,self.old,self.new)
        changed=copy.deepcopy(self.new);changed['actor'][-1]['bias'][0]+=.1
        with self.assertRaises(ValueError):migrate(self.cp,self.old,changed)
        with self.assertRaises(ValueError):expanded(torch.zeros(2,3),(2,2))

if __name__=='__main__':unittest.main()

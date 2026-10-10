"""GPU CLI integration: resume must export valid base + branch weights."""
import unittest,pathlib,tempfile,json,sys,contextlib,io,copy
from ppo_recurrent import torch,main,initialize,Recurrent,probabilities,sha
from combat_attention import initialize_attention,CausalAttention,EntityAttention
from test_combat_attention import model as base_model

class ExportResumeTest(unittest.TestCase):
    def test_two_updates_keep_critic_array_and_publish_checkpoint_tensors(self):
        self.assertTrue(torch.cuda.is_available());torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        cache=pathlib.Path(__file__).resolve().parents[1]/'workspace/build/training-cache';cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            root=pathlib.Path(temporary);parent=base_model();parent_path=root/'parent.json';parent_path.write_text(json.dumps(parent))
            features=[1.]+[0.]*809
            bank=dict(version='combat_retention_bank_v1',anchor_sha256=sha(parent_path),feature_version=parent['feature_version'],train_seeds=[10],validation_seeds=[11],forbidden_seeds=[99],source_sha256={})
            for split,seed in [('train',10),('validation',11)]:bank[split]=[dict(features=features,bucket='unit',seed=seed,index=0,weight=1.)]
            bank_path=root/'bank.json';bank_path.write_text(json.dumps(bank))
            config=dict(version='combat_ppo_training_v1',seed=20261006,gamma=.99,**{'lambda':.95},clip=.2,actor_lr=.0001,value_lr=.0001,actor_steps=1,value_steps=1,target_kl=.01,entropy=.001,max_grad_norm=.5)
            config_path=root/'config.json';config_path.write_text(json.dumps(config));original_argv=sys.argv
            try:
                for architecture in ('gru','attention','entity'):
                    weights=root/(architecture+'.json');current=initialize(parent,4) if architecture=='gru' else initialize_attention(parent,architecture=='entity');weights.write_text(json.dumps(current));checkpoint=None
                    for update in (1,2):
                        current=json.loads(weights.read_text());key={'gru':'memory','attention':'attention','entity':'entity_attention'}[architecture];spec=current[key]
                        modules={n:Recurrent(current[n],spec[n]) if architecture=='gru' else CausalAttention(current[n],spec[n],4,32) if architecture=='attention' else EntityAttention(current[n],spec[n],4) for n in ('actor','value')}
                        modules={name:module.to('cuda') for name,module in modules.items()}
                        x=torch.tensor([features,[.5]+[0.]*809,[.8]+[0.]*809],dtype=torch.float32,device='cuda');states={};raw={}
                        with torch.no_grad():
                            for name,module in modules.items():
                                if architecture=='gru':raw[name],_,states[name]=module(x[None]);raw[name]=raw[name][0]
                                elif architecture=='attention':raw[name],states[name]=module(x[None]);raw[name]=raw[name][0]
                                else:raw[name]=module.single(x)
                            z=raw['actor'][:,:4]+.001;attack=torch.zeros(3,device='cuda');vertical=torch.zeros(3,dtype=torch.long,device='cuda');lp,_=probabilities(raw['actor'],torch.tensor(current['log_std'],device='cuda'),z,attack,vertical)
                        rows=[];context=[]
                        for i in range(3):
                            memory=None
                            if architecture!='entity':
                                memory=dict(reset=i==0)
                                for name in ('actor','value'):
                                    if architecture=='gru':memory[name]=[0.]*4 if i==0 else states[name][0,i-1].tolist()
                                    else:memory[name]=states[name][0,:i].flatten().tolist()
                                if architecture=='attention':memory['position']=i
                                context.append(dict(features=x[i].tolist(),seed=update,index=i,frame=100+i,memory=memory))
                            sample=dict(version='unit',sampling_seed=update,latent=z[i].tolist(),attack=False,vertical=0,log_probability=float(lp[i]),value=float(raw['value'][i,0]))
                            if memory is not None:sample['memory']=memory
                            rows.append(dict(features=x[i].tolist(),seed=update,index=i,frame=100+i,next_frame=101+i,sample=sample,reward=[1.,-1.,.2][i],next_value=float(raw['value'][i+1,0]) if i<2 else 0.,terminal=i==2,truncated=False))
                        data=root/f'{architecture}-data-{update}';data.mkdir();(data/'rollout.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                        meta=dict(version='combat_ppo_rollout_v1',feature_version=parent['feature_version'],reward_version='combat_reward_v1',model_sha256=sha(weights),rollout_sha256=sha(data/'rollout.jsonl'),source_sha256={},rows=3,policy_version='unit')
                        if architecture!='entity':
                            (data/'sequence.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in context));meta.update(sequence_sha256=sha(data/'sequence.jsonl'),sequence_rows=3,recurrent_version=spec['version'])
                        (data/'report.json').write_text(json.dumps(meta));out=root/f'{architecture}-update-{update}'
                        sys.argv=['test','--model',str(weights),'--data',str(data),'--config',str(config_path),'--out',str(out),'--anchor-model',str(parent_path),'--retention-bank',str(bank_path)]
                        if checkpoint:sys.argv+=['--resume',str(checkpoint)]
                        with contextlib.redirect_stdout(io.StringIO()):main()
                        weights=out/'weights.json';checkpoint=out/'checkpoint.pt';published=json.loads(weights.read_text());cp=torch.load(checkpoint,map_location='cpu',weights_only=True)
                        self.assertIsInstance(published['value'],list);self.assertEqual(len(published['value']),3);self.assertEqual(published[key]['version'],spec['version']);self.assertEqual(cp['weights_sha256'],sha(weights));self.assertEqual(cp['updates_completed'],update)
                        restored=Recurrent(published['value'],published[key]['value']) if architecture=='gru' else CausalAttention(published['value'],published[key]['value'],4,32) if architecture=='attention' else EntityAttention(published['value'],published[key]['value'],4)
                        self.assertTrue(all(torch.equal(v,cp['value'][k].cpu()) for k,v in restored.state_dict().items()))
            finally:sys.argv=original_argv

if __name__=='__main__':unittest.main()

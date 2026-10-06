"""Residual GRU PPO: Go inference, full provider context, CUDA-only TBPTT."""
import argparse,copy,json,math,pathlib,time
from ppo_combat import torch,nn,network,layers,sha,training_devices,advantages,validate_objective,anchor_kl,guarded_actor_step
from combat_retention_bank import read as read_bank,validate as validate_bank
from combat_attention import CausalAttention,EntityAttention,initialize_attention,TEMPORAL,ENTITY

VERSION='combat_residual_gru_v1'
read=lambda p:json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))

class Recurrent(nn.Module):
    def __init__(self,base,cell):
        super().__init__();net=network(base);self.encoder=net[:4];self.head=net[4]
        width=len(cell['hidden']['weight'][0]);self.gru=nn.GRU(len(base[1]['bias']),width,batch_first=True)
        self.output=nn.Linear(width,len(base[-1]['bias']))
        with torch.no_grad():
            for name,l in [('ih',cell['input']),('hh',cell['hidden'])]:
                getattr(self.gru,'weight_'+name+'_l0').copy_(torch.tensor(l['weight']))
                getattr(self.gru,'bias_'+name+'_l0').copy_(torch.tensor(l['bias']))
            self.output.weight.copy_(torch.tensor(cell['output']['weight']));self.output.bias.copy_(torch.tensor(cell['output']['bias']))
    def forward(self,x,h=None):
        encoded=self.encoder(x);state,h=self.gru(encoded,h)
        return self.head(encoded)+self.output(state),h,state
    def single(self,x):return self(x[:,None,:])[0][:,0,:]
    def export(self):
        cell={}
        for key,tag in [('input','ih'),('hidden','hh')]:
            cell[key]={'weight':getattr(self.gru,'weight_'+tag+'_l0').detach().cpu().tolist(),'bias':getattr(self.gru,'bias_'+tag+'_l0').detach().cpu().tolist()}
        cell['output']={'weight':self.output.weight.detach().cpu().tolist(),'bias':self.output.bias.detach().cpu().tolist()}
        return layers(nn.Sequential(*self.encoder,self.head)),cell

def initialize(model,width=64):
    assert not model.get('memory') and 1<=width<=128
    memory={'version':VERSION}
    for name in ('actor','value'):
        base=model[name];g=nn.GRU(len(base[1]['bias']),width,batch_first=True,device='cuda')
        cell={key:{'weight':getattr(g,'weight_'+tag+'_l0').detach().cpu().tolist(),'bias':getattr(g,'bias_'+tag+'_l0').detach().cpu().tolist()} for key,tag in [('input','ih'),('hidden','hh')]}
        cell['output']={'weight':[[0.]*width for _ in base[-1]['bias']],'bias':[0.]*len(base[-1]['bias'])};memory[name]=cell
    return {**copy.deepcopy(model),'memory':memory}

def prepare_context(context,rows,device):
    selected={(r['seed'],r['index']):i for i,r in enumerate(rows)}
    assert len(selected)==len(rows)
    segments=[];seen=set();last=None
    for c in context:
        key=(c['seed'],c['index']);assert key not in seen;seen.add(key)
        reset=c['memory']['reset']
        if reset:segments.append([])
        else:assert last is not None and c['seed']==last['seed'] and c['frame']==last['frame']+1,'Missing memory context/reset'
        assert segments
        if key in selected:
            row=rows[selected[key]];assert row['features']==c['features'] and row['sample']['memory']==c['memory'] and row['frame']==c['frame']
        segments[-1].append(c);last=c
    assert set(selected)<=seen
    x=torch.zeros(len(segments),max(map(len,segments)),len(context[0]['features']),device=device)
    order=[];locations=[]
    for b,segment in enumerate(segments):
        x[b,:len(segment)]=torch.tensor([c['features'] for c in segment],dtype=torch.float32,device=device)
        for t,c in enumerate(segment):
            k=(c['seed'],c['index'])
            if k in selected:order.append(selected[k]);locations.append((b,t))
    assert sorted(order)==list(range(len(rows)))
    indices=torch.tensor(locations,dtype=torch.long,device=device)
    inverse=torch.argsort(torch.tensor(order,device=device))
    return x,indices,inverse,segments

def sequence(model,prepared,bptt=32,check_states=False,name='actor'):
    x,indices,inverse,segments=prepared;outputs=[];h=None;errors=[]
    if isinstance(model,CausalAttention):
        output,tokens=model(x)
        if check_states:
            for b,segment in enumerate(segments):
                for t,c in enumerate(segment):
                    assert c['memory'].get('position',0)==t
                    observed=tokens[b,max(0,t-model.window+1):t].flatten()
                    expected=torch.tensor(c['memory'][name],dtype=torch.float32,device=x.device)
                    assert observed.shape==expected.shape
                    if len(expected):errors.append(float((observed-expected).abs().max()))
        return output[indices[:,0],indices[:,1]][inverse],max(errors,default=0.)
    for start in range(0,x.shape[1],bptt):
        chunk=x[:,start:start+bptt];before=h
        y,h,states=model(chunk,h)
        if check_states:
            zero=states.new_zeros((x.shape[0],1,states.shape[2])) if before is None else before.transpose(0,1)
            observed=torch.cat((zero,states[:,:-1]),1)
            for b,segment in enumerate(segments):
                valid=segment[start:start+bptt]
                if valid:
                    expected=torch.tensor([c['memory'][name] for c in valid],dtype=torch.float32,device=x.device)
                    errors.append(float((observed[b,:len(valid)]-expected).abs().max()))
        outputs.append(y);h=h.detach()  # current-model prefix; gradients stop every32 frames
    output=torch.cat(outputs,1);result=output[indices[:,0],indices[:,1]][inverse]
    return result,max(errors,default=0.)

def probabilities(raw,std,z,attack,vertical):
    d=torch.distributions
    distributions=(d.Normal(raw[:,:4],std.exp()),d.Bernoulli(logits=raw[:,4]),d.Categorical(logits=raw[:,5:]))
    a,b,c=distributions
    return a.log_prob(z).sum(1)+b.log_prob(attack)+c.log_prob(vertical),a.entropy().sum(1)+b.entropy()+c.entropy()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True,type=pathlib.Path);ap.add_argument('--out',required=True,type=pathlib.Path)
    ap.add_argument('--init',action='store_true');ap.add_argument('--architecture',choices=('gru','attention','entity'),default='gru');ap.add_argument('--width',type=int,default=64);ap.add_argument('--data',type=pathlib.Path);ap.add_argument('--config',type=pathlib.Path);ap.add_argument('--resume',type=pathlib.Path)
    ap.add_argument('--anchor-model',type=pathlib.Path);ap.add_argument('--retention-bank',type=pathlib.Path)
    ap.add_argument('--retention-weight',type=float,default=1.);ap.add_argument('--bank-weight',type=float,default=1.)
    args=ap.parse_args();training_devices();torch.set_num_threads(2);torch.manual_seed(20261006);torch.use_deterministic_algorithms(True)
    assert all(math.isfinite(w) and w >= 0 for w in (args.retention_weight,args.bank_weight))
    # cuDNN GRU TF32 drifts from Go FP64 by ~2e-3 on real traces.
    # Keep FP32 CUDA training and the existing state parity tolerance.
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    model=read(args.model);assert model['kind']=='combat_ppo_v1' and not model['deterministic']
    if args.init:
        args.out.mkdir(exist_ok=False);updated=initialize(model,args.width) if args.architecture=='gru' else initialize_attention(model,args.architecture=='entity')
        (args.out/'weights.json').write_text(json.dumps(updated,allow_nan=False))
        return
    config=read(args.config);meta=read(args.data/'report.json');validate_objective(config,meta)
    key=next(k for k in ('memory','attention','entity_attention') if model.get(k));spec=model[key];entity=key=='entity_attention'
    expected_version={'memory':VERSION,'attention':TEMPORAL,'entity_attention':ENTITY}[key];assert spec['version']==expected_version
    if not entity:assert meta['recurrent_version']==expected_version
    assert sha(args.model)==meta['model_sha256'] and sha(args.data/'rollout.jsonl')==meta['rollout_sha256']
    if not entity:assert sha(args.data/'sequence.jsonl')==meta['sequence_sha256']
    for path,digest in meta['source_sha256'].items():assert sha(pathlib.Path(path))==digest
    rows=[json.loads(s) for s in (args.data/'rollout.jsonl').read_text().splitlines()]
    context=[] if entity else [json.loads(s) for s in (args.data/'sequence.jsonl').read_text().splitlines()]
    assert len(rows)==meta['rows']>1
    if not entity:assert len(context)==meta['sequence_rows']
    assert all(r['sample']['version']==meta['policy_version'] for r in rows)
    device='cuda';prepared=None if entity else prepare_context(context,rows,device);bptt=config.get('bptt_steps',32);assert 1<=bptt<=128
    def build(name):
        if key=='memory':module=Recurrent(model[name],spec[name])
        elif entity:module=EntityAttention(model[name],spec[name],spec['heads'])
        else:module=CausalAttention(model[name],spec[name],spec['heads'],spec['window'])
        return module.to(device)
    actor=build('actor');value=build('value')
    x=torch.tensor([r['features'] for r in rows],dtype=torch.float32,device=device)
    def compute(module,check=False,name='actor'):
        return (module.single(x),0.) if entity else sequence(module,prepared,bptt,check,name)
    std=nn.Parameter(torch.tensor(model['log_std'],device=device));adv,returns=advantages(rows,config['gamma'],config['lambda'])
    z=torch.tensor([r['sample']['latent'] for r in rows],device=device);attack=torch.tensor([float(r['sample']['attack']) for r in rows],device=device);vertical=torch.tensor([r['sample']['vertical'] for r in rows],dtype=torch.long,device=device)
    old=torch.tensor([r['sample']['log_probability'] for r in rows],device=device);ad=torch.tensor(adv,device=device);ret=torch.tensor(returns,device=device)
    with torch.no_grad():
        raw,actor_state_error=compute(actor,True,'actor');v,value_state_error=compute(value,True,'value')
        lp,_=probabilities(raw,std,z,attack,vertical)
        log_error=float((lp-old).abs().max());value_error=float((v[:,0]-torch.tensor([r['sample']['value'] for r in rows],device=device)).abs().max())
        assert log_error<.003 and value_error<1e-4 and max(actor_state_error,value_state_error)<2e-5,(log_error,value_error,actor_state_error,value_state_error)
    bank_tensors={}
    if args.retention_weight or args.bank_weight:
        anchor=read(args.anchor_model);assert not anchor.get('memory') and anchor['feature_version']==model['feature_version']
        anchor_net=network(anchor['actor']).to(device);teacher_std=torch.tensor(anchor['log_std'],device=device)
        with torch.no_grad():
            if args.retention_weight:teacher=anchor_net(x)
            if args.bank_weight:
                bank=read_bank(args.retention_bank);validate_bank(bank,sha(args.anchor_model),model['feature_version'],x.shape[1],{r['seed'] for r in rows})
                for split in ('train','validation'):
                    bx=torch.tensor([r['features'] for r in bank[split]],dtype=torch.float32,device=device);bw=torch.tensor([r['weight'] for r in bank[split]],device=device)
                    bank_tensors[split]=(bx,bw,anchor_net(bx))
        del anchor_net
    def bank_loss(split='train'):
        bx,bw,bt=bank_tensors[split];return anchor_kl(actor.single(bx),std,bt,teacher_std,bw)
    def objective():
        raw,_=compute(actor);lp,entropy=probabilities(raw,std,z,attack,vertical);ratio=(lp-old).exp()
        loss=-torch.minimum(ratio*ad,ratio.clamp(1-config['clip'],1+config['clip'])*ad).mean()-config['entropy']*entropy.mean()
        if args.retention_weight:loss=loss+args.retention_weight*anchor_kl(raw,std,teacher,teacher_std)
        if args.bank_weight:loss=loss+args.bank_weight*bank_loss()
        return loss,((ratio-1)-(lp-old)).mean()
    actor_opt=torch.optim.Adam(list(actor.parameters())+[std],lr=config['actor_lr']);value_opt=torch.optim.Adam(value.parameters(),lr=config['value_lr'])
    consumed=[];updates=0;total=0
    if args.resume:
        cp=torch.load(args.resume,map_location='cpu',weights_only=True)
        assert cp['version']=='combat_architecture_checkpoint_v1' and cp['architecture']==expected_version and cp['weights_sha256']==sha(args.model) and cp['config']==config
        assert cp['anchor_sha256']==sha(args.anchor_model) and cp['bank_sha256']==sha(args.retention_bank)
        assert cp.get('retention_weights',[1.,1.])==[args.retention_weight,args.bank_weight]
        for module_name,module in [('actor',actor),('value',value)]:
            assert all(torch.equal(v.cpu(),cp[module_name][k].cpu()) for k,v in module.state_dict().items())
        assert torch.equal(std.detach().cpu(),cp['log_std'].cpu())
        consumed=cp['consumed_rollouts'];assert meta['rollout_sha256'] not in consumed
        updates=cp['updates_completed'];total=cp['total_actor_steps']
        actor_opt.load_state_dict(cp['actor_optimizer']);value_opt.load_state_dict(cp['value_optimizer'])
        torch.set_rng_state(cp['rng']);torch.cuda.set_rng_state_all(cp['cuda_rng'])
    ad=(ad-ad.mean())/(ad.std(unbiased=False)+1e-8)
    with torch.no_grad():initial_bank={s:float(bank_loss(s)) for s in bank_tensors}
    trials=[];start=time.perf_counter();parameters=list(actor.parameters())+[std]
    for _ in range(config['actor_steps']):
        loss,kl=objective()
        if float(kl.detach())>config['target_kl']:break
        actor_opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(parameters,config['max_grad_norm'])
        trial=guarded_actor_step(parameters,actor_opt,objective,loss.detach(),config['actor_lr'],config['target_kl'],lambda:std.clamp_(-8,1));trials.append(trial)
        if not trial['accepted']:break
    for _ in range(config['value_steps']):
        prediction,_=compute(value);loss=(prediction[:,0]-ret).square().mean();assert torch.isfinite(loss)
        value_opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(value.parameters(),config['max_grad_norm']);value_opt.step()
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    with torch.no_grad():loss,kl=objective();final_bank={s:float(bank_loss(s)) for s in bank_tensors}
    assert torch.isfinite(kl) and float(kl)<=config['target_kl']+1e-6
    base_a,cell_a=actor.export();base_v,cell_v=value.export()
    updated={**model,'actor':base_a,'value':base_v,'log_std':std.detach().cpu().tolist(),key:{**spec,'actor':cell_a,'value':cell_v}}
    done=sum(t['accepted'] for t in trials)
    for path,digest in meta['source_sha256'].items():assert sha(pathlib.Path(path))==digest
    if not entity:assert sha(args.data/'sequence.jsonl')==meta['sequence_sha256']
    assert sha(args.data/'rollout.jsonl')==meta['rollout_sha256']
    args.out.mkdir(exist_ok=False);(args.out/'weights.json').write_text(json.dumps(updated,allow_nan=False))
    cp={'version':'combat_architecture_checkpoint_v1','architecture':expected_version,'weights_sha256':sha(args.out/'weights.json'),'config':config,'anchor_sha256':sha(args.anchor_model),'bank_sha256':sha(args.retention_bank),'actor':actor.state_dict(),'value':value.state_dict(),'log_std':std.detach(),'actor_optimizer':actor_opt.state_dict(),'value_optimizer':value_opt.state_dict(),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'consumed_rollouts':consumed+[meta['rollout_sha256']],'updates_completed':updates+1,'total_actor_steps':total+done}
    cp['retention_weights']=[args.retention_weight,args.bank_weight]
    torch.save(cp,args.out/'checkpoint.pt');(args.out/'trainer.py').write_bytes(pathlib.Path(__file__).read_bytes())
    report={'architecture':expected_version,'device':'cuda','torch':torch.__version__,'rows':len(rows),'sequence_rows':len(context),'sequences':len(prepared[3]) if prepared else 0,'bptt_steps':bptt if key=='memory' else None,'actor_steps':done,'total_actor_steps':total+done,'updates_completed':updates+1,'actor_trials':trials,'final_approx_kl':float(kl),'old_log_probability_max_error':log_error,'old_value_max_error':value_error,'old_actor_memory_max_error':actor_state_error,'old_value_memory_max_error':value_state_error,'bank_kl_before':initial_bank,'bank_kl_after':final_bank,'seconds':seconds,'weights_sha256':sha(args.out/'weights.json'),'behavior_sha256':sha(args.model),'rollout_sha256':meta['rollout_sha256'],'sequence_sha256':meta.get('sequence_sha256'),'trainer_sha256':sha(pathlib.Path(__file__)),'config_sha256':sha(args.config),'resume_sha256':sha(args.resume) if args.resume else None,'anchor_sha256':sha(args.anchor_model),'bank_sha256':sha(args.retention_bank),'actor_parameters':sum(p.numel() for p in actor.parameters())+std.numel(),'critic_parameters':sum(p.numel() for p in value.parameters()),'scope':'Fresh on-policy architecture PPO. GRU replays all provider context with current-model prefixes, detached every32 frames. Temporal attention is causal/window32; entity attention uses only current v4 values. Loss only on native eligible rows. Bank has no histories: retention measured at zero memory. No learned weapon choice or live promotion.'}
    report['retention_weight']=args.retention_weight;report['bank_weight']=args.bank_weight
    (args.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report,indent=2))

if __name__=='__main__':main()

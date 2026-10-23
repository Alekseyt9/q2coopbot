"""Residual GRU PPO: Go inference, full provider context, CUDA-only TBPTT."""
import argparse,copy,json,math,os,pathlib,time
from ppo_combat import torch,nn,network,layers,sha,training_devices,advantages,validate_objective,anchor_kl,guarded_actor_step
from combat_retention_bank import read as read_bank,validate as validate_bank
from combat_attention import CausalAttention,EntityAttention,initialize_attention,TEMPORAL,ENTITY
from combat_weapon_head import probabilities,feature_mask,HEAD_VERSION

VERSION='combat_residual_gru_v1'
read=lambda p:json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))

def durable_write(path, writer):
    """Flush a new artifact before publishing its final name."""
    path=pathlib.Path(path);pending=path.with_name(path.name+'.partial')
    with pending.open('xb') as output:
        writer(output);output.flush();os.fsync(output.fileno())
    os.replace(pending,path)

def durable_json(path,value):
    encoded=json.dumps(value,indent=2,allow_nan=False).encode('utf-8')
    durable_write(path,lambda output:output.write(encoded))

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
    core=getattr(model,'base',model)
    if isinstance(core,CausalAttention):
        output,tokens=model(x)
        if check_states:
            for b,segment in enumerate(segments):
                for t,c in enumerate(segment):
                    assert c['memory'].get('position',0)==t
                    observed=tokens[b,max(0,t-core.window+1):t].flatten()
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

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True,type=pathlib.Path);ap.add_argument('--out',required=True,type=pathlib.Path)
    ap.add_argument('--init',action='store_true');ap.add_argument('--architecture',choices=('gru','attention','entity'),default='gru');ap.add_argument('--width',type=int,default=64);ap.add_argument('--data',type=pathlib.Path);ap.add_argument('--config',type=pathlib.Path);ap.add_argument('--resume',type=pathlib.Path)
    ap.add_argument('--anchor-model',type=pathlib.Path);ap.add_argument('--retention-bank',type=pathlib.Path)
    ap.add_argument('--retention-weight',type=float,default=1.);ap.add_argument('--bank-weight',type=float,default=1.)
    ap.add_argument('--sequence-retention-spec',type=pathlib.Path)
    ap.add_argument('--sequence-retention-weight',type=float,default=0.)
    ap.add_argument('--fork-sequence-retention',action='store_true',
                    help='Explicit objective fork preserving actor/critic/Adam for sequence retention')
    ap.add_argument('--actor-head-scope',choices=('all','target','decisions','conditional-aim','movement','motor-residual','base-only','conditional-fire'),default='all')
    ap.add_argument('--target-head-lr',type=float,default=.003)
    ap.add_argument('--decision-head-lr',type=float,default=.003)
    ap.add_argument('--movement-head-lr',type=float,default=.00003)
    ap.add_argument('--motor-lr',type=float,default=.0003)
    ap.add_argument('--conditional-fire-lr',type=float,default=.0003)
    ap.add_argument('--fork-head-scope',action='store_true',help='Explicit optimizer-scope fork from a resumed checkpoint')
    ap.add_argument('--weapon-state-lr',type=float,default=.0003)
    ap.add_argument('--future-aux-data',type=pathlib.Path)
    ap.add_argument('--future-aux-decoder',type=pathlib.Path)
    ap.add_argument('--future-aux-weight',type=float,default=0.)
    ap.add_argument('--fork-future-aux',action='store_true',help='Explicitly initialize/change/disable a training-only future auxiliary objective')
    ap.add_argument('--initial-boundary',type=pathlib.Path,help='Optional fresh sealed pre-update actor/critic/optimizer/RNG snapshot')
    ap.add_argument('--critic-credit',choices=('legacy','mc-after','mc-before'),default='legacy')
    ap.add_argument('--fork-critic-credit',action='store_true',help='Explicit critic target/timing experiment fork')
    ap.add_argument('--policy-update',choices=('ppo','bc','awr'),default='ppo')
    ap.add_argument('--fork-policy-update',action='store_true')
    ap.add_argument('--awr-temperature',type=float,default=3.)
    ap.add_argument('--awr-weight-cap',type=float,default=20.)
    ap.add_argument('--awr-strength',type=float,choices=(0.,1.),default=1.)
    args=ap.parse_args();training_devices();torch.set_num_threads(2);torch.manual_seed(20261006);torch.use_deterministic_algorithms(True)
    assert all(math.isfinite(w) and w >= 0 for w in (args.retention_weight,args.bank_weight))
    assert math.isfinite(args.sequence_retention_weight) and args.sequence_retention_weight >= 0
    assert args.sequence_retention_spec is not None or args.sequence_retention_weight == 0
    assert not args.fork_sequence_retention or (args.resume and args.sequence_retention_spec)
    assert not args.fork_head_scope or args.resume
    assert math.isfinite(args.future_aux_weight) and args.future_aux_weight>=0
    assert bool(args.future_aux_data)==bool(args.future_aux_decoder)
    assert args.future_aux_data is not None or args.future_aux_weight==0
    assert not args.fork_future_aux or args.resume
    assert not args.future_aux_data or (args.resume and args.actor_head_scope=='all')
    assert not args.fork_critic_credit or args.resume
    assert not args.fork_policy_update or args.resume
    assert math.isfinite(args.awr_temperature) and args.awr_temperature>0
    assert math.isfinite(args.awr_weight_cap) and args.awr_weight_cap>=1
    assert args.policy_update=='ppo' or (args.critic_credit=='mc-before' and args.actor_head_scope=='all' and not args.future_aux_data)
    assert math.isfinite(args.target_head_lr) and 0 < args.target_head_lr <= .01
    assert math.isfinite(args.decision_head_lr) and 0 < args.decision_head_lr <= .01
    assert math.isfinite(args.movement_head_lr) and 0 < args.movement_head_lr <= .01
    # cuDNN GRU TF32 drifts from Go FP64 by ~2e-3 on real traces.
    # Keep FP32 CUDA training and the existing state parity tolerance.
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    model=read(args.model);assert model['kind']=='combat_ppo_v1' and not model['deterministic']
    assert not model.get('execution_mode'), 'Inference-only execution modes cannot train PPO'
    if model.get('conditional_fire'):
        from combat_conditional_fire import validate as validate_fire,VERSION as FIRE_VERSION
        validate_fire(model)
        assert args.resume and args.sequence_retention_spec and args.sequence_retention_weight>0
        assert args.actor_head_scope in ('all','conditional-fire') and not args.future_aux_data
        assert math.isfinite(args.conditional_fire_lr) and 0<args.conditional_fire_lr<=.01
    weapon_head=model.get('weapon_head')
    target_head=model.get('target_head')
    if target_head:
        if model.get('aim_mode_head'):
            from combat_precision_head import validate_precision_model as validate_model
        else:
            from combat_target_head import validate_model
        validate_model(model)
        assert not args.retention_weight and not args.bank_weight, 'Target-aware retention objective not implemented'
    elif weapon_head:
        assert weapon_head==HEAD_VERSION and model['feature_version']=='combat_features_v6' and not model.get('entity_attention')
        assert len(model['actor'][-1]['bias'])==20
    else:assert len(model['actor'][-1]['bias'])==8
    if args.init:
        args.out.mkdir(exist_ok=False);updated=initialize(model,args.width) if args.architecture=='gru' else initialize_attention(model,args.architecture=='entity')
        (args.out/'weights.json').write_text(json.dumps(updated,allow_nan=False))
        return
    config=read(args.config);meta=read(args.data/'report.json');validate_objective(config,meta)
    assert args.actor_head_scope in ('all','conditional-aim','movement','motor-residual','base-only','conditional-fire') or (target_head and not args.sequence_retention_spec)
    assert args.actor_head_scope!='conditional-fire' or model.get('conditional_fire')
    assert args.actor_head_scope not in ('motor-residual','base-only') or model.get('motor_residual')
    assert args.actor_head_scope!='movement' or target_head
    assert args.actor_head_scope != 'conditional-aim' or model.get('sampled_motion_aim'), 'Conditional aim scope requires the sampled-motion branch'
    assert args.actor_head_scope != 'decisions' or weapon_head, 'Decision scope requires the masked weapon head'
    actor_training=dict(scope=args.actor_head_scope,learning_rate={'target':args.target_head_lr,'decisions':args.decision_head_lr,'all':config['actor_lr'],'conditional-aim':config['actor_lr'],'movement':args.movement_head_lr,'motor-residual':config['actor_lr'],'base-only':config['actor_lr'],'conditional-fire':args.conditional_fire_lr}[args.actor_head_scope])
    if model.get('conditional_fire'):
        actor_training['conditional_fire']=dict(version=FIRE_VERSION,learning_rate=args.conditional_fire_lr)
    if model.get('motor_residual'):
        from combat_motor_residual import validate as validate_motor,VERSION as MOTOR_VERSION
        validate_motor(model);assert math.isfinite(args.motor_lr) and 0<args.motor_lr<=.01
        actor_training['motor_residual']=dict(version=MOTOR_VERSION,learning_rate=args.motor_lr)
    if model.get('weapon_state_residual'):
        from combat_weapon_state_residual import validate as validate_weapon_state,CONTRACT_VERSION
        validate_weapon_state(model)
        assert args.actor_head_scope in ('all','conditional-aim','movement','motor-residual','base-only','conditional-fire'), 'Observed weapon residual requires an explicit supported actor scope'
        assert math.isfinite(args.weapon_state_lr) and 0<args.weapon_state_lr<=.01
        actor_training['weapon_state_residual']=dict(version=CONTRACT_VERSION,learning_rate=args.weapon_state_lr)
    assert meta['feature_version']==model['feature_version'], 'Rollout feature contract differs'
    key=next(k for k in ('memory','attention','entity_attention') if model.get(k));spec=model[key];entity=key=='entity_attention'
    expected_version={'memory':VERSION,'attention':TEMPORAL,'entity_attention':ENTITY}[key];assert spec['version']==expected_version
    assert args.critic_credit=='legacy' or (key=='attention' and args.resume and args.actor_head_scope=='all' and not args.future_aux_data)
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
        if name=='actor' and model.get('spatial_aim'):
            from combat_spatial_aim import wrap
            module=wrap(module,model)
        return module.to(device)
    actor=build('actor');value=build('value')
    x=torch.tensor([r['features'] for r in rows],dtype=torch.float32,device=device)
    mask=feature_mask(x[:,:845]) if weapon_head else None
    target=torch.tensor([r['sample'].get('target',0) for r in rows],dtype=torch.long,device=device) if target_head else None
    if not target_head:assert all(r['sample'].get('target',0)==0 for r in rows)
    mode=torch.tensor([r['sample'].get('aim_mode',0) for r in rows],dtype=torch.long,device=device) if model.get('aim_mode_head') else None
    if mode is None:assert all(r['sample'].get('aim_mode',0)==0 for r in rows)
    def likelihood(raw,std,z,attack,vertical,weapon,mask):
        if mode is not None:
            from combat_precision_head import probabilities as precision_probabilities
            return precision_probabilities(raw,std,z,attack,vertical,weapon,x,target,mode,getattr(actor,'sampled_motion_aim_branch',None),getattr(actor,'motor_branch',None),getattr(actor,'conditional_fire_branch',None))
        if target_head:
            from combat_target_head import probabilities as target_probabilities
            return target_probabilities(raw,std,z,attack,vertical,weapon,x,target)
        return probabilities(raw,std,z,attack,vertical,weapon,mask)
    weapon=torch.tensor([r['sample'].get('weapon',0) for r in rows],dtype=torch.long,device=device) if weapon_head else None
    if not weapon_head:assert all(r['sample'].get('weapon',0)==0 for r in rows)
    def compute(module,check=False,name='actor'):
        return (module.single(x),0.) if entity else sequence(module,prepared,bptt,check,name)
    std=nn.Parameter(torch.tensor(model['log_std'],device=device));adv,returns=advantages(rows,config['gamma'],config['lambda'])
    z=torch.tensor([r['sample']['latent'] for r in rows],device=device);attack=torch.tensor([float(r['sample']['attack']) for r in rows],device=device);vertical=torch.tensor([r['sample']['vertical'] for r in rows],dtype=torch.long,device=device)
    old=torch.tensor([r['sample']['log_probability'] for r in rows],device=device);ad=torch.tensor(adv,device=device);ret=torch.tensor(returns,device=device)
    with torch.no_grad():
        raw,actor_state_error=compute(actor,True,'actor');v,value_state_error=compute(value,True,'value')
        lp,_=likelihood(raw,std,z,attack,vertical,weapon,mask)
        log_error=float((lp-old).abs().max());value_error=float((v[:,0]-torch.tensor([r['sample']['value'] for r in rows],device=device)).abs().max())
        assert log_error<.003 and value_error<1e-4 and max(actor_state_error,value_state_error)<2e-5,(log_error,value_error,actor_state_error,value_state_error)
    sequence_retention = None
    if args.sequence_retention_spec:
        assert target_head and not entity, 'Sequence retention requires an explicit target-aware history contract'
        from combat_sequence_retention import load_spec, policy_kl
        seq_spec, positions = load_spec(args.sequence_retention_spec,meta,rows,sha(args.model))
        seq_positions = torch.tensor(positions,dtype=torch.long,device=device)
        seq_weights = torch.tensor([m['weight'] for m in seq_spec['members']],dtype=torch.float64,device=device)
        seq_teacher, seq_std = raw.detach().clone(), std.detach().clone()
        sequence_retention = dict(version=seq_spec['version'],spec_sha256=sha(args.sequence_retention_spec),
            weight=args.sequence_retention_weight,rows=len(positions),behavior_sha256=sha(args.model))
        motion_branch=getattr(actor,'sampled_motion_aim_branch',None)
        teacher_motion=copy.deepcopy(motion_branch) if motion_branch is not None else None
        motor_branch=getattr(actor,'motor_branch',None)
        teacher_motor=copy.deepcopy(motor_branch) if motor_branch is not None else None
        fire_branch=getattr(actor,'conditional_fire_branch',None)
        teacher_fire=copy.deepcopy(fire_branch) if fire_branch is not None else None
        fire_context=None
        if teacher_fire is not None:
            from combat_sequence_retention import CONDITIONAL_FIRE_EXPECTATION
            for parameter in teacher_fire.parameters():parameter.requires_grad_(False)
            sequence_retention['conditional_fire_expectation']=CONDITIONAL_FIRE_EXPECTATION
            fire_context=tuple(v[seq_positions].detach() for v in (target,mode,z,vertical,weapon))
        if teacher_motor is not None:
            for parameter in teacher_motor.parameters():parameter.requires_grad_(False)
        if teacher_motion is not None:
            for parameter in teacher_motion.parameters():parameter.requires_grad_(False)
            sequence_retention['conditional_motion_expectation']='fixed_behavior_move_per_history_v1'
        def sequence_loss(current):
            terms = policy_kl(current[seq_positions],std,seq_teacher[seq_positions],seq_std,x[seq_positions],
                              motion_branch,teacher_motion,z[seq_positions,:2].detach(),motor_branch,teacher_motor,
                              fire_branch,teacher_fire,fire_context)
            return (terms * seq_weights).sum()
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
        bx,bw,bt=bank_tensors[split];return anchor_kl(actor.single(bx),std,bt,teacher_std,bw,feature_mask(bx) if weapon_head else None)
    future_aux=None;regression_weights=None;regression_stats=None
    def objective():
        raw,_=compute(actor);lp,entropy=likelihood(raw,std,z,attack,vertical,weapon,mask);ratio=(lp-old).exp()
        if args.policy_update=='ppo':
            loss=-torch.minimum(ratio*ad,ratio.clamp(1-config['clip'],1+config['clip'])*ad).mean()-config['entropy']*entropy.mean()
        else:
            loss=-(regression_weights*lp).mean()-config['entropy']*entropy.mean()
        if args.retention_weight:loss=loss+args.retention_weight*anchor_kl(raw,std,teacher,teacher_std,weapon_mask=mask)
        if args.bank_weight:loss=loss+args.bank_weight*bank_loss()
        if args.sequence_retention_weight:loss=loss+args.sequence_retention_weight*sequence_loss(raw)
        if future_aux is not None:loss=loss+future_aux.actor_loss(actor)
        return loss,((ratio-1)-(lp-old)).mean()
    from combat_weapon_state_residual import optimizer as actor_optimizer
    actor_opt=actor_optimizer(actor,std,config['actor_lr'],args.weapon_state_lr,args.motor_lr,args.conditional_fire_lr);value_opt=torch.optim.Adam(value.parameters(),lr=config['value_lr'])
    consumed=[];updates=0;total=0;migrations=[];config_forks=[];future_forks=[];previous_future=None
    critic_credit=dict(version='combat_critic_credit_v1',mode=args.critic_credit)
    critic_forks=[]
    policy_update=dict(version='combat_policy_update_v1',method=args.policy_update)
    if args.policy_update=='awr':policy_update.update(temperature=args.awr_temperature,weight_cap=args.awr_weight_cap,strength=args.awr_strength)
    policy_update_forks=[]
    if args.resume:
        cp=torch.load(args.resume,map_location='cpu',weights_only=True)
        migrations=cp.get('model_migrations',[])
        config_forks=cp.get('training_config_forks',[])
        assert cp['version']=='combat_architecture_checkpoint_v1' and cp['architecture']==expected_version and cp['weights_sha256']==sha(args.model) and cp['config']==config
        assert cp['anchor_sha256']==sha(args.anchor_model) and cp['bank_sha256']==sha(args.retention_bank)
        assert cp.get('retention_weights',[1.,1.])==[args.retention_weight,args.bank_weight]
        previous_training=cp.get('actor_training',dict(scope='all',learning_rate=config['actor_lr']))
        assert not cp.get('sac_training') or args.fork_head_scope, 'SAC to PPO requires an explicit optimizer/algorithm scope fork'
        assert previous_training == actor_training or args.fork_head_scope, 'Explicit actor scope fork required'
        if not args.fork_sequence_retention:
            assert cp.get('sequence_retention') == sequence_retention, 'Explicit sequence objective fork required'
        for module_name,module in [('actor',actor),('value',value)]:
            assert all(torch.equal(v,cp[module_name][k].to(device=v.device)) for k,v in module.state_dict().items())
        assert torch.equal(std.detach(),cp['log_std'].to(device=std.device))
        consumed=cp['consumed_rollouts'];assert meta['rollout_sha256'] not in consumed
        updates=cp['updates_completed'];total=cp['total_actor_steps']
        actor_opt.load_state_dict(cp['actor_optimizer']);value_opt.load_state_dict(cp['value_optimizer'])
        torch.set_rng_state(cp['rng']);torch.cuda.set_rng_state_all(cp['cuda_rng'])
        previous_future=cp.get('future_auxiliary')
        future_forks=cp.get('future_objective_forks',[])
        previous_credit=cp.get('critic_credit',dict(version='combat_critic_credit_v1',mode='legacy'))
        assert previous_credit==critic_credit or args.fork_critic_credit,'Explicit critic-credit fork required'
        critic_forks=cp.get('critic_credit_forks',[])
        previous_policy=cp.get('policy_update',dict(version='combat_policy_update_v1',method='ppo'))
        assert previous_policy==policy_update or args.fork_policy_update,'Explicit policy-update fork required'
        policy_update_forks=cp.get('policy_update_forks',[])
        if args.fork_policy_update:
            policy_update_forks=policy_update_forks+[dict(source_checkpoint_sha256=sha(args.resume),previous=previous_policy,current=policy_update)]
        if args.fork_critic_credit:
            critic_forks=critic_forks+[dict(source_checkpoint_sha256=sha(args.resume),previous=previous_credit,current=critic_credit)]
    if args.future_aux_data:
        assert key=='attention' and not entity and model['feature_version']=='combat_features_v10'
        assert previous_future is not None or args.fork_future_aux,'Explicit future objective fork required'
        from combat_future_auxiliary import FutureAuxiliary
        future_aux=FutureAuxiliary(actor,prepared,args.future_aux_data,args.future_aux_decoder,args.future_aux_weight,
            sha(args.model),None if args.fork_future_aux else previous_future)
    elif previous_future is not None:
        assert args.fork_future_aux,'Explicit fork required to disable future auxiliary objective'
    if args.fork_future_aux:
        future_forks=future_forks+[dict(source_checkpoint_sha256=sha(args.resume),
            previous=None if previous_future is None else previous_future['spec'],current=None if future_aux is None else future_aux.spec)]
    initial_boundary_sha=None
    if args.initial_boundary:
        assert not args.initial_boundary.exists()
        boundary=dict(actor=actor.state_dict(),value=value.state_dict(),log_std=std.detach(),
            actor_optimizer=actor_opt.state_dict(),value_optimizer=value_opt.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            future_auxiliary=None if future_aux is None else future_aux.checkpoint(),
            source_model_sha256=sha(args.model),source_checkpoint_sha256=sha(args.resume) if args.resume else None,rollout_sha256=meta['rollout_sha256'])
        durable_write(args.initial_boundary,lambda output:torch.save(boundary,output))
        initial_boundary_sha=sha(args.initial_boundary)
    if args.critic_credit!='legacy':
        _,mc_returns=advantages(rows,config['gamma'],1.)
        ret=torch.tensor(mc_returns,device=device)
    with torch.no_grad():initial_bank={s:float(bank_loss(s)) for s in bank_tensors}
    with torch.no_grad():sequence_kl_before=float(sequence_loss(raw)) if sequence_retention else None
    future_before=None if future_aux is None else future_aux.stats(actor)
    trials=[];start=time.perf_counter();parameters=list(actor.parameters())+[std]
    project=lambda:std.clamp_(-8,1)
    trainable_actor_parameters=sum(p.numel() for p in parameters)
    if args.actor_head_scope in ('target','decisions','movement'):
        from combat_target_head_scope import freeze_decision_scope
        project,trainable_actor_parameters=freeze_decision_scope(actor,std,actor_opt,args.actor_head_scope)
    elif args.actor_head_scope == 'conditional-aim':
        from combat_conditional_aim_scope import freeze_conditional_aim
        project,trainable_actor_parameters=freeze_conditional_aim(actor,std,actor_opt)
    elif args.actor_head_scope in ('motor-residual','base-only'):
        from combat_motor_residual import freeze_scope
        project,trainable_actor_parameters=freeze_scope(actor,std,actor_opt,args.actor_head_scope=='motor-residual')
    elif args.actor_head_scope=='conditional-fire':
        from combat_conditional_fire import freeze_scope
        project,trainable_actor_parameters=freeze_scope(actor,std,actor_opt)
    credit_stats=None
    if args.critic_credit=='mc-before':
        from combat_critic_credit import fit,reestimate
        actor_before={k:v.detach().clone() for k,v in actor.state_dict().items()}
        std_before=std.detach().clone()
        credit_stats=fit(value,value_opt,compute,ret,config['value_steps'],config['max_grad_norm'],torch,nn)
        assert all(torch.equal(v,actor_before[k]) for k,v in actor.state_dict().items()) and torch.equal(std,std_before)
        adv,baselines,next_baselines=reestimate(rows,prepared,value,config['gamma'],config['lambda'],advantages,torch)
        ad=torch.tensor(adv,device=device)
        credit_stats.update(actor_unchanged=True,baseline_mean=sum(baselines)/len(baselines),
                            next_baseline_mean=sum(next_baselines)/len(next_baselines))
    ad=(ad-ad.mean())/(ad.std(unbiased=False)+1e-8)
    if args.policy_update!='ppo':
        from combat_advantage_regression import weights as advantage_weights
        regression_weights,regression_stats=advantage_weights(ret,torch.tensor(baselines,device=device),
            args.awr_temperature,args.awr_weight_cap,0. if args.policy_update=='bc' else args.awr_strength,torch)
    for _ in range(config['actor_steps']):
        loss,kl=objective()
        if float(kl.detach())>config['target_kl']:break
        actor_opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(parameters,config['max_grad_norm'])
        group_lrs=[actor_training['learning_rate']]+[{'weapon_state_residual':args.weapon_state_lr,'motor_residual':args.motor_lr,'conditional_fire':args.conditional_fire_lr}[group['name']] for group in actor_opt.param_groups[1:]] if len(actor_opt.param_groups)>1 else None
        trial=guarded_actor_step(parameters,actor_opt,objective,loss.detach(),actor_training['learning_rate'],config['target_kl'],project,group_lrs=group_lrs);trials.append(trial)
        if not trial['accepted']:break
        if future_aux is not None:future_aux.head_step(actor)
    if args.critic_credit=='mc-after':
        from combat_critic_credit import fit
        credit_stats=fit(value,value_opt,compute,ret,config['value_steps'],config['max_grad_norm'],torch,nn)
    elif args.critic_credit=='legacy':
        for _ in range(config['value_steps']):
            prediction,_=compute(value);loss=(prediction[:,0]-ret).square().mean();assert torch.isfinite(loss)
            value_opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(value.parameters(),config['max_grad_norm']);value_opt.step()
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    with torch.no_grad():loss,kl=objective();final_bank={s:float(bank_loss(s)) for s in bank_tensors}
    with torch.no_grad():sequence_kl_after=float(sequence_loss(compute(actor)[0])) if sequence_retention else None
    future_after=None if future_aux is None else future_aux.stats(actor)
    if future_aux is not None:future_aux.verify()
    assert torch.isfinite(kl) and float(kl)<=config['target_kl']+1e-6
    base_a,cell_a=actor.export();base_v,cell_v=value.export()
    updated={**model,'actor':base_a,'value':base_v,'log_std':std.detach().cpu().tolist(),key:{**spec,'actor':cell_a,'value':cell_v}}
    from combat_spatial_aim import export as export_spatial
    export_spatial(updated,actor)
    done=sum(t['accepted'] for t in trials)
    for path,digest in meta['source_sha256'].items():assert sha(pathlib.Path(path))==digest
    if not entity:assert sha(args.data/'sequence.jsonl')==meta['sequence_sha256']
    assert sha(args.data/'rollout.jsonl')==meta['rollout_sha256']
    args.out.mkdir(exist_ok=False);durable_json(args.out/'weights.json',updated)
    cp={'version':'combat_architecture_checkpoint_v1','architecture':expected_version,'weights_sha256':sha(args.out/'weights.json'),'config':config,'anchor_sha256':sha(args.anchor_model),'bank_sha256':sha(args.retention_bank),'actor':actor.state_dict(),'value':value.state_dict(),'log_std':std.detach(),'actor_optimizer':actor_opt.state_dict(),'value_optimizer':value_opt.state_dict(),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'consumed_rollouts':consumed+[meta['rollout_sha256']],'updates_completed':updates+1,'total_actor_steps':total+done}
    cp['retention_weights']=[args.retention_weight,args.bank_weight]
    cp['actor_training']=actor_training
    if config_forks:cp['training_config_forks']=config_forks
    if sequence_retention:cp['sequence_retention']=sequence_retention
    if migrations:cp['model_migrations']=migrations
    if future_aux is not None:cp['future_auxiliary']=future_aux.checkpoint()
    if future_forks:cp['future_objective_forks']=future_forks
    cp['critic_credit']=critic_credit
    if critic_forks:cp['critic_credit_forks']=critic_forks
    cp['policy_update']=policy_update
    if policy_update_forks:cp['policy_update_forks']=policy_update_forks
    durable_write(args.out/'checkpoint.pt',lambda output:torch.save(cp,output))
    durable_write(args.out/'trainer.py',lambda output:output.write(pathlib.Path(__file__).read_bytes()))
    report={'architecture':expected_version,'device':'cuda','torch':torch.__version__,'rows':len(rows),'sequence_rows':len(context),'sequences':len(prepared[3]) if prepared else 0,'bptt_steps':bptt if key=='memory' else None,'actor_steps':done,'total_actor_steps':total+done,'updates_completed':updates+1,'actor_trials':trials,'final_approx_kl':float(kl),'old_log_probability_max_error':log_error,'old_value_max_error':value_error,'old_actor_memory_max_error':actor_state_error,'old_value_memory_max_error':value_state_error,'bank_kl_before':initial_bank,'bank_kl_after':final_bank,'seconds':seconds,'weights_sha256':sha(args.out/'weights.json'),'behavior_sha256':sha(args.model),'rollout_sha256':meta['rollout_sha256'],'sequence_sha256':meta.get('sequence_sha256'),'trainer_sha256':sha(pathlib.Path(__file__)),'config_sha256':sha(args.config),'resume_sha256':sha(args.resume) if args.resume else None,'anchor_sha256':sha(args.anchor_model),'bank_sha256':sha(args.retention_bank),'actor_parameters':sum(p.numel() for p in actor.parameters())+std.numel(),'critic_parameters':sum(p.numel() for p in value.parameters()),'scope':'Fresh on-policy architecture PPO. GRU replays all provider context with current-model prefixes, detached every32 frames. Temporal attention is causal/window32; entity attention uses only current v4 values. Loss only on native eligible rows. Bank has no histories: retention measured at zero memory. No learned weapon choice or live promotion.'}
    report['retention_weight']=args.retention_weight;report['bank_weight']=args.bank_weight
    report['weapon_head']=weapon_head
    report['model_migrations']=migrations
    report['sequence_retention']=sequence_retention
    report['sequence_retention_kl_before']=sequence_kl_before
    report['sequence_retention_kl_after']=sequence_kl_after
    report['sequence_retention_objective_fork']=args.fork_sequence_retention
    report['actor_training']=actor_training
    report['actor_scope_fork']=args.fork_head_scope
    report['training_config_forks']=config_forks
    report['trainable_actor_parameters']=trainable_actor_parameters
    report['future_auxiliary']=None if future_aux is None else dict(spec=future_aux.spec,before=future_before,after=future_after)
    report['future_objective_fork']=args.fork_future_aux
    report['initial_boundary_sha256']=initial_boundary_sha
    report['critic_credit']=dict(**critic_credit,stats=credit_stats)
    report['critic_credit_forks']=critic_forks
    report['policy_update']=dict(spec=policy_update,weights=regression_stats)
    report['policy_update_forks']=policy_update_forks
    if model.get('weapon_state_residual'):
        report['scope']+=' Observed weapon-state residual keeps its separate Adam group; old log_std and parameter IDs stay in group1. Both groups backtrack by the same factor. Trainability follows the explicit actor scope.'
        report['weapon_state_residual']=actor_training['weapon_state_residual']
        report['actor_optimizer_group_lrs']=[group['lr'] for group in actor_opt.param_groups]
    if args.actor_head_scope=='target':report['scope']+=' Projected target rows20:29 only; other actor outputs/encoder/spatial/std and their Adam moments preserved. Output tensor Adam clocks advance for target updates; switching scope requires explicit checkpoint fork.'
    if args.actor_head_scope=='decisions':report['scope']+=' Projected attack row4, weapon rows8:20 and target rows20:29 only; movement/vertical/aim/mode/encoder/spatial/std and their Adam moments preserved. Shared output tensor Adam clocks advance; changing scope or its learning rate requires explicit checkpoint fork. Joint on-policy PPO likelihood and masks unchanged.'
    if args.actor_head_scope=='conditional-aim':report['scope']+=' Only sampled-motion conditional aim branch and log_std[2:4] train. All raw actor logits/encoder/spatial/weapon branches and movement variance/moments stay frozen; shared std Adam clock advances. This is an isolated aiming experiment, not full-policy algorithm replacement.'
    if args.actor_head_scope=='movement':report['scope']+=' Only raw horizontal rows0:2, vertical rows5:8 and log_std[:2] train. Encoder/attention, aim/target/fire/weapon parameters and their Adam moments stay frozen; shared tensor Adam clocks advance. Spatial and sampled-motion controllers consume movement means, so frozen aim parameters do not imply frozen conditional aim outputs. Runtime actions remain entirely learned.'
    if model.get('shared_target'):
        report['scope']+=' Explicit shared per-enemy target residual is trained in target/decision scopes; its new Adam states start empty after a sealed checkpoint migration. Availability masks and none output unchanged.'
    if sequence_retention:
        report['scope']+=' Sequence retention uses pinned train-only rows and frozen initial actor on actual full histories; target/weapon masks and all aim branches included. No zero-memory anchor or validation bank.'
    report['scope']=report['scope'].replace('No learned weapon choice or live promotion.','Masked weapon likelihood enabled only with the explicit V6 weapon head; weapon learning acceptance and live promotion are not established.')
    if model.get('sampled_motion_aim'):
        from combat_sampled_motion_aim import VERSION as motion_v1
        vertical_conditioned=model['sampled_motion_aim']['version']!=motion_v1
        report['sampled_motion_aim']=dict(version=model['sampled_motion_aim']['version'],
            joint_likelihood='target_mode_move_vertical_conditional_aim_v2' if vertical_conditioned else 'target_mode_move_conditional_aim_v1',
            conditional_retention_expectation=('fixed_behavior_move_exact_target_mode_vertical_v2' if vertical_conditioned else 'fixed_behavior_move_per_history_v1') if sequence_retention else None)
        report['scope']='Fresh on-policy full-actor/critic CUDA PPO with sampled-motion conditional aim likelihood. Native eligible train rows, actual temporal histories, availability masks and guards as environment execution. Conditional retention uses fixed behavior movement draws. No automatic model promotion or campaign/coop acceptance.'
    if model.get('motor_residual'):
        report['motor_residual']=actor_training['motor_residual']
        report['scope']+=' Observed motor residual uses a separate Adam group and stable base reference for conditional aim.'
        if args.actor_head_scope=='motor-residual':report['scope']+=' Only motor branch parameters train; existing actor/std parameters and Adam states are projected unchanged.'
        if args.actor_head_scope=='base-only':report['scope']+=' Only the original actor/std train; the new motor branch and its Adam states are projected unchanged.'
    if model.get('conditional_fire'):
        report['conditional_fire']={**actor_training['conditional_fire'],
            'joint_likelihood':'fire_after_complete_sampled_action_v1',
            'retention_expectation':sequence_retention['conditional_fire_expectation']}
        report['scope']+=' Fire uses the actual recorded full action; conditional Bernoulli retention uses a fixed behavior action per history, not an exact categorical expectation. No fire mask is imposed.'
        if args.actor_head_scope=='conditional-fire':report['scope']+=' Only the new fire branch trains; old actor/std and Adam states are projected unchanged.'
    durable_json(args.out/'report.json',report)
    durable_json(args.out/'complete.json',{'version':'combat_update_complete_v1','weights_sha256':sha(args.out/'weights.json'),'checkpoint_sha256':sha(args.out/'checkpoint.pt'),'report_sha256':sha(args.out/'report.json')})
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

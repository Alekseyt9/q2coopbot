"""CUDA-only masked aim/fire BC warmstart; native proof is exported by Go."""
import argparse, copy, json, pathlib, time
from ppo_combat import torch, nn, sha
from ppo_recurrent import durable_json, durable_write
from combat_attention import CausalAttention, TEMPORAL

def read(path): return json.loads(pathlib.Path(path).read_text(encoding='utf-8-sig'))

def sequences(rows):
    groups=[];last=None
    for row in rows:
        i=row['identity'];key=(row['seed'],i['connection'],i['map'],i['spawncount'],i['actor'],i['life'])
        if last is None or key!=last[0] or i['frame']!=last[1]+1: groups.append([])
        groups[-1].append(row);last=(key,i['frame'])
    return groups

def tensors(rows,device,require_attack=True):
    groups=sequences(rows);length=max(map(len,groups));width=len(rows[0]['features'])
    x=torch.zeros((len(groups),length,width),device=device);target=torch.zeros((len(groups),length,2),device=device)
    aim=torch.zeros((len(groups),length),device=device,dtype=torch.bool);attack_mask=aim.clone();attack=x[:,:,0].clone();valid=aim.clone()
    for i,group in enumerate(groups):
        n=len(group);x[i,:n]=torch.tensor([r['features'] for r in group],device=device)
        target[i,:n]=torch.tensor([r['targets'][2:4] for r in group],device=device)
        aim[i,:n]=torch.tensor([r['mask'][1] for r in group],device=device)
        attack_mask[i,:n]=torch.tensor([r['mask'][2] for r in group],device=device)
        attack[i,:n]=torch.tensor([float(r['attack']) for r in group],device=device);valid[i,:n]=True
    assert aim.any()
    if require_attack:assert attack_mask.any() and attack[attack_mask].min()==0 and attack[attack_mask].max()==1
    return x,target,aim,attack_mask,attack,valid

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=pathlib.Path,required=True);ap.add_argument('--checkpoint',type=pathlib.Path,required=True);ap.add_argument('--data',type=pathlib.Path,required=True);ap.add_argument('--config',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args()
    assert torch.cuda.is_available(),'CUDA required';device='cuda';config=read(a.config);model=read(a.model);meta=read(a.data/'report.json')
    assert config['version']=='combat_sequence_bc_v1' and 1<=config['epochs']<=2000 and config['retention_weight']>0
    assert meta['version']=='combat_bc_sequence_v1' and meta['test_deferred'] and model['feature_version']==meta['feature_version']
    assert model.get('attention') and not model.get('memory') and not model.get('entity_attention') and model['attention']['version']==TEMPORAL
    rows={}
    for split in ('train','validation'):
        assert sha(a.data/(split+'.jsonl'))==meta['data_sha256'][split]
        rows[split]=[json.loads(s) for s in (a.data/(split+'.jsonl')).read_text().splitlines()]
    assert not {r['seed'] for r in rows['train']}&{r['seed'] for r in rows['validation']}
    torch.set_num_threads(2);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    torch.use_deterministic_algorithms(True)
    spec=model['attention'];actor=CausalAttention(model['actor'],spec['actor'],spec['heads'],spec['window']).to(device)
    cp=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    assert cp['version']=='combat_architecture_checkpoint_v1' and cp['architecture']==TEMPORAL and cp['weights_sha256']==sha(a.model)
    assert all(torch.equal(v.cpu(),cp['actor'][k].cpu()) for k,v in actor.state_dict().items())
    train_scope=config.get('train_scope','full_actor')
    assert train_scope in ('full_actor','aim_fire_heads','aim_heads','fire_head')
    selected={'aim_fire_heads':[2,3,4],'aim_heads':[2,3],'fire_head':[4]}.get(train_scope)
    preserved=[i for i in range(len(model['actor'][-1]['bias'])) if selected is not None and i not in selected]
    original_state={k:v.detach().clone() for k,v in actor.state_dict().items()}
    if selected is not None:
        for p in actor.parameters():p.requires_grad_(False)
        for module in (actor.head,actor.residual):
            for p in module.parameters():
                p.requires_grad_(True)
                mask=torch.zeros_like(p);mask[selected]=1
                p.register_hook(lambda grad,mask=mask:grad*mask)
    aim_loss_kind=config.get('aim_loss','coordinate_mse')
    assert aim_loss_kind in ('coordinate_mse','wrapped_yaw_v1')
    datasets={s:tensors(r,device,train_scope!='aim_heads') for s,r in rows.items()}
    with torch.no_grad(): reference={s:actor(d[0])[0].detach() for s,d in datasets.items()}
    # Native teacher tracks aim/fire. Other heads retain this branch's behavior;
    # full mixed live evaluation is still needed to detect forgetting.
    cols=[0,1]+list(range(5,len(model['actor'][-1]['bias'])))
    def losses(split):
        x,target,aim,am,attack,valid=datasets[split];raw=actor(x)[0]
        error=raw[... ,2:4].tanh()[aim]-target[aim]
        if aim_loss_kind=='wrapped_yaw_v1':error=torch.stack(((error[:,0]+1).remainder(2)-1,error[:,1]),-1)
        aim_loss=error.square().mean()
        logits=raw[...,4][am];labels=attack[am];pos=labels==1;neg=~pos
        bce=nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='none')
        attack_loss=.5*(bce[pos].mean()+bce[neg].mean()) if pos.any() and neg.any() else raw.sum()*0
        retention=(raw[...,cols][valid]-reference[split][...,cols][valid]).square().mean()
        loss=config['aim_weight']*aim_loss+config['attack_weight']*attack_loss+config['retention_weight']*retention
        return loss,dict(aim_rmse_degrees=float(aim_loss.detach().sqrt()*180),attack_balanced_bce=float(attack_loss.detach()) if pos.any() and neg.any() else None,attack_rows=int(am.sum()),retention_mse=float(retention.detach()))
    opt=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],lr=config['learning_rate']);history=[];start=time.perf_counter()
    with torch.no_grad(): before={s:losses(s)[1] for s in datasets}
    for epoch in range(config['epochs']):
        opt.zero_grad();loss,_=losses('train');assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),1.);opt.step()
        if epoch%25==0 or epoch+1==config['epochs']:
            with torch.no_grad():history.append(dict(epoch=epoch+1,**{s:losses(s)[1] for s in datasets}))
    with torch.no_grad(): after={s:losses(s)[1] for s in datasets}
    if selected is not None:
        for key,value in actor.state_dict().items():
            if key.startswith(('head.','residual.')):
                assert torch.equal(value[preserved],original_state[key][preserved]),'Unselected action rows changed'
            else:assert torch.equal(value,original_state[key]),'Frozen encoder/attention changed'
        with torch.no_grad():
            for split,d in datasets.items():
                assert torch.equal(actor(d[0])[0][...,preserved],reference[split][...,preserved]),'Unselected action outputs changed'
    base,cell=actor.export();updated=copy.deepcopy(model);updated['actor']=base;updated['attention']['actor']=cell
    assert updated['value']==model['value'] and updated['attention']['value']==model['attention']['value'] and updated['log_std']==model['log_std']
    for split in rows:assert sha(a.data/(split+'.jsonl'))==meta['data_sha256'][split]
    a.out.mkdir(exist_ok=False);durable_json(a.out/'weights.json',updated)
    std=torch.tensor(model['log_std'],device=device,requires_grad=True)
    for p in actor.parameters():p.requires_grad_(True)
    # PPO resumes a changed policy with a fresh actor optimizer. Critic state,
    # consumed rollout hashes and PPO experience counters remain intact.
    ppo_opt=torch.optim.Adam(list(actor.parameters())+[std],lr=cp['config']['actor_lr'])
    cp['actor']=actor.state_dict();cp['actor_optimizer']=ppo_opt.state_dict();cp['weights_sha256']=sha(a.out/'weights.json')
    cp['rng']=torch.get_rng_state();cp['cuda_rng']=torch.cuda.get_rng_state_all()
    migration=dict(kind='masked_sequence_bc',train_scope=train_scope,parent_weights_sha256=sha(a.model),parent_checkpoint_sha256=sha(a.checkpoint),data_sha256=meta['data_sha256'],config_sha256=sha(a.config),epochs=config['epochs'],actor_optimizer='reset',critic='preserved')
    cp['model_migrations']=cp.get('model_migrations',[])+[migration]
    durable_write(a.out/'checkpoint.pt',lambda f:torch.save(cp,f))
    report=dict(version='combat_sequence_bc_update_v1',device=device,architecture=TEMPORAL,epochs=config['epochs'],train_context=meta['counts']['train'],validation_context=meta['counts']['validation'],train_aim_rows=meta['aim_rows']['train'],validation_aim_rows=meta['aim_rows']['validation'],before=before,after=after,history=history,seconds=time.perf_counter()-start,weights_sha256=sha(a.out/'weights.json'),checkpoint_sha256=sha(a.out/'checkpoint.pt'),parent=sha(a.model),data_sha256=meta['data_sha256'],config_sha256=sha(a.config),trainer_sha256=sha(pathlib.Path(__file__)),ppo_updates_completed=cp['updates_completed'],ppo_total_actor_steps=cp['total_actor_steps'],scope='Masked aim/fire BC on verified teacher sequences; fixed final epoch; final test deferred. No live acceptance, architecture superiority, tactical movement or weapon-choice claim.')
    report['train_scope']=train_scope
    report['aim_loss']=aim_loss_kind
    report['non_aim_fire_parameters_exactly_preserved']=selected is not None
    report['selected_action_rows']=selected
    report['unselected_parameters_exactly_preserved']=selected is not None
    durable_json(a.out/'report.json',report);durable_json(a.out/'complete.json',dict(version='combat_update_complete_v1',weights_sha256=report['weights_sha256'],checkpoint_sha256=report['checkpoint_sha256'],report_sha256=sha(a.out/'report.json')))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

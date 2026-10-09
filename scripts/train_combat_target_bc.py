"""CUDA-only bootstrap of new target/aim rows; shared combat rows stay fixed."""
import argparse,copy,math,pathlib,time,gzip,json
from process_combat_architecture_pool import read,sha,save
from prepare_combat_target_heads import build,forward
from ppo_combat import torch,layers
from train_combat_sequence_bc import sequences
from combat_target_head import availability,distribution,selected_means,validate_model

def annotations(x):
    assert x.is_cuda and x.shape[-1]==854
    n=len(x);points=x[:,73:169].reshape(n,8,12)[:,:,2:5]*512
    velocity=x[:,73:169].reshape(n,8,12)[:,:,6:9]*400
    known_velocity=x[:,73:169].reshape(n,8,12)[:,:,5]==1
    box=x[:,466:786].reshape(n,8,40)[:,:,36:40]
    bottom,top=box[:,:,2]*64,box[:,:,3]*64
    height=torch.where(top-bottom<16,(top+bottom)/2,torch.minimum(torch.maximum(torch.full_like(bottom,22),bottom+8),top-8))
    points=points.clone();points[:,:,2]+=height-torch.where(x[:,3:4]==1,-2.,22.)
    aa=velocity.square().sum(-1)-1000000;bb=2*(points*velocity).sum(-1);cc=points.square().sum(-1)
    disc=bb.square()-4*aa*cc
    # Speeds near projectile speed use the linear equation; no guessed root.
    denom=torch.where(aa.abs()>1e-6,2*aa,torch.ones_like(aa))
    r1=(-bb-disc.clamp_min(0).sqrt())/denom;r2=(-bb+disc.clamp_min(0).sqrt())/denom
    inf=torch.full_like(r1,float('inf'));t=torch.minimum(torch.where(r1>0,r1,inf),torch.where(r2>0,r2,inf))
    linear=-cc/torch.where(bb.abs()>1e-6,bb,torch.ones_like(bb))
    t=torch.where(aa.abs()<1e-6,torch.where((bb.abs()>1e-6)&(linear>0),linear,inf),t)
    lead=(disc>=0)&known_velocity&(t<=2)&(x[:,17:18]==1)
    moved=points+velocity*torch.where(lead,t,torch.zeros_like(t))[:,:,None]
    yaw=torch.atan2(moved[:,:,1],moved[:,:,0]);pitch=-torch.atan2(moved[:,:,2],moved[:,:,:2].norm(dim=-1))-torch.atan2(x[:,8:9],x[:,9:10])
    angles=torch.stack((yaw,torch.atan2(pitch.sin(),pitch.cos())),2)/math.pi
    valid=availability(x)[:,1:];label_mask=valid&lead
    # Bootstrap nearest visible target. This is a heuristic query, not an
    # optimal kill-order label; future own-policy target choices supply history.
    target=valid.long().argmax(1)+1;target=torch.where(valid.any(1),target,torch.zeros_like(target))
    previous=x[:,845:854].argmax(1)
    previous_valid=availability(x).gather(1,previous[:,None])[:,0] & (previous>0)
    target=torch.where(previous_valid,previous,target)
    return angles,label_mask,target

def prepare(rows):
    groups=sequences(rows);x=torch.zeros(len(groups),max(map(len,groups)),854,device='cuda')
    eligible=torch.zeros(x.shape[:2],dtype=torch.bool,device='cuda')
    for i,group in enumerate(groups):
        base=torch.tensor([r['features'] for r in group],device='cuda');assert base.shape[1] in (845,854)
        x[i,:len(group),:base.shape[1]]=base
        if base.shape[1]==845:x[i,:len(group),845]=1
        eligible[i,:len(group)]=torch.tensor([r['mask'][1] for r in group],dtype=torch.bool,device='cuda')
    flat=x.reshape(-1,854);angles,mask,target=annotations(flat)
    mask &= eligible.flatten()[:,None]
    choice=eligible.flatten() & (target>0)
    assert mask.any() and choice.any()
    return x,angles,mask,target,choice

def loss(actor,data):
    x,angles,mask,target,choice=data;raw=forward(actor,x).reshape(-1,45)
    predicted=raw[:,29:45].reshape(-1,8,2).tanh()
    aim=(predicted[mask]-angles[mask]).square().mean()
    selector=-distribution(raw,x.reshape(-1,854)).log_prob(target)[choice].mean()
    return aim+.1*selector,dict(aim_rmse_degrees=float(aim.detach().sqrt()*180),selector_loss=float(selector.detach()),aim_labels=int(mask.sum()),target_labels=int(choice.sum()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',type=pathlib.Path,required=True);ap.add_argument('--data',type=pathlib.Path,required=True);ap.add_argument('--epochs',type=int,default=50)
    ap.add_argument('--names',default='');ap.add_argument('--train-intent',action='store_true');a=ap.parse_args()
    assert torch.cuda.is_available() and 1<=a.epochs<=2000
    torch.set_num_threads(2);torch.manual_seed(20261009);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    meta=read(a.data/'report.json');assert (meta['version'],meta['feature_version']) in [('combat_bc_sequence_v1','combat_features_v6'),('combat_target_sequence_v1','combat_features_v7')] and meta['test_deferred']
    rows={}
    for split in ('train','validation'):
        compressed=meta['version']=='combat_target_sequence_v1'
        path=a.data/(split+('.jsonl.gz' if compressed else '.jsonl'));assert sha(path)==meta['data_sha256'][split]
        with gzip.open(path,'rt',encoding='utf-8') if compressed else path.open(encoding='utf-8') as stream:
            rows[split]=[json.loads(s) for s in stream]
    assert not {r['seed'] for r in rows['train']} & {r['seed'] for r in rows['validation']}
    data={split:prepare(values) for split,values in rows.items()};results=[]
    paths=[a.models/name/'weights.json' for name in a.names.split(',')] if a.names else sorted(a.models.glob('m[0-7]/weights.json'))
    assert paths
    for path in paths:
        out=path.parent/'target-bc';out.mkdir(exist_ok=False);model=read(path);validate_model(model);actor=build(model,'actor')
        original={k:v.detach().clone() for k,v in actor.state_dict().items()}
        for p in actor.parameters():p.requires_grad_(False)
        head=actor.head if hasattr(actor,'head') else actor[-1]
        trainable=[head]
        if hasattr(actor,'output'):trainable.append(actor.output)
        if hasattr(actor,'residual'):trainable.append(actor.residual)
        for module in trainable:
            for p in module.parameters():p.requires_grad_(True)
        first=actor.encoder[0] if hasattr(actor,'encoder') else actor[0]
        reference=None
        if a.train_intent:
            assert meta['feature_version']=='combat_features_v7'
            first.weight.requires_grad_(True)
            with torch.no_grad():reference=forward(actor,data['train'][0])[...,:20].detach()
        opt=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],lr=.001)
        with torch.no_grad():before={s:loss(actor,v)[1] for s,v in data.items()}
        start=time.perf_counter()
        for _ in range(a.epochs):
            opt.zero_grad();objective,_=loss(actor,data['train'])
            if reference is not None:objective=objective+100*(forward(actor,data['train'][0])[...,:20]-reference).square().mean()
            objective.backward()
            if a.train_intent:first.weight.grad[:,:846]=0
            for module in trainable:
                for p in module.parameters():p.grad[:20]=0
            opt.step()
        for key,value in actor.state_dict().items():
            prefix='4.' if not hasattr(actor,'head') else 'head.'
            if key.startswith((prefix,'output.','residual.')):assert torch.equal(value[:20],original[key][:20])
            elif a.train_intent and key==('encoder.0.weight' if hasattr(actor,'encoder') else '0.weight'):
                assert torch.equal(value[:,:846],original[key][:,:846])
            else:assert torch.equal(value,original[key])
        with torch.no_grad():after={s:loss(actor,v)[1] for s,v in data.items()}
        updated=copy.deepcopy(model)
        if model.get('memory'):updated['actor'],cell=actor.export();updated['memory']['actor'].update(cell)
        elif model.get('attention'):updated['actor'],cell=actor.export();updated['attention']['actor'].update(cell)
        else:updated['actor']=layers(actor)
        save(out/'weights.json',updated)
        torch.save(dict(version='combat_target_bc_checkpoint_v1',actor=actor.state_dict(),optimizer=opt.state_dict(),weights_sha256=sha(out/'weights.json'),parent_sha256=sha(path),epochs=a.epochs),out/'checkpoint.pt')
        drift=None
        if reference is not None:
            with torch.no_grad():drift=float((forward(actor,data['train'][0])[...,:20]-reference).square().mean().sqrt())
        report=dict(model=path.parent.name,device='cuda',epochs=a.epochs,before=before,after=after,seconds=time.perf_counter()-start,parent_sha256=sha(path),weights_sha256=sha(out/'weights.json'),checkpoint_sha256=sha(out/'checkpoint.pt'),data_report_sha256=sha(a.data/'report.json'),data_sha256=meta['data_sha256'],shared_parameters_preserved=not a.train_intent,old_feature_columns_and_common_rows_preserved=True,intent_columns_trained=a.train_intent,common_raw_drift_rmse=drift,trainer_sha256=sha(pathlib.Path(__file__)),optimizer='new BC optimizer; inherited PPO optimizer not resumed',scope='Unexecuted observed blaster intercept queries; nearest visible/retain observed previous target bootstrap. Eye origin approximation; machinegun recoil unknown and aim labels excluded. Context and masks retained. Optional intent columns can change common outputs: raw-output penalty100 bounds training drift, no live preservation guarantee. No tactical optimality, hit credit or native quality claim.')
        save(out/'report.json',report);results.append(report);print(path.parent.name,after['validation'],flush=True)
    save(a.models/'target-bc-report.json',dict(state='complete',device='cuda',models=results))

if __name__=='__main__':main()

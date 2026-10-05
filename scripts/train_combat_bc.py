"""Offline PyTorch BC only. No game, socket, native reward, or second harness."""
import argparse, hashlib, json, pathlib, time, os
# Every generated cache/temp path is on the repository's F: workspace.
cache=pathlib.Path(__file__).resolve().parents[1]/'workspace'/'build'/'training-cache'
cache.mkdir(parents=True,exist_ok=True)
for key in ['TMP','TEMP','TMPDIR','PIP_CACHE_DIR','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','XDG_CACHE_HOME']:
    path=cache/key.lower();path.mkdir(exist_ok=True);os.environ[key]=str(path)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
from torch import nn

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--config',required=True);a=ap.parse_args()
    root=pathlib.Path(a.data);out=pathlib.Path(a.out);out.mkdir(exist_ok=False)
    config=json.loads(pathlib.Path(a.config).read_text(encoding='utf-8-sig'))
    assert config['version']=='combat_bc_training_v1' and config['hidden']==[64,64] and config['checkpoint']=='fixed_final_epoch' and config['weapon_head'] is False
    assert 1<=config['epochs']<=10000 and 0<config['learning_rate']<=0.01
    torch.set_num_threads(2);torch.manual_seed(config['seed']);torch.use_deterministic_algorithms(True)
    input_hash={s:sha(root/(s+'.jsonl')) for s in ['train','validation','test']}
    splits={s:[json.loads(line) for line in (root/(s+'.jsonl')).read_text().splitlines()] for s in input_hash}
    sets={s:set(r['seed'] for r in rows) for s,rows in splits.items()}
    assert not sets['train']&sets['validation'] and not sets['train']&sets['test'] and not sets['validation']&sets['test']
    def tensors(rows):
        return (torch.tensor([r['features'] for r in rows],dtype=torch.float32),torch.tensor([r['targets'] for r in rows],dtype=torch.float32),torch.tensor([r['mask'] for r in rows],dtype=torch.bool),torch.tensor([r['attack'] for r in rows],dtype=torch.float32),torch.tensor([r['vertical'] for r in rows],dtype=torch.long))
    datasets={s:tensors(rows) for s,rows in splits.items()};x,y,m,attack,vertical=datasets['train']
    def make_model(): return nn.Sequential(nn.Linear(x.shape[1],64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,8))
    attack_y=attack[m[:,2]];pos=attack_y.sum();neg=len(attack_y)-pos;assert pos>0 and neg>0
    pos_weight=neg/pos
    vcounts=torch.bincount(vertical[m[:,3]],minlength=3);assert (vcounts>0).all();vweight=vcounts.sum()/(3*vcounts.float())
    def loss(raw,y,m,attack,vertical):
        terms=[]
        for head,cols in [(0,slice(0,2)),(1,slice(2,4))]:
            if m[:,head].any():terms.append(((raw[m[:,head],cols].tanh()-y[m[:,head],cols])**2).mean())
        if m[:,2].any():terms.append(nn.functional.binary_cross_entropy_with_logits(raw[m[:,2],4],attack[m[:,2]],pos_weight=pos_weight.to(raw.device)))
        if m[:,3].any():terms.append(nn.functional.cross_entropy(raw[m[:,3],5:],vertical[m[:,3]],weight=vweight.to(raw.device)))
        return sum(terms)
    benchmark={}
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        probe=make_model().to(device);px,py,pm,pa,pv=[t.to(device) for t in datasets['train']];probe_opt=torch.optim.Adam(probe.parameters(),lr=config['learning_rate'])
        def probe_step():
            probe_opt.zero_grad();loss(probe(px),py,pm,pa,pv).backward();probe_opt.step()
        for _ in range(10):probe_step()
        if device=='cuda':torch.cuda.synchronize()
        started=time.perf_counter()
        for _ in range(50):probe_step()
        if device=='cuda':torch.cuda.synchronize()
        benchmark[device]=(time.perf_counter()-started)/50
    device=min(benchmark,key=benchmark.get)
    torch.manual_seed(config['seed']);model=make_model().to(device);datasets={s:tuple(t.to(device) for t in ds) for s,ds in datasets.items()};x,y,m,attack,vertical=datasets['train']
    opt=torch.optim.Adam(model.parameters(),lr=config['learning_rate']);start=time.perf_counter()
    initial=float(loss(model(x),y,m,attack,vertical).detach())
    for _ in range(config['epochs']):
        opt.zero_grad();objective=loss(model(x),y,m,attack,vertical);assert torch.isfinite(objective);objective.backward();opt.step()
    if device=='cuda':torch.cuda.synchronize()
    seconds=time.perf_counter()-start;model.eval();report={'torch':torch.__version__,'device':device,'benchmark_seconds_per_step':benchmark,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'config':config,'config_sha256':sha(pathlib.Path(a.config)),'data_sha256':input_hash,'training_seconds':seconds,'initial_train_loss':initial,'attack_positive_weight':float(pos_weight),'vertical_class_weights':vweight.tolist(),'metrics':{},'scope':'Masked imitation pilot, not battle improvement or PPO; test evaluated once after fixed final epoch'}
    references=[]
    with torch.no_grad():
        for split,(sx,sy,sm,sa,sv) in datasets.items():
            raw=model(sx);metrics={'rows':len(sx),'masked_loss':float(loss(raw,sy,sm,sa,sv))}
            coverage={}
            for row in splits[split]:
                weapon=row['observation']['weapon'];counts=coverage.setdefault(weapon,{'rows':0,'movement':0,'aim':0,'attack':0,'vertical':0})
                counts['rows']+=1
                for index,head in enumerate(['movement','aim','attack','vertical']):counts[head]+=int(row['mask'][index])
            metrics['head_coverage_by_weapon']=coverage
            for head,cols,name,scale in [(0,slice(0,2),'movement_mae',1),(1,slice(2,4),'aim_mae_degrees',180)]:
                metrics[name]=float((raw[sm[:,head],cols].tanh()-sy[sm[:,head],cols]).abs().mean()*scale)
            predicted=raw[:,4]>=0;am=sm[:,2];metrics['attack_accuracy']=float((predicted[am]==sa[am].bool()).float().mean());metrics['attack_positive_recall']=float(predicted[am & (sa==1)].float().mean());metrics['attack_negative_recall']=float((~predicted[am & (sa==0)]).float().mean());metrics['vertical_accuracy']=float((raw[sm[:,3],5:].argmax(1)==sv[sm[:,3]]).float().mean());report['metrics'][split]=metrics
            references.extend({'observation':row['observation'],'raw':r.tolist()} for row,r in zip(splits[split],raw))
    layers=[{'weight':l.weight.detach().tolist(),'bias':l.bias.detach().tolist()} for l in model if isinstance(l,nn.Linear)]
    (out/'weights.json').write_text(json.dumps({'kind':'combat_bc_mlp_v1','feature_version':'combat_features_v1','layers':layers},allow_nan=False))
    report['weights_sha256']=sha(out/'weights.json');torch.save(model.state_dict(),out/'checkpoint.pt')
    (out/'reference.jsonl').write_text('\n'.join(json.dumps(r,allow_nan=False) for r in references)+'\n')
    assert input_hash=={s:sha(root/(s+'.jsonl')) for s in input_hash}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report,indent=2))
if __name__=='__main__':main()

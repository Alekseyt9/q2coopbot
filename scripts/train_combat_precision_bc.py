"""CUDA bootstrap of fine aim and mode rows; all 45 parent rows stay fixed."""
import argparse,copy,gzip,json,pathlib,time
from process_combat_architecture_pool import read,save,sha
from train_combat_target_bc import prepare
from prepare_combat_target_heads import build,forward
from ppo_combat import torch,layers
from combat_precision_head import migrate,query_modes,validate_precision_model

def loss(actor,data):
    x,angles,mask,target,choice=data
    raw=forward(actor,x).reshape(-1,81)
    modes=query_modes(angles)
    fine_mask=mask & (modes==1)
    fine=raw[:,47:63].reshape(-1,8,2).tanh()*15
    assert fine_mask.any()
    aim=(fine[fine_mask]-angles[fine_mask]*180).square().mean()/225
    logits=raw[:,65:81].reshape(-1,8,2)
    stage=torch.nn.functional.cross_entropy(logits[mask],modes[mask])
    objective=aim+.1*stage
    return objective,dict(fine_rmse_degrees=float((aim.detach()*225).sqrt()),mode_accuracy=float((logits[mask].argmax(-1)==modes[mask]).float().mean()),fine_labels=int(fine_mask.sum()),mode_labels=int(mask.sum()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=pathlib.Path,required=True);ap.add_argument('--data',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);ap.add_argument('--epochs',type=int,default=100);a=ap.parse_args()
    assert torch.cuda.is_available() and 1<=a.epochs<=2000
    torch.set_num_threads(2);torch.manual_seed(20261010)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    meta=read(a.data/'report.json');assert meta['version']=='combat_target_sequence_v1' and meta['test_deferred']
    rows={}
    for split in ('train','validation'):
        path=a.data/(split+'.jsonl.gz');assert sha(path)==meta['data_sha256'][split]
        with gzip.open(path,'rt',encoding='utf-8') as f:rows[split]=[json.loads(s) for s in f]
    assert not {r['seed'] for r in rows['train']} & {r['seed'] for r in rows['validation']}
    data={s:prepare(r) for s,r in rows.items()}
    parent=read(a.model);model=migrate(parent);actor=build(model,'actor');validate_precision_model(model)
    original={k:v.detach().clone() for k,v in actor.state_dict().items()}
    with torch.no_grad():
        for name in ('actor','value'):
            old=forward(build(parent,name),data['validation'][0]);new=forward(build(model,name),data['validation'][0])
            assert torch.equal(old,new[...,:old.shape[-1]])
    for p in actor.parameters():p.requires_grad_(False)
    head=actor.head if hasattr(actor,'head') else actor[-1]
    trainable=[head]+[getattr(actor,n) for n in ('output','residual') if hasattr(actor,n)]
    allowed={id(p) for m in trainable for p in m.parameters()}
    for m in trainable:
        for p in m.parameters():p.requires_grad_(True)
    opt=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],lr=.003)
    with torch.no_grad():before={s:loss(actor,d)[1] for s,d in data.items()}
    start=time.perf_counter()
    for _ in range(a.epochs):
        opt.zero_grad();objective,_=loss(actor,data['train']);objective.backward()
        for m in trainable:
            for p in m.parameters():p.grad[:45]=0
        opt.step()
    for name,p in actor.named_parameters():
        assert torch.equal(p[:45] if id(p) in allowed else p,original[name][:45] if id(p) in allowed else original[name])
    with torch.no_grad():after={s:loss(actor,d)[1] for s,d in data.items()}
    updated=copy.deepcopy(model)
    if model.get('memory'):updated['actor'],cell=actor.export();updated['memory']['actor'].update(cell)
    elif model.get('attention'):updated['actor'],cell=actor.export();updated['attention']['actor'].update(cell)
    else:updated['actor']=layers(actor)
    a.out.mkdir(exist_ok=False);save(a.out/'before-weights.json',model);save(a.out/'weights.json',updated)
    torch.save(dict(version='combat_precision_bc_checkpoint_v1',actor=actor.state_dict(),optimizer=opt.state_dict(),weights_sha256=sha(a.out/'weights.json'),parent_sha256=sha(a.model),epochs=a.epochs),a.out/'checkpoint.pt')
    report=dict(device='cuda',gpu=torch.cuda.get_device_name(),epochs=a.epochs,seconds=time.perf_counter()-start,before=before,after=after,parent_sha256=sha(a.model),weights_sha256=sha(a.out/'weights.json'),checkpoint_sha256=sha(a.out/'checkpoint.pt'),data_report_sha256=sha(a.data/'report.json'),trainer_sha256=sha(pathlib.Path(__file__)),parent_parameters_preserved=True,scope='Only new fine/mode rows trained. Observed unexecuted blaster intercept labels; fine when both corrections <=15 degrees. No machinegun recoil labels, no hit-rate or native quality claim. All previous 45 rows and encoders frozen. New BC optimizer.')
    save(a.out/'report.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

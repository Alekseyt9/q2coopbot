"""CUDA-only BC of a shared spatial branch; parent actor/value remain fixed."""
import argparse,copy,gzip,json,pathlib,time
from process_combat_architecture_pool import read,save,sha
from train_combat_target_bc import prepare
from prepare_combat_target_heads import build,forward
from combat_precision_head import migrate,query_modes
from combat_spatial_aim import initialize,inputs,export,validate
from train_combat_bc import torch

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=pathlib.Path,required=True);ap.add_argument('--data',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);ap.add_argument('--epochs',type=int,default=500);a=ap.parse_args()
    assert torch.cuda.is_available() and 1<=a.epochs<=3000;torch.set_num_threads(2);torch.manual_seed(20261010)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    meta=read(a.data/'report.json');assert meta['version']=='combat_target_sequence_v1' and meta['test_deferred']
    rows={}
    for split in ('train','validation'):
        path=a.data/(split+'.jsonl.gz');assert sha(path)==meta['data_sha256'][split]
        with gzip.open(path,'rt',encoding='utf-8') as f:rows[split]=[json.loads(s) for s in f]
    assert not {r['seed'] for r in rows['train']} & {r['seed'] for r in rows['validation']}
    model=initialize(migrate(read(a.model)));actor=build(model,'actor');validate(model)
    for p in actor.base.parameters():p.requires_grad_(False)
    original={k:v.detach().clone() for k,v in actor.base.state_dict().items()};data={}
    for split,values in rows.items():
        x,angles,mask,_,_=prepare(values,meta['target_query_version'])
        with torch.no_grad():
            raw=forward(actor.base,x).reshape(-1,81);flat=x.reshape(-1,854)
            input_tensor=inputs(flat,raw)[mask];mode_bias=raw[:,65:81].reshape(-1,8,2)[mask]
            assert torch.equal(forward(actor,x).reshape(-1,81),raw),'Initial branch changed parent'
        desired=angles[mask]*180;mode=query_modes(angles)[mask];assert (mode==1).any()
        data[split]=(input_tensor,desired,mode,mode_bias)
    def loss(split):
        x,desired,mode,bias=data[split];prediction=actor.branch(x);fine_mask=mode==1
        mse=(prediction[fine_mask,:2].tanh()*15-desired[fine_mask]).square().mean()/225
        logits=bias+prediction[:,2:];ce=torch.nn.functional.cross_entropy(logits,mode)
        return mse+.1*ce,dict(fine_rmse_degrees=float((mse.detach()*225).sqrt()),mode_accuracy=float((logits.argmax(-1)==mode).float().mean()),fine_labels=int(fine_mask.sum()),mode_labels=len(mode))
    with torch.no_grad():before={s:loss(s)[1] for s in data}
    opt=torch.optim.Adam(actor.branch.parameters(),lr=.003);start=time.perf_counter()
    for _ in range(a.epochs):opt.zero_grad();objective,_=loss('train');objective.backward();opt.step()
    assert all(torch.equal(v,original[k]) for k,v in actor.base.state_dict().items())
    with torch.no_grad():after={s:loss(s)[1] for s in data}
    updated=copy.deepcopy(model);export(updated,actor);validate(updated)
    a.out.mkdir(exist_ok=False);save(a.out/'before-weights.json',model);save(a.out/'weights.json',updated)
    torch.save(dict(version='combat_spatial_aim_bc_checkpoint_v1',actor=actor.state_dict(),optimizer=opt.state_dict(),weights_sha256=sha(a.out/'weights.json'),parent_sha256=sha(a.model),epochs=a.epochs),a.out/'checkpoint.pt')
    report=dict(device='cuda',gpu=torch.cuda.get_device_name(),epochs=a.epochs,seconds=time.perf_counter()-start,before=before,after=after,branch_parameters=sum(p.numel() for p in actor.branch.parameters()),parent_parameters_preserved=True,target_query_version=meta['target_query_version'],parent_sha256=sha(a.model),weights_sha256=sha(a.out/'weights.json'),checkpoint_sha256=sha(a.out/'checkpoint.pt'),data_report_sha256=sha(a.data/'report.json'),trainer_sha256=sha(pathlib.Path(__file__)),spatial_helper_sha256=sha(pathlib.Path(__file__).with_name('combat_spatial_aim.py')),scope='Shared119/64/32/4 spatial residual for eight observed targets. Only branch trained; all parent actor/value/std preserved. Inputs current observations and planned mean movement, not actual sampled displacement or future values. BC queries, no tactical optimality/hit/quality claim; no machinegun recoil labels.')
    save(a.out/'report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()

"""Held-out query audit using the model's chosen mode, exclusively on CUDA."""
import argparse,gzip,json,pathlib
from process_combat_architecture_pool import read,save,sha
from prepare_combat_target_heads import build
from train_combat_machinegun_aim import prepare
from train_combat_bc import torch

def metrics(actor,data):
    spatial,coarse,fine,mode,desired,machinegun=data
    with torch.no_grad():
        delta=actor.branch(spatial);choice=(mode+delta[:,2:4]).argmax(-1)
        predicted=torch.where(choice[:,None]==1,(fine+delta[:,:2]).tanh()*15,(coarse+delta[:,4:6]).tanh()*180)
        error=(predicted-desired+180).remainder(360)-180
        return {name:dict(pairs=int(mask.sum()),rmse_degrees=float(error[mask].square().mean().sqrt()),above_10_degrees=float((error[mask].abs().amax(-1)>10).float().mean()),fine_fraction=float((choice[mask]==1).float().mean())) for name,mask in [('machinegun',machinegun),('blaster',~machinegun)]}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',type=pathlib.Path,required=True);ap.add_argument('--training',type=pathlib.Path,required=True);ap.add_argument('--data',type=pathlib.Path,required=True);a=ap.parse_args()
    assert torch.cuda.is_available();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    report=read(a.training/'report.json');assert report['state']=='complete' and report['device']=='cuda'
    meta=read(a.data/'report.json');path=a.data/'validation.jsonl.gz';assert sha(path)==meta['data_sha256']['validation']
    with gzip.open(path,'rt',encoding='utf-8') as stream:rows=[json.loads(line) for line in stream]
    results=[]
    for entry in report['models']:
        measurements={}
        for label,folder,digest in [('before',a.models,entry['parent_sha256']),('after',a.training,entry['weights_sha256'])]:
            path=folder/entry['model']/'weights.json';assert sha(path)==digest
            actor=build(read(path),'actor');actor.eval()
            data=prepare(actor,rows);measurements[label]=metrics(actor,data)
        results.append(dict(model=entry['model'],architecture=entry['architecture'],**measurements))
        print(entry['model'],json.dumps(measurements),flush=True)
    save(a.training/'chosen-mode-cuda-audit.json',dict(state='passed',device='cuda',models=results,training_report_sha256=sha(a.training/'report.json'),data_report_sha256=sha(a.data/'report.json'),audit_sha256=sha(pathlib.Path(__file__)),scope='Development held-out labels with actual model argmax coarse/fine mode, not oracle mode. No target-selector optimality, sampled-action likelihood, collision or native win-rate claim; live evaluation separately required.'))

if __name__=='__main__':main()

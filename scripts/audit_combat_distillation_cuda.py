"""Offline CUDA action agreement, on held-out sequences, with frozen FireBC."""
import argparse, json, pathlib
from prepare_combat_architecture_priors import build, output
from train_combat_sequence_bc import read, tensors, sequences
from ppo_combat import torch, sha
from combat_weapon_head import feature_mask
from ppo_recurrent import durable_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parent',type=pathlib.Path,required=True)
    ap.add_argument('--data',type=pathlib.Path,required=True);ap.add_argument('--priors',type=pathlib.Path,nargs='+',required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True);args=ap.parse_args()
    assert torch.cuda.is_available(),'CUDA required';torch.set_num_threads(2)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    parent=read(args.parent);meta=read(args.data/'report.json');path=args.data/'validation.jsonl'
    assert sha(path)==meta['data_sha256']['validation']
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()]
    x,*rest=tensors(rows,'cuda',require_attack=False);valid=rest[4];mask=feature_mask(x[valid])
    with torch.no_grad():teacher=output(build(parent,'actor'),x)[valid]
    variants=[]
    for folder in args.priors:
        for model_path in sorted(folder.glob('*/weights.json')):
            model=read(model_path);assert model['feature_version']==parent['feature_version']
            with torch.no_grad():raw=output(build(model,'actor'),x)[valid]
            diff=raw[:,2:4].tanh()-teacher[:,2:4].tanh();diff[:,0]=(diff[:,0]+1).remainder(2)-1
            sl=raw[:,8:20].masked_fill(~mask,-1e9);tl=teacher[:,8:20].masked_fill(~mask,-1e9)
            variants.append({'prior':str(model_path),'weights_sha256':sha(model_path),
                'movement_rmse':float((raw[:,:2].tanh()-teacher[:,:2].tanh()).square().mean().sqrt()),
                'aim_rmse_degrees':float(diff.square().mean().sqrt()*180),
                'aim_error_p95_degrees':float((diff.abs()*180).flatten().quantile(.95)),
                'fire_disagreement':float(((raw[:,4]>=0)!=(teacher[:,4]>=0)).float().mean()),
                'pose_disagreement':float((raw[:,5:8].argmax(-1)!=teacher[:,5:8].argmax(-1)).float().mean()),
                'weapon_disagreement':float((sl.argmax(-1)!=tl.argmax(-1)).float().mean())})
    report={'version':'combat_distillation_cuda_audit_v1','device':'cuda','validation_rows':len(rows),'segments':len(sequences(rows)),
            'parent_sha256':sha(args.parent),'data_sha256':sha(path),'variants':variants,
            'scope':'Offline action agreement on validation contexts, zero state at identity/frame gaps. No new gradients, no Go replay, no live quality claim.'}
    durable_json(args.out,report);print(json.dumps(report))

if __name__=='__main__':main()

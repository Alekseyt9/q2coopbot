"""CUDA-only real-history diagnostic of an observed-Grenades exploration gate.

Does not change weights, sample new actions, or train on evaluation histories.
Temperature preserves argmax; only fresh stochastic training can change choices.
"""
import argparse
import json
import pathlib

from ppo_recurrent import torch,read,prepare_context,sequence
from migrate_combat_navigation_cuda import branch
from combat_weapon_head import feature_mask
from process_combat_architecture_pool import sha,save


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--update',type=pathlib.Path,required=True)
    ap.add_argument('--data',type=pathlib.Path,required=True)
    ap.add_argument('--execution',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    args=ap.parse_args(); assert not args.out.exists() and torch.cuda.is_available()
    torch.set_default_device('cuda'); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    seal=read(args.update/'complete.json')
    pins={}
    for name,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
        path=args.update/name; assert sha(path)==seal[key]; pins[path]=seal[key]
    model=read(args.update/'weights.json'); assert model['feature_version']=='combat_features_v10'
    meta=read(args.data/'report.json'); assert meta['model_sha256']==seal['weights_sha256']
    for name in ['rollout','sequence']:
        path=args.data/(name+'.jsonl'); assert sha(path)==meta[name+'_sha256'];pins[path]=meta[name+'_sha256']
    pins[args.execution]=sha(args.execution);execution=read(args.execution)
    releases={c['seed']:c['network_release_frame'] for c in execution['cases']}
    with (args.data/'rollout.jsonl').open(encoding='utf-8-sig') as f:rows=[json.loads(l) for l in f]
    with (args.data/'sequence.jsonl').open(encoding='utf-8-sig') as f:history=[json.loads(l) for l in f]
    prepared=prepare_context(history,rows,'cuda'); actor=branch(model,'actor')
    features=torch.tensor([r['features'] for r in rows],dtype=torch.float32,device='cuda')
    with torch.no_grad():
        raw,error=sequence(actor,prepared,check_states=True);assert error<2e-5
        mask=feature_mask(features[:,:845]);gate=(features[:,1132]==1)&mask[:,1:5].any(-1)
        initial=torch.tensor([r['frame']==releases[r['seed']] for r in rows],dtype=torch.bool,device='cuda')
        groups={'gated_rows':gate,'initial_gated_rows':gate&initial,'ungated_rows':~gate}
        result={}
        for temperature in [1.,2.,4.]:
            changed=raw.clone();changed[:,8:20]=torch.where(gate[:,None],raw[:,8:20]/temperature,raw[:,8:20])
            assert torch.equal(changed[~gate],raw[~gate])
            assert torch.equal(changed[:,:8],raw[:,:8]) and torch.equal(changed[:,20:],raw[:,20:])
            p=changed[:,8:20].double().masked_fill(~mask,-torch.inf).softmax(-1)
            assert torch.equal(p.argmax(-1),raw[:,8:20].masked_fill(~mask,-torch.inf).argmax(-1))
            result[str(temperature)]={}
            for name,selected in groups.items():
                q=p[selected];assert len(q)>0
                result[str(temperature)][name]=dict(rows=len(q),keep_probability_mean=float(q[:,0].mean()),
                    supported_probability_mean=float(q[:,1:5].sum(-1).mean()),
                    entropy_mean=float(-(q*q.clamp_min(1e-300).log()).sum(-1).mean()),
                    keep_argmax_fraction=float((q.argmax(-1)==0).double().mean()))
    for path,digest in pins.items():assert sha(path)==digest,str(path)
    save(args.out,dict(state='complete',device='cuda',version='conditional_weapon_exploration_probe_v1',
         weights_sha256=seal['weights_sha256'],data_report_sha256=sha(args.data/'report.json'),
         execution_sha256=pins[args.execution],script_sha256=sha(__file__),real_history_rows=len(rows),
         state_max_error=error,results=result,training_performed=False,promotion=False,
         scope='Observed equipped Grenades plus availability of Blaster/Shotgun/SSG/MG gates temperature on all masked weapon logits. Ungated outputs and all other heads unchanged; argmax unchanged. Metadata/CUDA diagnostic only; no fitting, new policy or evaluation of improved behavior. Fresh own-policy training required for an experiment.'))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

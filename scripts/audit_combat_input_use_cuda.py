"""CUDA feature-band ablation on captured histories; no fitting or quality claim."""
import argparse
import json
import pathlib

from ppo_recurrent import (torch, training_devices, read, sha, prepare_context,
                           sequence, Recurrent, durable_json)
from combat_attention import CausalAttention
from combat_spatial_aim import wrap
from combat_target_head import distribution
from combat_precision_head import probabilities
from audit_combat_head_learning_cuda import components


def build(model):
    if model.get('memory'):
        base=Recurrent(model['actor'],model['memory']['actor'])
    else:
        spec=model['attention']
        base=CausalAttention(model['actor'],spec['actor'],spec['heads'],spec['window'])
    return wrap(base,model).to('cuda').eval()


def main():
    parser=argparse.ArgumentParser()
    for name in ('behavior','model','data','out'):
        parser.add_argument('--'+name,type=pathlib.Path,required=True)
    args=parser.parse_args()
    assert not args.out.exists()
    training_devices();torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    behavior,model,meta=read(args.behavior),read(args.model),read(args.data/'report.json')
    assert sha(args.behavior)==meta['model_sha256']
    assert model['feature_version']==behavior['feature_version']==meta['feature_version']=='combat_features_v9'
    assert model.get('target_head')==behavior.get('target_head') and model.get('aim_mode_head')==behavior.get('aim_mode_head')
    for path,digest in meta['source_sha256'].items():
        assert sha(pathlib.Path(path))==digest,path
    for name in ('rollout','sequence'):
        assert sha(args.data/(name+'.jsonl'))==meta[name+'_sha256']
    rows=[json.loads(line) for line in (args.data/'rollout.jsonl').read_text().splitlines()]
    context=[json.loads(line) for line in (args.data/'sequence.jsonl').read_text().splitlines()]
    assert len(rows)==meta['rows'] and len(context)==meta['sequence_rows']
    prepared=prepare_context(context,rows,'cuda')
    features=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float32)
    # The native JSON contract omits zero-valued categorical selections.
    sample=lambda name,dtype:torch.tensor([r['sample'].get(name,0) if name in
                                          ('weapon','target','aim_mode') else r['sample'][name]
                                          for r in rows],device='cuda',dtype=dtype)
    with torch.no_grad():
        old,state_error=sequence(build(behavior),prepared,check_states=True)
        old_std=torch.tensor(behavior['log_std'],device='cuda')
        lp,_=probabilities(old,old_std,sample('latent',torch.float32),sample('attack',torch.float32),
                           sample('vertical',torch.long),sample('weapon',torch.long),features,
                           sample('target',torch.long),sample('aim_mode',torch.long))
        log_error=float((lp-sample('log_probability',torch.float32)).abs().max())
        assert state_error<2e-5 and log_error<.003,(state_error,log_error)
        actor=build(model);std=torch.tensor(model['log_std'],device='cuda')
        raw,_=sequence(actor,prepared)
        original=distribution(raw[:,:45],features).probs
        results={}
        for band,(start,end) in dict(previous_target=(845,854),navigation=(854,881),remembered_threats=(881,1121)).items():
            x=prepared[0].clone();x[...,start:end]=0
            changed,_=sequence(actor,(x,*prepared[1:]))
            choice=distribution(changed[:,:45],features).probs
            terms,total=components(raw,changed,std,std,features)
            results[band]=dict(columns=end-start,exact_policy_kl_mean=float(total.mean()),
                               target_total_variation_mean=float((original-choice).abs().sum(-1).mean()/2),
                               target_argmax_changed_rows=int((original.argmax(-1)!=choice.argmax(-1)).sum()),
                               heads_kl_mean={name:float(value.mean()) for name,value in terms.items()})
    report=dict(version='combat_input_use_cuda_v1',device='cuda',rows=len(rows),
                model_sha256=sha(args.model),behavior_sha256=sha(args.behavior),
                data_sha256=sha(args.data/'report.json'),script_sha256=sha(pathlib.Path(__file__)),
                behavior_log_probability_max_error=log_error,behavior_memory_max_error=state_error,
                bands=results,scope='Counterfactual zero-band replay over full captured histories, holding current physical availability masks fixed. No fitting, gameplay trajectory, causal reward attribution or quality claim; no CPU neural fallback.')
    durable_json(args.out,report)
    print(json.dumps(dict(rows=len(rows),bands=results)),flush=True)


if __name__=='__main__':
    main()

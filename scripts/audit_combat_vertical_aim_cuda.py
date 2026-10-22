"""Audit a matched BC pair and measure vertical sensitivity on held-out histories."""
import argparse
import gzip
import json
import pathlib
from ppo_recurrent import torch, prepare_context, sequence
from migrate_combat_navigation_cuda import branch
from combat_sampled_motion_aim import correction, VERTICAL_VERSION, VERTICAL_CONTROL_VERSION
from combat_precision_head import selected_means
from process_combat_architecture_pool import read, save, sha


def main():
    ap=argparse.ArgumentParser()
    for name in ('protocol','control','candidate','labels','out'):
        ap.add_argument('--'+name,type=pathlib.Path,required=True)
    args=ap.parse_args();assert torch.cuda.is_available() and not args.out.exists()
    torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    pins={}
    def pin(path,expected=None):
        path=pathlib.Path(path);digest=sha(path);assert expected is None or expected==digest
        pins[str(path.resolve())]=digest;return path
    protocol=read(pin(args.protocol));assert protocol['device']=='cuda' and not protocol['selection_on_development']
    for item in protocol['inputs']:pin(item['path'],item['sha256'])
    reports={};models={};actors={}
    for name,path,version in [('control',args.control,VERTICAL_CONTROL_VERSION),('candidate',args.candidate,VERTICAL_VERSION)]:
        seal=read(pin(path/'complete.json'))
        for file,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:pin(path/file,seal[key])
        report=read(path/'report.json');model=read(path/'weights.json')
        assert report['version']==model['sampled_motion_aim']['version']==version
        assert report['device']=='cuda' and not report['selection_on_development']
        for key in ('steps','learning_rate','retention_weight'):assert report[key]==protocol[key]
        cp=torch.load(path/'checkpoint.pt',map_location='cuda',weights_only=True)
        actor=branch(model,'actor')
        assert cp['weights_sha256']==seal['weights_sha256']
        from fork_conditional_weapon_exploration_cuda import equal_tree
        assert equal_tree(cp['actor'],actor.state_dict())
        initial=torch.load(pathlib.Path(protocol['source_updates'][len(reports)])/'checkpoint.pt',map_location='cuda',weights_only=True)
        assert equal_tree(cp['value'],initial['value']) and torch.equal(cp['log_std'],initial['log_std'])
        for key,value in cp['actor'].items():
            if not key.startswith('sampled_motion_aim_branch.'):assert torch.equal(value,initial['actor'][key])
        reports[name],models[name],actors[name]=report,model,actor
    for key in ('history_source_weights_sha256','retention_train_keys'):
        assert reports['control'][key]==reports['candidate'][key]
    assert reports['control']['trajectory'][0]==reports['candidate']['trajectory'][0]
    assert reports['control']['vertical_columns_norm']==0 and reports['candidate']['vertical_columns_norm']>0
    meta=read(pin(args.labels/'report.json'));info=meta['splits']['development']
    with gzip.open(pin(info['output'],info['sha256']),'rt',encoding='utf-8') as stream:labels=list(map(json.loads,stream))
    root=pathlib.Path(meta['corpus'])/'processing-development/merged';report=read(pin(root/'report.json'))
    streams={}
    for name in ('rollout','sequence'):
        path=pin(root/(name+'.jsonl'),report[name+'_sha256'])
        streams[name]=[json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()]
    rows=streams['rollout'];lookup={(r['seed'],r['index']):i for i,r in enumerate(rows)}
    indices=torch.tensor([lookup[(l['seed'],l['index'])] for l in labels],device='cuda',dtype=torch.long)
    x=torch.tensor([l['features'] for l in labels],device='cuda',dtype=torch.float32)
    movement=torch.tensor([l['sample']['latent'][:2] for l in labels],device='cuda',dtype=torch.float32)
    target=torch.tensor([l['sample']['target'] for l in labels],device='cuda',dtype=torch.long)
    mode=torch.tensor([l['sample'].get('aim_mode',0) for l in labels],device='cuda',dtype=torch.long)
    results={}
    with torch.no_grad():
        prepared=prepare_context(streams['sequence'],rows,'cuda')
        for name,actor in actors.items():
            raw,error=sequence(actor,prepared,check_states=True);assert error<2e-5;raw=raw[indices]
            baseline=selected_means(raw,target,mode)[:,2:4]
            scale=torch.where(mode==1,15.,180.)[:,None]
            physical=[]
            for v in range(3):
                delta=correction(actor.sampled_motion_aim_branch,x,raw,movement,target,mode,torch.full_like(target,v))
                physical.append((baseline+delta).tanh()*scale)
            span=torch.stack(physical,1).amax(1)-torch.stack(physical,1).amin(1)
            results[name]=dict(samples=len(labels),mean_vertical_span_axis_degrees=float(span.mean()),max_vertical_span_axis_degrees=float(span.max()))
        assert results['control']['max_vertical_span_axis_degrees']==0
        assert results['candidate']['max_vertical_span_axis_degrees']>1e-5
    for p,h in pins.items():assert sha(p)==h
    save(args.out,dict(state='complete',version='combat_vertical_aim_pair_audit_v1',device='cuda',
        matched_initial_metrics=True,matched_retention_histories=True,frozen_original_actor_value_std=True,
        results=results,training_performed=False,promotion=False,
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),
        scope='Counterfactual selected vertical on identical development histories. Sensitivity is not improved combat, accuracy or evidence of optimal compensation.'))
    print(json.dumps(results))


if __name__=='__main__':main()

"""Cache actual frozen actor outputs from full original temporal histories on CUDA."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from evaluate_registered_combat_models import read,save,sha
from ppo_combat import torch,training_devices
from ppo_recurrent import prepare_context,sequence
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import JointAim,base_model,events,probabilities
from combat_conditional_fire import logits as fire_logits
from combat_precision_head import selected_means


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();a.out.mkdir(parents=True);(a.out/'chunks').mkdir();pins={};live={};snapshots={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path);h=pins[key] if key in pins else sha(path)
        assert expected is None or h==expected,str(path);pins[key]=h;return path
    def freeze_sources():
        scripts=Path(__file__).resolve().parent
        for module in list(sys.modules.values()):
            name=getattr(module,'__file__',None)
            if not name:continue
            path=Path(name).resolve()
            if path.parent!=scripts or path.suffix!='.py' or str(path) in live:continue
            h=sha(path);target=a.out/'source'/path.name;target.parent.mkdir(exist_ok=True)
            shutil.copy2(path,target);pin(target,h);live[str(path)]=h
            snapshots[str(path)]=dict(snapshot=str(target.resolve()),sha256=h)
    p=read(pin(a.protocol));assert p['version']=='combat_memory_recovery_training_v1'
    dataset=Path(p['dataset']);closed=read(pin(dataset/'complete.json'))
    ds=read(pin(dataset/'report.json',closed['report_sha256']));assert ds['version']=='combat_target_lifecycle_dataset_v1'
    source=Path(p['source']);model=read(pin(source/'weights.json',p['source_weights_sha256']))
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    actor=branch(base_model(model,allow_shared=True),'actor').requires_grad_(False)
    joint=JointAim(model,allow_shared=True).requires_grad_(False)
    before=copy.deepcopy(actor.state_dict());joint_before=copy.deepcopy(joint.state_dict())
    freeze_sources();chunks=[];max_memory=max_lp=0.
    for ci,item in enumerate(ds['chunks']):
        with np.load(pin(item['path'],item['sha256'])) as data:original={k:np.array(data[k],copy=True) for k in data.files}
        folder=Path(item['case']['folder']);meta=read(pin(folder/'cuda/report.json'))
        assert meta['model_sha256']==p['source_weights_sha256'] and meta['observation_timing_contract']=='frozen_world_v1'
        rows=list(map(json.loads,pin(folder/'cuda/rollout.jsonl',item['case']['source_rollout_sha256']).open(encoding='utf-8-sig')))
        histories=list(map(json.loads,pin(folder/'cuda/sequence.jsonl',item['case']['source_sequence_sha256']).open(encoding='utf-8-sig')))
        lookup={(r['index'],r['frame']):i for i,r in enumerate(rows)};assert len(lookup)==len(rows)
        positions=[lookup[int(step),int(frame)] for step,frame in zip(original['step'],original['frame'])]
        selected=[rows[i] for i in positions]
        x=torch.tensor(original['features'],device='cuda');cat=torch.tensor(original['categories'],device='cuda',dtype=torch.long)
        target,mode,vertical,weapon=(cat[:,i] for i in range(4))
        actual=torch.tensor(original['action'],device='cuda');attack=torch.tensor(original['attack'],device='cuda')
        assert torch.equal(x,torch.tensor([r['features'] for r in selected],device='cuda'))
        assert all(int(r['seed'])==item['case']['seed'] for r in selected)
        with torch.no_grad():
            prepared=prepare_context(histories,rows,'cuda');all_raw,error=sequence(actor,prepared,32,True,'actor');assert error<2e-5,error
            raw=all_raw[torch.tensor(positions,device='cuda',dtype=torch.long)]
            source_action,choice,timing,eligible=events(selected,x,target,mode,vertical,weapon)
            assert torch.equal(source_action,torch.tensor(original['source_action'],device='cuda'))
            std=torch.tensor(model['log_std'],device='cuda')
            lp=probabilities(joint,raw,std,source_action,actual,attack,vertical,weapon,x,target,mode,choice,timing,
                actor.sampled_motion_aim_branch,actor.motor_branch,actor.conditional_fire_branch)[0]
            lp_error=float((lp-torch.tensor([r['sample']['log_probability'] for r in selected],device='cuda')).abs().max())
            assert lp_error<.003,lp_error
            fire=fire_logits(actor.conditional_fire_branch,x,raw,target,mode,actual,vertical,weapon)
            means=selected_means(raw,target,mode,x,source_action[:,:2],actor.sampled_motion_aim_branch,vertical,actor.motor_branch)
            recovery=torch.tensor(original['recovery_known'],device='cuda')
            assert bool(((target[recovery]==0)&(mode[recovery]==0)).all())
            assert torch.equal(means[recovery,2:4],raw[recovery,2:4])
            original.update(raw_actor=raw.cpu().numpy(),source_fire_logit=fire.cpu().numpy())
        path=a.out/'chunks'/f'{ci:04d}-{item["split"]}.npz';np.savez_compressed(path,**original)
        chunks.append(dict(path=str(path.resolve()),sha256=sha(path),rows=len(selected),split=item['split'],case=item['case'],
            memory_max_error=error,source_lp_max_error=lp_error))
        max_memory=max(max_memory,error);max_lp=max(max_lp,lp_error)
        save(a.out/'progress.json',dict(stage='cuda-source-replay',cases_done=len(chunks),cases=len(ds['chunks'])))
    assert all(torch.equal(v,actor.state_dict()[k]) for k,v in before.items())
    assert all(torch.equal(v,joint.state_dict()[k]) for k,v in joint_before.items())
    freeze_sources()
    for path,h in live.items():assert sha(path)==h,path
    for path,h in pins.items():assert sha(path)==h,path
    report=dict(state='complete',version='combat_memory_recovery_source_cache_v1',device='cuda',chunks=chunks,inputs=pins,
        source_code_snapshots=snapshots,source_weights_sha256=p['source_weights_sha256'],protocol_sha256=sha(a.protocol),
        source_parameters_exact=True,joint_parameters_exact=True,memory_max_error=max_memory,source_lp_max_error=max_lp,
        rows=sum(c['rows'] for c in chunks),compressed_bytes=sum(Path(c['path']).stat().st_size for c in chunks),
        training_performed=False,source_model_execution=True,ordinary_training_eligible=False,runtime_action_admitted=False,promotion=False)
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    save(a.out/'progress.json',dict(stage='complete',cases=len(chunks)));print(dict(rows=report['rows'],memory_error=max_memory,lp_error=max_lp,device='cuda'))


if __name__=='__main__':main()

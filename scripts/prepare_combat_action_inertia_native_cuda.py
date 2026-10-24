"""Export evaluation-only inertia variants and check actual temporal actors on CUDA."""
import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
import numpy as np
from evaluate_registered_combat_models import read,save,sha
from ppo_combat import torch,training_devices
from ppo_recurrent import prepare_context,sequence
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import JointAim,base_model,events,probabilities
from combat_action_inertia import VERSION,apply,movement_log_std

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('protocol','cache','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();a.out.mkdir(parents=True);pins={};live={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=pins.get(str(path)) or sha(path);assert expected is None or h==expected,str(path);pins[str(path)]=h;return path
    def snapshot():
        scripts=Path(__file__).resolve().parent
        for module in list(sys.modules.values()):
            name=getattr(module,'__file__',None)
            if not name:continue
            path=Path(name).resolve()
            if path.parent!=scripts or path.suffix!='.py' or str(path) in live:continue
            h=sha(path);target=a.out/'source'/path.name;target.parent.mkdir(exist_ok=True);shutil.copy2(path,target);pin(target,h);live[str(path)]=h
    p=read(pin(a.protocol));assert p['version']=='combat_action_inertia_experiment_v1'
    seal=read(pin(a.cache/'complete.json'));cache=read(pin(a.cache/'report.json',seal['report_sha256']))
    assert cache['version']=='combat_action_inertia_source_cache_v1' and cache['source_parameters_exact'] and len(cache['chunks'])==24
    source=Path(p['source']);model=read(pin(source/'weights.json',p['source_weights_sha256']))
    assert not model.get('movement_inertia') and not model.get('memory_recovery')
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    original=branch(base_model(model,allow_shared=True),'actor').requires_grad_(False);before=copy.deepcopy(original.state_dict())
    header=dict(version=VERSION,rho=p['rho']);projected=movement_log_std(torch.tensor(model['log_std'],device='cuda',dtype=torch.float64),header).cpu().tolist()
    actors={};models={};definitions={}
    for label in ('zero','noise-only','inertia'):
        changed=copy.deepcopy(model)
        if label in ('noise-only','inertia'):changed['log_std'][:2]=projected[:2]
        if label!='noise-only':changed['movement_inertia']=dict(version=VERSION,rho=0. if label=='zero' else p['rho'])
        core=copy.deepcopy(changed);core.pop('movement_inertia',None);core['log_std']=model['log_std'];assert core==model
        folder=a.out/label;folder.mkdir();save(folder/'weights.json',changed)
        actors[label]=branch(base_model(changed,allow_shared=True),'actor').requires_grad_(False)
        assert set(actors[label].state_dict())==set(before)
        assert all(torch.equal(v,actors[label].state_dict()[k]) for k,v in before.items())
        definitions[label]=changed;models[label]=dict(path=str((folder/'weights.json').resolve()),sha256=sha(folder/'weights.json'))
    joint=JointAim(model,allow_shared=True).requires_grad_(False);snapshot();rows=0;maximum=0.;lp_max=0.
    for ci,item in enumerate(cache['chunks']):
        assert item['split']=='train'
        with np.load(pin(item['path'],item['sha256'])) as d:arrays={k:np.array(d[k],copy=True) for k in ('features','raw_actor','frame','step')}
        folder=Path(item['case']['folder']);recorded=list(map(json.loads,pin(folder/'cuda/rollout.jsonl',item['case']['source_rollout_sha256']).open(encoding='utf-8-sig')))
        history=list(map(json.loads,pin(folder/'cuda/sequence.jsonl',item['case']['source_sequence_sha256']).open(encoding='utf-8-sig')))
        lookup={(r['index'],r['frame']):i for i,r in enumerate(recorded)};positions=[lookup[int(s),int(f)] for s,f in zip(arrays['step'],arrays['frame'])]
        selected=[recorded[i] for i in positions];idx=torch.tensor(positions,device='cuda',dtype=torch.long)
        with torch.no_grad():
            context=prepare_context(history,recorded,'cuda');raw,error=sequence(original,context,32,True,'actor');assert error<2e-5
            x=torch.tensor(arrays['features'],device='cuda');cached=torch.tensor(arrays['raw_actor'],device='cuda');assert float((raw[idx]-cached).abs().max())<2e-5
            outputs={label:sequence(actor,context,32,True,'actor')[0] for label,actor in actors.items()}
            assert torch.equal(outputs['zero'],raw) and torch.equal(outputs['noise-only'],raw)
            expected=apply(x,raw[idx],header);difference=float((outputs['inertia'][idx]-expected).abs().max());assert difference<2e-5
            assert torch.equal(outputs['inertia'][idx][:,2:],raw[idx][:,2:]);maximum=max(maximum,difference)
            # Zero header keeps full conditional likelihood, not just movement.
            categorical=lambda k:torch.tensor([r['sample'].get(k,0) for r in selected],device='cuda',dtype=torch.long)
            target,mode,vertical,weapon=(categorical(k) for k in ('target','aim_mode','vertical','weapon'))
            final=torch.tensor([r['sample']['latent'] for r in selected],device='cuda');attack=torch.tensor([float(r['sample']['attack']) for r in selected],device='cuda')
            source_action,choice,timing,_=events(selected,x,target,mode,vertical,weapon)
            actor=actors['zero'];lp=probabilities(joint,outputs['zero'][idx],torch.tensor(model['log_std'],device='cuda'),source_action,final,attack,vertical,weapon,x,target,mode,choice,timing,actor.sampled_motion_aim_branch,actor.motor_branch,actor.conditional_fire_branch)[0]
            error=float((lp-torch.tensor([r['sample']['log_probability'] for r in selected],device='cuda')).abs().max());assert error<.003;lp_max=max(lp_max,error)
        rows+=len(x);save(a.out/'progress.json',dict(stage='cuda-native-inertia-preparation',cases_done=ci+1,cases=24))
    assert all(torch.equal(v,original.state_dict()[k]) for k,v in before.items())
    snapshot()
    for path,h in live.items():assert sha(path)==h,path
    for path,h in pins.items():assert sha(path)==h,path
    report=dict(state='complete',version='combat_action_inertia_native_models_v1',device='cuda',models=models,source=str(source.resolve()),source_weights_sha256=p['source_weights_sha256'],source_parameters_exact=True,zero_source_raw_exact=True,noise_only_source_raw_exact=True,inertia_other_outputs_exact=True,projection_max_error=maximum,zero_source_lp_max_error=lp_max,rows=rows,inputs=pins,
        training_performed=False,ppo_checkpoint_written=False,native_compatibility_proven=False,promotion=False,scope='Actual full temporal actor source/zero/noise/inertia CUDA projection and zero conditional likelihood. New nonzero runtime samples/likelihood need native replay. No PPO resume or quality proof.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')));save(a.out/'progress.json',dict(stage='complete',rows=rows));print(dict(rows=rows,max_error=maximum,zero_lp_error=lp_max))

if __name__=='__main__':main()

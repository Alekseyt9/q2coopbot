"""Export source-preserving zero/admitted recovery heads and verify actual actor integration."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import numpy as np
from evaluate_registered_combat_models import read,save,sha
from ppo_combat import torch,training_devices
from ppo_recurrent import prepare_context,sequence
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model
from combat_memory_recovery_v2 import build_zero,project,active_mask
from combat_memory_recovery_runtime import export as export_head


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','fit','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--fire-bias-fit',type=Path,help='Separately sealed scalar fire correction; never the rejected shared endpoint')
    a=ap.parse_args();assert not a.out.exists();a.out.mkdir(parents=True);pins={};live={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path);h=pins[key] if key in pins else sha(path)
        assert expected is None or h==expected,str(path);pins[key]=h;return path
    for name in (Path(__file__).name,'combat_memory_recovery.py','combat_memory_recovery_v2.py','combat_memory_recovery_runtime.py','combat_spatial_aim.py'):
        path=Path(__file__).with_name(name).resolve();h=sha(path);target=a.out/'source'/name;target.parent.mkdir(exist_ok=True)
        shutil.copy2(path,target);pin(target,h);live[str(path)]=h
    p=read(pin(a.protocol));assert p['version']=='combat_memory_recovery_training_v1'
    fs=read(pin(a.fit/'complete.json'));fit=read(pin(a.fit/'report.json',fs['report_sha256']))
    assert fit['device']=='cuda' and fit['results']['gaze_only']['offline_probe_admitted']
    assert not fit['results']['gaze_and_mg_fire']['offline_probe_admitted'],'Revisit explicit candidate admission before export'
    source=Path(p['source']).resolve();model=read(pin(source/'weights.json',p['source_weights_sha256']));assert not model.get('memory_recovery')
    cache_root=Path(p['cache']);cs=read(pin(cache_root/'complete.json'));cache=read(pin(cache_root/'report.json',cs['report_sha256']))
    assert cache['source_weights_sha256']==p['source_weights_sha256']
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    zero=build_zero();gaze=build_zero();state=torch.load(pin(a.fit/'gaze_only/checkpoint.pt',fit['results']['gaze_only']['checkpoint_sha256']),map_location='cuda',weights_only=True)
    gaze.load_state_dict(state['state_dict']);assert bool((gaze[-1].weight[2]==0).all()) and float(gaze[-1].bias[2].detach())==0
    heads={'zero':zero,'gaze_only':gaze};models={};actors={}
    if a.fire_bias_fit:
        bs=read(pin(a.fire_bias_fit/'complete.json'));br=read(pin(a.fire_bias_fit/'report.json',bs['report_sha256']))
        assert br['version']=='combat_memory_recovery_fire_bias_fit_v1' and br['offline_probe_admitted'] and br['gaze_and_other_parameters_exact']
        bstate=torch.load(pin(a.fire_bias_fit/'checkpoint.pt',br['checkpoint_sha256']),map_location='cuda',weights_only=True)
        assert bstate['source_gaze_checkpoint_sha256']==fit['results']['gaze_only']['checkpoint_sha256']
        bhead=build_zero();bhead.load_state_dict(bstate['state_dict'])
        for k,v in gaze.state_dict().items():
            if k=='4.bias':assert torch.equal(v[:2],bhead.state_dict()[k][:2])
            else:assert torch.equal(v,bhead.state_dict()[k])
        heads['gaze_fire_bias']=bhead
    original=branch(base_model(model,allow_shared=True),'actor').requires_grad_(False);before=copy.deepcopy(original.state_dict())
    for name,head in heads.items():
        changed=copy.deepcopy(model);export_head(changed,head)
        core=copy.deepcopy(changed);core.pop('memory_recovery');assert core==model
        folder=a.out/name;folder.mkdir();save(folder/'weights.json',changed)
        models[name]=dict(path=str((folder/'weights.json').resolve()),sha256=sha(folder/'weights.json'))
        actors[name]=branch(base_model(changed,allow_shared=True),'actor').requires_grad_(False)
        assert all(torch.equal(v,actors[name].state_dict()[k]) for k,v in before.items())
    rows=active_rows=changed_rows=0;max_error=0.
    for ci,item in enumerate(cache['chunks']):
        with np.load(pin(item['path'],item['sha256'])) as d:arrays={k:np.array(d[k],copy=True) for k in ('features','raw_actor','step','frame')}
        folder=Path(item['case']['folder'])
        recorded=list(map(json.loads,pin(folder/'cuda/rollout.jsonl',item['case']['source_rollout_sha256']).open(encoding='utf-8-sig')))
        history=list(map(json.loads,pin(folder/'cuda/sequence.jsonl',item['case']['source_sequence_sha256']).open(encoding='utf-8-sig')))
        lookup={(r['index'],r['frame']):i for i,r in enumerate(recorded)}
        idx=torch.tensor([lookup[int(s),int(f)] for s,f in zip(arrays['step'],arrays['frame'])],device='cuda',dtype=torch.long)
        with torch.no_grad():
            prepared=prepare_context(history,recorded,'cuda');raw,_=sequence(original,prepared,32,True,'actor')
            zraw,_=sequence(actors['zero'],prepared,32,True,'actor');assert torch.equal(zraw,raw),'Zero projection changed actual actor'
            craw,_=sequence(actors['gaze_only'],prepared,32,True,'actor')
            x=torch.tensor(arrays['features'],device='cuda');baseline=torch.tensor(arrays['raw_actor'],device='cuda')
            assert float((raw[idx]-baseline).abs().max())<2e-5
            expected=project(gaze,x,baseline);error=float((craw[idx]-expected).abs().max());assert error<2e-5,error
            active=active_mask(x);assert torch.equal(craw[idx][~active],raw[idx][~active])
            ids=torch.tensor([0,1,*range(4,81)],device='cuda',dtype=torch.long)
            assert torch.equal(craw[idx][:,ids],raw[idx][:,ids])
            if 'gaze_fire_bias' in actors:
                braw,_=sequence(actors['gaze_fire_bias'],prepared,32,True,'actor')
                bexpected=project(heads['gaze_fire_bias'],x,baseline)
                assert float((braw[idx]-bexpected).abs().max())<2e-5
                assert torch.equal(braw[idx][:,:4],craw[idx][:,:4]) and torch.equal(braw[idx][:,5:],craw[idx][:,5:])
                scope=active & (x[:,1125]==1)
                assert torch.equal(braw[idx][~scope],craw[idx][~scope])
            changed_rows+=int(((craw[idx][:,2:4]-raw[idx][:,2:4]).abs().amax(-1)>1e-6).sum())
            active_rows+=int(active.sum());rows+=len(x);max_error=max(max_error,error)
        save(a.out/'progress.json',dict(stage='cuda-native-model-preparation',cases_done=ci+1,cases=len(cache['chunks'])))
    assert all(torch.equal(v,original.state_dict()[k]) for k,v in before.items()) and changed_rows>0
    for path,h in live.items():assert sha(path)==h,path
    for path,h in pins.items():assert sha(path)==h,path
    report=dict(state='complete',version='combat_memory_recovery_native_models_v1',device='cuda',models=models,source=str(source),
        source_weights_sha256=p['source_weights_sha256'],inputs=pins,source_parameters_exact=True,zero_source_raw_exact=True,
        actual_actor_rows=rows,active_rows=active_rows,changed_gaze_rows=changed_rows,gaze_projection_max_error=max_error,
        gaze_fire_and_unrelated_outputs_exact=True,training_performed=False,policy_json_written=True,ppo_checkpoint_written=False,
        native_compatibility_proven=False,runtime_action_admitted=False,promotion=False,
        scope='Actual full temporal source/zero/gaze CUDA replay. Joint fit rejected unchanged; only admitted gaze and zero exported for technical native tests.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    save(a.out/'progress.json',dict(stage='complete',rows=rows));print(dict(rows=rows,active=active_rows,changed=changed_rows,max_error=max_error))


if __name__=='__main__':main()

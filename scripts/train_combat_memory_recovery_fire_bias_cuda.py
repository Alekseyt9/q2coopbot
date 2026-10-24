"""Fit one blind-MG fire offset while retaining the learned gaze exactly."""
import argparse
import copy
from pathlib import Path
import shutil
import numpy as np
from evaluate_registered_combat_models import read,save,sha
from ppo_combat import torch,nn,training_devices
from combat_memory_recovery_v2 import build_zero,project,active_mask


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();a.out.mkdir(parents=True);pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path);h=pins[key] if key in pins else sha(path)
        assert expected is None or h==expected,str(path);pins[key]=h;return path
    for name in (Path(__file__).name,'combat_memory_recovery.py','combat_memory_recovery_v2.py'):
        path=Path(__file__).with_name(name);target=a.out/'source'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(path,target);pin(target,sha(path))
    p=read(pin(a.protocol));assert p['version']=='combat_memory_recovery_fire_bias_fit_v1'
    fit=Path(p['source_gaze_fit']);fs=read(pin(fit/'complete.json'));fr=read(pin(fit/'report.json',fs['report_sha256']))
    assert fr['results']['gaze_only']['offline_probe_admitted'] and fr['source_weights_sha256']==p['source_weights_sha256']
    cache=Path(p['cache']);cs=read(pin(cache/'complete.json'));cr=read(pin(cache/'report.json',cs['report_sha256']))
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.manual_seed(p['training']['seed']);torch.cuda.manual_seed_all(p['training']['seed'])
    saved=torch.load(pin(fit/'gaze_only/checkpoint.pt',fr['results']['gaze_only']['checkpoint_sha256']),map_location='cuda',weights_only=True)
    head=build_zero();head.load_state_dict(saved['state_dict']);head.requires_grad_(False);old=copy.deepcopy(head.state_dict())
    assert bool((head[-1].weight[2]==0).all()) and float(head[-1].bias[2])==0
    data={}
    for split in ('train','development'):
        parts=[]
        for chunk in cr['chunks']:
            if chunk['split']==split:
                with np.load(pin(chunk['path'],chunk['sha256'])) as d:parts.append({k:np.array(d[k],copy=True) for k in ('features','raw_actor','world','effects','attack','recovery_known','source_fire_logit')})
        d={k:torch.tensor(np.concatenate([v[k] for v in parts]),device='cuda') for k in parts[0]}
        y=d['effects'][:,0];mask=d['recovery_known'].bool() & active_mask(d['features']) & (d['features'][:,1125]==1) & (d['attack']==1) & (y[:,6]>0) & ((y[:,7]+y[:,8]+y[:,9])>0)
        worlds,inv,count=torch.unique(d['world'][mask],return_inverse=True,return_counts=True)
        weights=1./(len(worlds)*count[inv]);d.update(mask=mask,weights=weights,fire_target=(y[mask,7]>0).float(),eligible_worlds=len(worlds))
        data[split]=d
    bias=nn.Parameter(torch.zeros((),device='cuda'));conf=p['training'];assert conf['steps']==200
    optimizer=torch.optim.Adam([bias],lr=conf['learning_rate'])
    train=data['train']
    for step in range(conf['steps']):
        logits=train['source_fire_logit'][train['mask']]+bias
        loss=(nn.functional.binary_cross_entropy_with_logits(logits,train['fire_target'],reduction='none')*train['weights']).sum()+conf['l2_bias_weight']*bias.square()
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    with torch.no_grad():head[-1].bias[2].copy_(bias)
    scores={}
    for split,d in data.items():
        with torch.no_grad():
            before=build_zero();before.load_state_dict(old);before.requires_grad_(False)
            baseline=project(before,d['features'],d['raw_actor']);changed=project(head,d['features'],d['raw_actor'])
            assert torch.equal(changed[:,:4],baseline[:,:4]) and torch.equal(changed[:,5:],baseline[:,5:])
            fire_scope=active_mask(d['features']) & (d['features'][:,1125]==1)
            assert torch.equal(changed[~fire_scope],baseline[~fire_scope])
            logits=d['source_fire_logit'][d['mask']]
            source_p=float((logits.sigmoid()*d['weights']).sum());new_p=float(((logits+bias).sigmoid()*d['weights']).sum())
            scores[split]=dict(rows=int(d['mask'].sum()),worlds=d['eligible_worlds'],positive_rows=int(d['fire_target'].sum()),source_probability=source_p,
                corrected_probability=new_p,ratio=new_p/source_p,bce=float((nn.functional.binary_cross_entropy_with_logits(logits+bias,d['fire_target'],reduction='none')*d['weights']).sum()))
    for k,v in old.items():
        if k=='4.bias':assert torch.equal(v[:2],head.state_dict()[k][:2])
        else:assert torch.equal(v,head.state_dict()[k])
    torch.save(dict(version='combat_memory_recovery_fire_bias_fit_v1',state_dict=head.state_dict(),fire_bias=bias.detach(),optimizer=optimizer.state_dict(),
        steps=conf['steps'],updated_parameters=1,source_gaze_checkpoint_sha256=fr['results']['gaze_only']['checkpoint_sha256'],runtime_admitted=False),a.out/'checkpoint.pt')
    reloaded=torch.load(a.out/'checkpoint.pt',map_location='cuda',weights_only=True);assert all(torch.equal(v,reloaded['state_dict'][k]) for k,v in head.state_dict().items())
    for path,h in pins.items():assert sha(path)==h,path
    report=dict(state='complete',version=p['version'],device='cuda',updated_parameters=1,steps=conf['steps'],learned_fire_bias=float(bias.detach()),metrics=scores,
        offline_probe_admitted=scores['development']['ratio']<=.75,checkpoint_sha256=sha(a.out/'checkpoint.pt'),inputs=pins,
        gaze_and_other_parameters_exact=True,gpu_reload_exact=True,source_weights_sha256=p['source_weights_sha256'],training_performed=True,
        source_model_training=False,runtime_action_admitted=False,policy_weights_written=False,promotion=False,
        scope='One trained intercept in negative-only fresh-memory MG contexts. No firing calibration or fight quality claim; all old gaze tensors/outputs exact.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(bias=report['learned_fire_bias'],metrics=scores,offline_probe_admitted=report['offline_probe_admitted']))


if __name__=='__main__':main()

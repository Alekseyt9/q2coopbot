"""Fit only the new memory recovery head on cached real source outputs, CUDA-only."""
import argparse
import copy
from pathlib import Path
import shutil
import numpy as np
from evaluate_registered_combat_models import read,save,sha
from ppo_combat import torch,nn,training_devices
from combat_memory_recovery_v2 import VERSION,PARAMETERS,active_mask,build_zero,project


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();a.out.mkdir(parents=True);pins={};live={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path);h=pins[key] if key in pins else sha(path)
        assert expected is None or h==expected,str(path);pins[key]=h;return path
    for name in (Path(__file__).name,'combat_memory_recovery.py','combat_memory_recovery_v2.py'):
        path=Path(__file__).with_name(name).resolve();h=sha(path);target=a.out/'source'/name;target.parent.mkdir(exist_ok=True)
        shutil.copy2(path,target);pin(target,h);live[str(path)]=h
    p=read(pin(a.protocol));assert p['version']=='combat_memory_recovery_training_v1'
    source=Path(p['source']);model=read(pin(source/'weights.json',p['source_weights_sha256']))
    root=Path(p['cache']);closed=read(pin(root/'complete.json'));cache=read(pin(root/'report.json',closed['report_sha256']))
    assert cache['device']=='cuda' and cache['source_model_execution'] and cache['source_parameters_exact']
    assert cache['source_weights_sha256']==p['source_weights_sha256'] and cache['protocol_sha256']==sha(a.protocol)
    for path,h in cache['inputs'].items():pin(path,h)
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data={}
    for split in ('train','development'):
        parts=[]
        for chunk in cache['chunks']:
            if chunk['split']!=split:continue
            with np.load(pin(chunk['path'],chunk['sha256'])) as z:parts.append({k:np.array(z[k],copy=True) for k in z.files})
        data[split]={k:torch.tensor(np.concatenate([z[k] for z in parts]),device='cuda') for k in parts[0]}
    def masks(d):
        gaze=d['recovery_known'].bool() & active_mask(d['features'])
        assert bool(((d['categories'][gaze,0]==0)&(d['categories'][gaze,1]==0)).all())
        y=d['effects'][:,0]
        fire=gaze & (d['features'][:,1125]==1) & (d['attack']==1) & (y[:,6]>0) & ((y[:,7]+y[:,8]+y[:,9])>0)
        return gaze,fire
    def balanced(world,mask):
        unique,inverse,counts=torch.unique(world[mask],return_inverse=True,return_counts=True)
        assert len(unique)>0
        w=torch.zeros(len(world),device='cuda');w[mask]=1./(len(unique)*counts[inverse]);return w
    weights={split:tuple(balanced(d['world'],mask) for mask in masks(d)) for split,d in data.items()}
    train,dev=data['train'],data['development'];gm,fm=masks(train)
    assert not set(train['world'].unique().tolist()).intersection(dev['world'].unique().tolist())
    std=torch.tensor(model['log_std'],device='cuda').exp()[2:4];assert bool((std>0).all())
    def teacher(d):return torch.atanh((d['recovery_direction'][:,[1,0]]/180.).clamp(-.999,.999))
    truths={name:teacher(d) for name,d in data.items()}
    conf=p['training'];assert conf['device']=='cuda' and conf['steps']==400
    torch.manual_seed(conf['seed']);torch.cuda.manual_seed_all(conf['seed'])
    gaze_draws=torch.multinomial(weights['train'][0],conf['steps']*conf['gaze_batch'],replacement=True).reshape(conf['steps'],-1)
    fire_draws=torch.multinomial(weights['train'][1],conf['steps']*conf['fire_batch'],replacement=True).reshape(conf['steps'],-1)
    prototype=Path(p['prototype']);ps=read(pin(prototype/'complete.json'));pr=read(pin(prototype/'report.json',ps['report_sha256']))
    assert pr['version']==VERSION and pr['state']=='passed'
    initial=torch.load(pin(prototype/'checkpoint.pt',pr['checkpoint_sha256']),map_location='cuda',weights_only=True)['state_dict']
    def metrics(head,split):
        d=data[split];gaze_w,fire_w=weights[split]
        out=project(head,d['features'],d['raw_actor']) if head is not None else d['raw_actor']
        degrees=180*out[:,2:4].tanh();truth=d['recovery_direction'][:,[1,0]]
        angular=(degrees-truth+180).remainder(360)-180
        fire=d['source_fire_logit']+out[:,4]-d['raw_actor'][:,4]
        y=(d['effects'][:,0,7]>0).float()
        return dict(gaze_rmse_degrees=float((angular.square().mean(-1)*gaze_w).sum().sqrt()),
            gaze_latent_nll_without_constant=float((.5*((out[:,2:4]-truths[split])/std).square().mean(-1)*gaze_w).sum()),
            negative_fire_probability=float((fire.sigmoid()*fire_w).sum()),
            fire_bce=float((nn.functional.binary_cross_entropy_with_logits(fire,y,reduction='none')*fire_w).sum()))
    baselines={s:metrics(None,s) for s in data}
    for s,d in data.items():baselines[s]['zero_command_gaze_rmse_degrees']=float((((d['recovery_direction'].square()).mean(-1))*weights[s][0]).sum().sqrt())
    results={}
    for name in p['models']:
        head=build_zero(conf['initialization_seed']);head.load_state_dict(copy.deepcopy(initial))
        assert all(torch.equal(v,head.state_dict()[k]) for k,v in initial.items())
        optimizer=torch.optim.AdamW(head.parameters(),lr=conf['learning_rate'],weight_decay=conf['weight_decay'])
        for step in range(conf['steps']):
            gi,fi=gaze_draws[step],fire_draws[step]
            out=project(head,train['features'][gi],train['raw_actor'][gi])
            gaze_loss=.5*((out[:,2:4]-truths['train'][gi])/std).square().mean()
            out_fire=project(head,train['features'][fi],train['raw_actor'][fi])
            fire_logits=train['source_fire_logit'][fi]+out_fire[:,4]-train['raw_actor'][fi,4]
            fire_loss=nn.functional.binary_cross_entropy_with_logits(fire_logits,(train['effects'][fi,0,7]>0).float())
            loss=gaze_loss+(conf['fire_loss_weight']*fire_loss if name=='gaze_and_mg_fire' else 0.)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),conf['gradient_clip']);optimizer.step()
            if name=='gaze_only':
                with torch.no_grad():head[-1].weight[2].zero_();head[-1].bias[2].zero_()
            if (step+1)%100==0:save(a.out/'progress.json',dict(stage='cuda-fitting',model=name,steps=step+1,total_steps=conf['steps']))
        with torch.no_grad():
            scores={s:metrics(head,s) for s in data}
            for d in data.values():
                projected=project(head,d['features'],d['raw_actor']);active=active_mask(d['features'])
                assert torch.equal(projected[~active],d['raw_actor'][~active])
                ids=torch.tensor([0,1,*range(5,81)],device='cuda',dtype=torch.long)
                assert torch.equal(projected[:,ids],d['raw_actor'][:,ids])
                non_mg=d['features'][:,1125]!=1
                assert torch.equal(projected[non_mg,4],d['raw_actor'][non_mg,4])
                if name=='gaze_only':assert torch.equal(projected[:,4],d['raw_actor'][:,4])
        target=a.out/name;target.mkdir();torch.save(dict(version='combat_memory_recovery_fit_v1',prototype_version=VERSION,state_dict=head.state_dict(),
            optimizer=optimizer.state_dict(),steps=conf['steps'],model=name,source_weights_sha256=p['source_weights_sha256'],
            runtime_admitted=False,role='Standalone experimental head; not policy or PPO checkpoint'),target/'checkpoint.pt')
        reloaded=torch.load(target/'checkpoint.pt',map_location='cuda',weights_only=True)
        assert all(torch.equal(v,reloaded['state_dict'][k]) for k,v in head.state_dict().items())
        dev_score=scores['development'];gate=p['offline_probe_admission']
        ratios=dict(to_source=dev_score['gaze_rmse_degrees']/baselines['development']['gaze_rmse_degrees'],
            to_zero_command=dev_score['gaze_rmse_degrees']/baselines['development']['zero_command_gaze_rmse_degrees'],
            negative_fire_probability=dev_score['negative_fire_probability']/baselines['development']['negative_fire_probability'])
        admitted=ratios['to_source']<=gate['development_gaze_rmse_ratio_to_source_max'] and ratios['to_zero_command']<=gate['development_gaze_rmse_ratio_to_zero_command_max']
        if name=='gaze_and_mg_fire':admitted &= ratios['negative_fire_probability']<=gate['joint_development_negative_fire_probability_ratio_to_source_max']
        result=dict(state='complete',device='cuda',model=name,steps=conf['steps'],parameters=PARAMETERS,metrics=scores,development_ratios=ratios,
            offline_probe_admitted=bool(admitted),checkpoint_sha256=sha(target/'checkpoint.pt'),gpu_reload_exact=True,
            outside_gate_and_output_scope_exact=True,source_weights_untouched=True,runtime_action_admitted=False,promotion=False)
        save(target/'report.json',result);save(target/'complete.json',dict(state='complete',report_sha256=sha(target/'report.json')));results[name]=result
    for path,h in live.items():assert sha(path)==h,path
    for path,h in pins.items():assert sha(path)==h,path
    coverage={s:dict(rows=len(d['features']),gaze_rows=int(masks(d)[0].sum()),fire_rows=int(masks(d)[1].sum()),
        gaze_worlds=int(d['world'][masks(d)[0]].unique().numel()),fire_worlds=int(d['world'][masks(d)[1]].unique().numel())) for s,d in data.items()}
    report=dict(state='complete',version='combat_memory_recovery_training_v1',device='cuda',results=results,baselines=baselines,coverage=coverage,
        inputs=pins,source_weights_sha256=p['source_weights_sha256'],source_cache_report_sha256=sha(root/'report.json'),
        training_performed=True,source_model_training=False,source_parameters_untouched=True,policy_weights_written=False,
        runtime_action_admitted=False,promotion=False,
        scope='Fixed standalone head endpoints, held-out whole-world direction and negative-only MG fire checks. No Go/runtime integration, new battles, restored-contact or combat-superiority claim.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    save(a.out/'progress.json',dict(stage='complete',models=len(results)))
    print(dict(baselines=baselines,coverage=coverage,results={name:dict(metrics=r['metrics'],ratios=r['development_ratios'],offline_probe_admitted=r['offline_probe_admitted']) for name,r in results.items()}))


if __name__=='__main__':main()

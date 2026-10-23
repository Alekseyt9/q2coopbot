"""Fixed exploratory movement selector fit with explicit per-enemy relative geometry."""
import argparse
import json
from pathlib import Path
import shutil

from evaluate_registered_combat_models import read,save,sha
from ppo_combat import torch,nn,training_devices
from ppo_recurrent import prepare_context,sequence
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model,JointAim
from combat_movement_geometry_selector import geometry_features,GeometryMovementSelector
from fork_conditional_weapon_exploration_cuda import equal_tree


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','source','protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}

    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or expected==h,str(path)
        pins[str(path)]=h;return path

    p=read(pin(a.protocol));assert p['version']=='combat_movement_geometry_selector_fit_v1' and p['steps']==500
    ds=read(pin(a.dataset/'report.json',p['dataset_report_sha256']));assert ds['gate_passed']
    records=read(pin(a.dataset/'records.json',p['records_sha256']))['records']
    for path,h in ds['inputs'].items():pin(path,h)
    motor_report_path=pin(p['motor_audit']);motor_audit=read(motor_report_path)
    assert motor_audit['feature_admission_passed'] and motor_audit['device']=='cuda' and motor_audit['split']=='train_only'
    motor_seal=read(pin(motor_report_path.with_name('complete.json')));assert motor_seal['report_sha256']==sha(motor_report_path)
    ss=read(pin(a.source/'complete.json'));model=read(pin(a.source/'weights.json',ss['weights_sha256']))
    assert ss['weights_sha256']==p['source_weights_sha256']==ds['source_weights_sha256']
    pin(a.source/'checkpoint.pt',ss['checkpoint_sha256'])
    pin(Path(__file__).with_name('combat_movement_geometry_selector.py'))
    candidates=p['candidates'];assert candidates==['control','left','right','forward','backward','near_stationary']
    train=[i for i,r in enumerate(records) if r['split']=='train'];dev=[i for i,r in enumerate(records) if r['split']=='development']
    assert len(train)==14 and len(dev)==4 and not {records[i]['seed'] for i in train}&{records[i]['seed'] for i in dev}
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    actor=branch(base_model(model,allow_shared=True),'actor').requires_grad_(False)
    joint=JointAim(model,allow_shared=True).requires_grad_(False)
    before_actor={k:v.detach().clone() for k,v in actor.state_dict().items()}
    before_joint={k:v.detach().clone() for k,v in joint.state_dict().items()}
    a.out.mkdir(parents=True);shutil.copy2(__file__,a.out/Path(__file__).name);shutil.copy2(a.protocol,a.out/'protocol.json')
    shutil.copy2(Path(__file__).with_name('combat_movement_geometry_selector.py'),a.out/'combat_movement_geometry_selector.py')
    features=[];raw_outputs=[];samples=[];memory_errors=[]
    for i,r in enumerate(records):
        save(a.out/'progress.json',dict(stage='cuda-source-context',context=i,contexts=len(records)))
        c=r['context'];directory=Path(c['directory'])/'cuda'
        meta=read(pin(directory/'report.json'));assert meta['observation_timing_contract']=='frozen_world_v1'
        rows=list(map(json.loads,pin(directory/'rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig')))
        context=list(map(json.loads,pin(directory/'sequence.jsonl',meta['sequence_sha256']).open(encoding='utf-8-sig')))
        with torch.no_grad():
            raw,error=sequence(actor,prepare_context(context,rows,'cuda'),32,True,'actor')
        assert error<2e-5;memory_errors.append(error)
        at=next(j for j,q in enumerate(rows) if q['frame']==c['frame']);assert rows[at]['seed']==r['seed']
        features.append(rows[at]['features']);raw_outputs.append(raw[at].detach().clone());samples.append(rows[at]['sample'])
    n=len(records);x=torch.tensor(features,device='cuda');raw=torch.stack(raw_outputs)
    cat=lambda key:torch.tensor([s.get(key,0) for s in samples],device='cuda',dtype=torch.long)
    timing=torch.tensor([s['aim_candidate']['timing'] for s in samples],device='cuda')
    valid=torch.zeros((n,6),device='cuda',dtype=torch.bool)
    deltas=torch.zeros((n,6,2),device='cuda');utility=torch.zeros((n,6),device='cuda')
    useful=torch.zeros_like(utility);harmful=torch.zeros_like(utility)
    for i,r in enumerate(records):
        zero=next(v for v in r['labels'] if v['label']=='control')
        for v in r['labels']:
            j=candidates.index(v['label']);valid[i,j]=True;deltas[i,j]=torch.tensor(v['mean_shift'],device='cuda')
            utility[i,j]=4*float(v['goal'] and not v['death'])+.25*v['kills']-v['received']/100.-float(v['death'])
            useful[i,j]=float(v['useful'])
            harmful[i,j]=float((zero['goal'] and not zero['death'] and not(v['goal'] and not v['death'])) or
                (zero['goal'] and v['goal'] and not zero['death'] and not v['death'] and v['kills']==zero['kills'] and v['received']-zero['received']>=5))
    ti,di=(torch.tensor(ids,device='cuda') for ids in (train,dev))
    with torch.no_grad():
        tokens,wall,present,motor=geometry_features(joint,actor,x,raw,cat('target'),cat('aim_mode'),cat('vertical'),cat('weapon'),timing,deltas)
        token_mask=valid[ti,:,None]&present[ti,None,:]
        pool=tokens[ti][token_mask];token_mean=pool.mean(0);token_std=pool.std(0,unbiased=False).clamp_min(.1)
        pool=wall[ti][valid[ti]];wall_mean=pool.mean(0);wall_std=pool.std(0,unbiased=False).clamp_min(.1)
        normalized_tokens=((tokens-token_mean)/token_std).clamp(-10,10).detach()
        normalized_wall=((wall-wall_mean)/wall_std).clamp(-10,10).detach()
        assert bool(torch.isfinite(normalized_tokens).all()) and bool(torch.isfinite(normalized_wall).all())
    pair_mask=valid[:,:,None]&valid[:,None,:]&(utility[:,:,None]-utility[:,None,:]>=.05)
    pair_count=pair_mask.sum((1,2));results=[]
    for name in p['architectures']:
        torch.manual_seed(20261023)
        scorer=GeometryMovementSelector('mlp' in name)
        optimizer=torch.optim.Adam(scorer.parameters(),lr=.003,weight_decay=.0001)

        def probs():
            scores=scorer(normalized_tokens,normalized_wall,present)
            logp=scores.masked_fill(~valid,-torch.inf).log_softmax(-1)
            return scores,logp,logp.exp()

        def loss():
            scores,logp,_=probs();pair=nn.functional.softplus(scores[:,None,:]-scores[:,:,None])
            per=(pair*pair_mask).sum((1,2))/pair_count.clamp_min(1)
            return per[ti[pair_count[ti]>0]].mean()+.05*(-logp[ti,0]).mean()

        start=float(loss().detach())
        save(a.out/'progress.json',dict(stage='cuda-fitting',model=name,steps=p['steps']))
        for step in range(p['steps']):
            optimizer.zero_grad(set_to_none=True);objective=loss();assert bool(torch.isfinite(objective))
            objective.backward();nn.utils.clip_grad_norm_(scorer.parameters(),.5,error_if_nonfinite=True);optimizer.step()
        with torch.no_grad():
            scores,_,probability=probs();constant=probability[ti].mean(0)[None,:]*valid
            constant/=constant.sum(-1,keepdim=True);delta=utility-utility[:,:1]

            def metrics(ids,q):
                return dict(useful_mass=float((q[ids]*useful[ids]).sum(-1).mean()),harmful_mass=float((q[ids]*harmful[ids]).sum(-1).mean()),
                    observed_utility_delta=float((q[ids]*delta[ids]).sum(-1).mean()),control_mass=float(q[ids,0].mean()))

            development,control_dev=metrics(di,probability),metrics(di,constant);g=p['gate']
            gain=((probability[di]-constant[di])*useful[di]).sum(-1)
            gates=dict(useful=development['useful_mass']>=control_dev['useful_mass']+g['useful_gain'],
                harmful=development['harmful_mass']<=control_dev['harmful_mass']+g['harmful_excess_max'],
                utility=development['observed_utility_delta']>=control_dev['observed_utility_delta']+g['observed_utility_gain'],
                contexts=int((gain>0).sum())>=g['minimum_useful_gain_contexts'])
            folder=a.out/name;folder.mkdir()
            exported=scorer.export()
            save(folder/'head.json',dict(version='combat_movement_geometry_selector_offline_v1',source_weights_sha256=p['source_weights_sha256'],
                candidates=candidates,token_mean=token_mean.tolist(),token_std=token_std.tolist(),wall_mean=wall_mean.tolist(),wall_std=wall_std.tolist(),
                input_clip=10,score_clip=20,scorer=exported,ordinary_policy_export=False,runtime_implemented=False))
            # Verify the serialized head on CUDA; no resume or ordinary policy export is implied.
            recovered=GeometryMovementSelector('mlp' in name)
            for module,spec_layer in zip([m for m in recovered.enemy if isinstance(m,nn.Linear)],exported['enemy_layers']):
                module.weight.copy_(torch.tensor(spec_layer['weight'],device='cuda'));module.bias.copy_(torch.tensor(spec_layer['bias'],device='cuda'))
            recovered.wall.weight.copy_(torch.tensor([exported['wall_weight']],device='cuda'));recovered.beta.copy_(torch.tensor(exported['beta'],device='cuda'))
            roundtrip_error=float((recovered(normalized_tokens,normalized_wall,present)-scores).abs().max());assert roundtrip_error==0
            save(folder/'probabilities.json',[dict(seed=r['seed'],split=r['split'],context_id=r['context']['context_id'],
                probabilities=probability[i].tolist(),constant=constant[i].tolist(),available=valid[i].tolist()) for i,r in enumerate(records)])
            results.append(dict(name=name,parameters=sum(v.numel() for v in scorer.parameters()),steps=p['steps'],
                loss_before=start,loss_after=float(loss()),train=metrics(ti,probability),development=development,constant_development=control_dev,
                gates=gates,gate_passed=all(gates.values()),roundtrip_error=roundtrip_error,head_sha256=sha(folder/'head.json')))
    assert equal_tree(actor.state_dict(),before_actor) and equal_tree(joint.state_dict(),before_joint)
    for path,h in pins.items():assert sha(path)==h,path
    passed=sorted([r for r in results if r['gate_passed']],key=lambda r:(-(r['development']['useful_mass']-r['constant_development']['useful_mass']),r['development']['harmful_mass'],r['parameters']))
    report=dict(state='complete',version='combat_movement_geometry_selector_fit_v1',device='cuda',results=results,selected=passed[0]['name'] if passed else None,
        gate_passed=bool(passed),train_contexts=len(train),development_contexts=len(dev),development_reused_exploratory=True,
        source_neural_parameters_exact=True,max_memory_error=max(memory_errors),inputs=pins,script_sha256=sha(__file__),
        fitting_performed=True,new_games=False,ordinary_policy_export=False,runtime_implemented=False,promotion=False,
        scope='Exploratory observed-geometry representation iteration on reused internal development. Mean-action forecasts, not expected Q or integrated future outcomes. Fresh native quality and versioned CUDA/native input/likelihood acceptance still required.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    save(a.out/'progress.json',dict(stage='complete',selected=report['selected'],gate_passed=report['gate_passed']))
    print(dict(selected=report['selected'],results=results))


if __name__=='__main__':main()

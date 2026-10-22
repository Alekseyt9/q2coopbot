"""Offline input-information probe, not a controller or a JEPA implementation."""
import argparse
import collections
import copy
import gzip
import json
import pathlib
from ppo_recurrent import torch,nn,prepare_context,sequence,durable_write
from migrate_combat_navigation_cuda import branch
from combat_sampled_motion_aim import inputs
from process_combat_architecture_pool import read,save,sha


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--labels',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True)
    ap.add_argument('--steps',type=int,default=500)
    args=ap.parse_args();assert torch.cuda.is_available() and not args.out.exists() and 1<=args.steps<=2000
    torch.set_default_device('cuda');torch.set_num_threads(2);torch.manual_seed(20261021)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    pins={}
    def pin(path,expected=None):
        path=pathlib.Path(path);digest=sha(path);assert expected is None or digest==expected
        pins[str(path)]=digest;return path
    meta=read(pin(args.labels/'report.json'));assert meta['state']=='complete'
    corpus=pathlib.Path(meta['corpus']);protocol=read(pin(corpus/'protocol.json'))
    source=pathlib.Path(protocol['source']);seal=read(pin(source/'complete.json'))
    model=read(pin(source/'weights.json',seal['weights_sha256']))
    assert model['sampled_motion_aim']['version']=='combat_sampled_motion_aim_v1'
    actor=branch(model,'actor')
    for p in actor.parameters():p.requires_grad_(False)
    data={};used_seeds=set()
    with torch.no_grad():
        for split,binding in meta['splits'].items():
            assert split in ('train','development')
            with gzip.open(pin(binding['output'],binding['sha256']),'rt',encoding='utf-8') as stream:labels=list(map(json.loads,stream))
            root=corpus/('processing-'+split);report=read(pin(root/'report.json'))
            merged=root/'merged';info=read(pin(merged/'report.json'));assert info['model_sha256']==seal['weights_sha256']
            streams={}
            for name in ('rollout','sequence'):
                path=pin(merged/(name+'.jsonl'),info[name+'_sha256'])
                streams[name]=[json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()]
            rows=streams['rollout'];seeds={r['seed'] for r in rows};assert not used_seeds.intersection(seeds);used_seeds.update(seeds)
            positions={(r['seed'],r['index']):i for i,r in enumerate(rows)}
            raw,error=sequence(actor,prepare_context(streams['sequence'],rows,'cuda'),check_states=True);assert error<2e-5
            needed=collections.defaultdict(set)
            for label in labels:needed[label['seed']].add(label['index'])
            native={}
            for case in report['cases']:
                path=pin(root/f"case-{case['job']:04d}/native/replay-0/steps.jsonl")
                with path.open(encoding='utf-8-sig') as stream:
                    for step in map(json.loads,stream):
                        if step['index'] in needed[case['seed']]:
                            key=(case['seed'],step['index']);assert key not in native;native[key]=step
            ix=torch.tensor([positions[(l['seed'],l['index'])] for l in labels],dtype=torch.long,device='cuda')
            x=torch.tensor([l['features'] for l in labels],dtype=torch.float32,device='cuda')
            movement=torch.tensor([l['sample']['latent'][:2] for l in labels],dtype=torch.float32,device='cuda')
            target=torch.tensor([l['sample']['target'] for l in labels],dtype=torch.long,device='cuda')
            mode=torch.tensor([l['sample'].get('aim_mode',0) for l in labels],dtype=torch.long,device='cuda')
            vertical=torch.tensor([l['sample']['vertical'] for l in labels],dtype=torch.long,device='cuda')
            base=inputs(x,raw[ix],movement,target,mode);assert base.shape==(len(labels),125)
            onehot=nn.functional.one_hot(vertical,3).float()
            y=[];strata=[]
            for label in labels:
                step=native[(label['seed'],label['index'])]
                assert step['sample']==label['sample'] and step['observation']['identity']['frame']==label['frame']
                y.append(label['offline_label']['native_muzzle'][2]-step['observation']['position'][2])
                action=step['applied_action']['vertical'];o=step['observation']
                strata.append(action+('_ground' if o['on_ground'] else '_air'))
            data[split]=dict(baseline=torch.cat((base,torch.zeros_like(onehot)),1),
                vertical=torch.cat((base,onehot),1),y=torch.tensor(y,dtype=torch.float32,device='cuda')[:,None]/32,
                strata=strata,samples=len(labels))
    assert set(data)=={'train','development'}
    first=nn.Sequential(nn.Linear(128,32,device='cuda'),nn.ReLU(),nn.Linear(32,16,device='cuda'),nn.ReLU(),nn.Linear(16,1,device='cuda'))
    with torch.no_grad():first[0].weight[:,125:].zero_()
    models=dict(baseline=first,vertical=copy.deepcopy(first))
    def metrics(module,key,d):
        error=(module(d[key])-d['y'])*32
        result=dict(samples=d['samples'],rmse_units=float(error.square().mean().sqrt()),mae_units=float(error.abs().mean()))
        result['strata']={}
        for group in sorted(set(d['strata'])):
            mask=torch.tensor([g==group for g in d['strata']],device='cuda')
            result['strata'][group]=dict(samples=int(mask.sum()),rmse_units=float(error[mask].square().mean().sqrt()))
        return result
    with torch.no_grad():
        torch.testing.assert_close(first(data['train']['baseline']),models['vertical'](data['train']['vertical']),rtol=0,atol=1e-6)
    args.out.mkdir(parents=True);results={};states={}
    for name,module in models.items():
        with torch.no_grad():before=metrics(module,name,data['development'])
        opt=torch.optim.Adam(module.parameters(),lr=.003)
        for _ in range(args.steps):
            opt.zero_grad(set_to_none=True);loss=(module(data['train'][name])-data['train']['y']).square().mean()
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(module.parameters(),5.);opt.step()
        with torch.no_grad():results[name]=dict(initial_development=before,train=metrics(module,name,data['train']),development=metrics(module,name,data['development']))
        states[name]=dict(model=module.state_dict(),optimizer=opt.state_dict())
    assert torch.equal(models['baseline'][0].weight[:,125:],torch.zeros_like(models['baseline'][0].weight[:,125:]))
    durable_write(args.out/'probe.pt',lambda f:torch.save(states,f))
    for path,digest in pins.items():assert sha(path)==digest
    save(args.out/'report.json',dict(state='complete',version='vertical_actuator_information_probe_v1',device='cuda',
        gpu=torch.cuda.get_device_name(0),steps=args.steps,learning_rate=.003,fixed_budget=True,selection_on_development=False,
        results=results,inputs=[dict(path=p,sha256=h) for p,h in pins.items()],checkpoint_sha256=sha(args.out/'probe.pt'),script_sha256=sha(__file__),
        controller_training_performed=False,promotion=False,
        scope='Two same-initialization 128-32-16-1 predictors of native muzzle-Z minus pre-command player-origin-Z. Baseline sees exact v1 125 inputs with three zero channels; candidate adds selected vertical one-hot. Native pose is a label only. Independent seed development, no NN controller weights or actions changed; not a JEPA or gameplay improvement claim.'))
    print(json.dumps(results))


if __name__=='__main__':main()
